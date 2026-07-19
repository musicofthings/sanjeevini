"""R / Bioconductor repository Scout (new in Sanjeevini).

Detects R packages (a ``DESCRIPTION`` file) and, among them, Bioconductor
packages (a ``biocViews`` field), and produces an :class:`RResurrectionPlan`.

Resurrection differs from Python: the repair loop drives ``BiocManager::install()``
and ``renv::restore()`` inside a ``rocker`` image rather than ``pip install``.
The base image is chosen from the ``rocker/bioconductor`` line for released
Bioconductor versions, falling back to ``rocker/r-ver`` for plain-CRAN packages
or Bioconductor ≥ 3.20 (which has no dedicated rocker/bioconductor tag here).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sanjeevini.pinners.bioc import BIOC_RELEASES
from sanjeevini.scouts.python_scout import ensure_falsifiable
from sanjeevini.scouts.repo import RepoSnapshot, fetch_snapshot

Fetcher = Callable[[str], Awaitable[RepoSnapshot]]

# Bioconductor releases with a dedicated rocker/bioconductor image (PRD table).
_ROCKER_BIOC_MAX = (3, 19)


@dataclass
class RResurrectionPlan:
    """A Scout's plan for resurrecting an R/Bioconductor tool.

    Attributes:
        capability: What the package does, in 1-2 sentences.
        bioc_release: Bioconductor release (e.g. ``"3.14"``), or ``None`` for a
            pure-CRAN package.
        r_version: R version to run under (e.g. ``"4.1"``).
        base_image: rocker Docker image to start from.
        goal: Full goal statement for the repair loop.
        sanity_check: Falsifiable, measurable pass criterion.
        package_name: Package name from the DESCRIPTION ``Package`` field.
        depends: Dependencies from ``Depends``/``Imports`` (R itself excluded).
        known_issues: Issues surfaced from the repo.
    """

    capability: str
    bioc_release: str | None
    r_version: str
    base_image: str
    goal: str
    sanity_check: str
    package_name: str
    depends: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)


_R_DEP_RE = re.compile(r"\bR\s*\(\s*[>=]=?\s*([0-9]+\.[0-9]+)")


def parse_description(text: str) -> dict[str, str]:
    """Parse a DCF ``DESCRIPTION`` file into a field mapping.

    Handles continuation lines (leading whitespace continues the prior field),
    which ``Depends``/``Imports`` commonly use.

    Args:
        text: The DESCRIPTION file contents.

    Returns:
        A mapping of field name to its (whitespace-collapsed) value.
    """
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            if current:
                fields[current].append(line.strip())
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        current = key.strip()
        fields.setdefault(current, [])
        if value.strip():
            fields[current].append(value.strip())
    return {k: " ".join(v) for k, v in fields.items()}


def parse_depends(fields: dict[str, str]) -> tuple[str | None, list[str]]:
    """Return the required R version and the dependency list from DESCRIPTION.

    Args:
        fields: Parsed DESCRIPTION fields.

    Returns:
        A ``(r_version, depends)`` pair. ``r_version`` is ``None`` if no R
        constraint is declared; ``depends`` excludes R itself and drops version
        constraints.
    """
    combined = ", ".join(
        v for k, v in fields.items() if k in ("Depends", "Imports") and v
    )
    r_match = _R_DEP_RE.search(combined)
    r_version = r_match.group(1) if r_match else None

    depends: list[str] = []
    for token in combined.split(","):
        name = re.split(r"[\s(]", token.strip(), maxsplit=1)[0].strip()
        if name and name != "R":
            depends.append(name)
    # De-dupe, order-preserving.
    return r_version, list(dict.fromkeys(depends))


def resolve_bioc_release(r_version: str | None) -> str | None:
    """Map a required R version to the newest matching Bioconductor release.

    Args:
        r_version: A ``"major.minor"`` R version, or ``None``.

    Returns:
        The Bioconductor release string, or ``None`` if no release targets that
        R version.
    """
    if r_version is None:
        return None
    matches = [r for r in BIOC_RELEASES if r.r_version == r_version]
    if not matches:
        return None
    return max(matches, key=lambda r: r.release_date).bioc_version


def _release_r_version(bioc_release: str) -> str | None:
    """Return the R version shipped with a Bioconductor release, if known."""
    for r in BIOC_RELEASES:
        if r.bioc_version == bioc_release:
            return r.r_version
    return None


def select_rocker_image(bioc_release: str | None, r_version: str) -> str:
    """Select the rocker base image per the PRD table.

    Args:
        bioc_release: Bioconductor release, or ``None`` for plain CRAN.
        r_version: R version, used for the ``rocker/r-ver`` fallback.

    Returns:
        A rocker Docker image reference.
    """
    if bioc_release is None:
        return f"rocker/r-ver:{r_version}"
    major_minor = tuple(int(x) for x in bioc_release.split("."))
    if major_minor > _ROCKER_BIOC_MAX:
        return f"rocker/r-ver:{r_version}"
    return f"rocker/bioconductor:{bioc_release}"


def _capability(fields: dict[str, str], name: str) -> str:
    """Return a capability summary from Title/Description DESCRIPTION fields."""
    title = fields.get("Title", "").strip()
    description = fields.get("Description", "").strip()
    summary = f"{title}. {description}".strip(". ").strip()
    if summary:
        return summary[:280]
    return f"{name}: an R/Bioconductor package revived from source."


class RScout:
    """Reads an R/Bioconductor repo and writes a falsifiable :class:`RResurrectionPlan`."""

    def __init__(
        self,
        github_url: str,
        *,
        snapshot: RepoSnapshot | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        """Configure the Scout.

        Args:
            github_url: URL of the target repository.
            snapshot: A pre-built snapshot to plan from (skips fetching).
            fetcher: Async callable returning a :class:`RepoSnapshot`; defaults
                to fetching from GitHub.
        """
        self.github_url = github_url
        self._snapshot = snapshot
        self._fetcher: Fetcher = fetcher or fetch_snapshot

    async def plan(self, confirm: bool = True) -> RResurrectionPlan:
        """Read the repo, build the plan, and validate its sanity check.

        Args:
            confirm: If ``True``, print the plan and wait for acknowledgement.

        Returns:
            The R resurrection plan.

        Raises:
            ValueError: If the repo has no DESCRIPTION, or the generated sanity
                check is not falsifiable.
        """
        snapshot = self._snapshot or await self._fetcher(self.github_url)
        plan = self._build_plan(snapshot)
        ensure_falsifiable(plan.sanity_check)
        if confirm:
            self._confirm(plan)
        return plan

    def _build_plan(self, snapshot: RepoSnapshot) -> RResurrectionPlan:
        """Assemble an R plan from a snapshot (pure)."""
        description = snapshot.get("DESCRIPTION")
        if not description:
            raise ValueError(
                f"{snapshot.owner}/{snapshot.name} has no DESCRIPTION file; "
                "this is not an R package."
            )
        fields = parse_description(description)
        package_name = fields.get("Package", snapshot.name)
        is_bioc = "biocViews" in fields

        required_r, depends = parse_depends(fields)
        if is_bioc:
            bioc_release = resolve_bioc_release(required_r) or BIOC_RELEASES[-1].bioc_version
            r_version = required_r or (_release_r_version(bioc_release) or "4.4")
        else:
            bioc_release = None
            r_version = required_r or "4.4"

        base_image = select_rocker_image(bioc_release, r_version)
        capability = _capability(fields, package_name)
        sanity_check = self._sanity_check(package_name, is_bioc)
        known_issues = [t.strip() for t, _ in snapshot.open_issues if t.strip()]

        strategy = "BiocManager::install() + renv::restore()" if is_bioc else "renv::restore()"
        goal = (
            f"Resurrect {snapshot.owner}/{snapshot.name} "
            f"({'Bioconductor ' + bioc_release if bioc_release else 'CRAN'}, R {r_version}) "
            f"using {strategy}. {capability} Success criterion: {sanity_check}"
        )
        return RResurrectionPlan(
            capability=capability,
            bioc_release=bioc_release,
            r_version=r_version,
            base_image=base_image,
            goal=goal,
            sanity_check=sanity_check,
            package_name=package_name,
            depends=depends,
            known_issues=known_issues,
        )

    @staticmethod
    def _sanity_check(package_name: str, is_bioc: bool) -> str:
        """Return a falsifiable sanity check for an R package."""
        extra = (
            " and the package's first man/example runs and returns a non-empty result"
            if is_bioc
            else " and the first example in the man pages runs without error and "
            "returns a non-empty value"
        )
        return (
            f"`R CMD check` reports 0 ERRORs; `library({package_name})` loads"
            f"{extra}."
        )

    def _confirm(self, plan: RResurrectionPlan) -> None:
        """Print the plan and pause for user acknowledgement."""
        print("── R Resurrection Plan ───────────────────────────────────")
        print(f"repo         : {self.github_url}")
        print(f"package      : {plan.package_name}")
        print(f"capability   : {plan.capability}")
        release = plan.bioc_release or "CRAN"
        print(f"bioc release : {release} (R {plan.r_version})")
        print(f"base image   : {plan.base_image}")
        print(f"sanity check : {plan.sanity_check}")
        with contextlib.suppress(EOFError):
            input("Press Enter to proceed (Ctrl-C to abort)…")
