"""Python repository Scout (ported from Lazarus).

Given a GitHub URL, the Scout reads the repo (and, when cited, its paper) and
produces a :class:`PythonResurrectionPlan`: the capability to revive, a base
Docker image, and — most importantly — a *falsifiable* sanity check.

This is the highest-leverage organ in the system: the plan's quality determines
everything downstream. The scientific-correctness principle from Lazarus is
non-negotiable here — a sanity check that only proves the code *ran* is not a
sanity check. :func:`ensure_falsifiable` enforces this, and :meth:`PythonScout.plan`
rejects any plan whose sanity check carries no measurable pass criterion.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sanjeevini.contracts.output_type import GENERIC_CHECK, infer_output_type
from sanjeevini.scouts.repo import RepoSnapshot, fetch_snapshot

Fetcher = Callable[[str], Awaitable[RepoSnapshot]]


@dataclass
class PythonResurrectionPlan:
    """A Scout's plan for resurrecting a Python research tool.

    Attributes:
        capability: What the tool does, in 1-2 sentences.
        base_image: Docker image to start the resurrection from.
        goal: Full goal statement handed to the repair loop.
        sanity_check: Falsifiable, measurable pass criterion.
        test_input: Description of the input to exercise the tool with.
        known_issues: Issues surfaced from the repo (open issues, README warnings).
        paper_doi: DOI or arXiv id of an associated paper, if found.
        estimated_turns: Scout's estimate of repair complexity, in loop turns.
        framework: Canonical framework label (e.g. ``"tensorflow-1.x"``).
        python_version: Target Python version (e.g. ``"3.6"``).
    """

    capability: str
    base_image: str
    goal: str
    sanity_check: str
    test_input: str
    known_issues: list[str] = field(default_factory=list)
    paper_doi: str | None = None
    estimated_turns: int = 12
    framework: str = "plain-python"
    python_version: str = "3.10"


# ---------------------------------------------------------------------------
# Falsifiability guard (the non-negotiable scientific-correctness principle)
# ---------------------------------------------------------------------------

_MEASURABLE_RE = re.compile(
    r"(≥|≤|>=|<=|(?<![A-Za-z])[<>](?![A-Za-z])|\bat least\b|\bat most\b|\bno fewer\b|"
    r"non-?empty|"
    r"\d+\s*(kb|mb|gb|bytes?|records?|sequences?|variants?|reads?|rows?|lines?|poses?|structures?|models?)|"
    r"\d+\.\d+|"
    r"parse(s|d|able)?|valid(ates?|ated)?|quickcheck|bcftools|samtools|"
    r"\bwithin\b|\bauc\b|\brmsd\b|\bf1\b|\baccuracy\b|\bprecision\b|\brecall\b|\bdice\b|\biou\b|"
    r"0 errors?)",
    re.IGNORECASE,
)


def is_falsifiable(sanity_check: str) -> bool:
    """Return whether ``sanity_check`` carries a measurable pass criterion.

    A check is falsifiable if it references a threshold, a count/size, a metric,
    a structural validity test, or an explicit non-empty requirement — anything
    a run could concretely fail. "Runs without error" and the like are not.

    Args:
        sanity_check: The proposed sanity-check text.

    Returns:
        ``True`` if a measurable criterion is present.
    """
    return bool(_MEASURABLE_RE.search(sanity_check))


def ensure_falsifiable(sanity_check: str) -> str:
    """Return ``sanity_check`` unchanged, or raise if it is not falsifiable.

    Args:
        sanity_check: The proposed sanity-check text.

    Returns:
        The same string, when it passes.

    Raises:
        ValueError: If the check has no measurable threshold — e.g. it only
            asserts the tool runs without error.
    """
    if not is_falsifiable(sanity_check):
        raise ValueError(
            "sanity check is not falsifiable — it has no measurable pass "
            f"criterion (a run could not fail it): {sanity_check!r}"
        )
    return sanity_check


# ---------------------------------------------------------------------------
# Analysis helpers (pure, snapshot -> facts)
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_ARXIV_RE = re.compile(r"arxiv[:\s/]*(\d{4}\.\d{4,5})", re.IGNORECASE)
_TF_VER_RE = re.compile(r"tensorflow(?:-gpu)?\s*[=<>~!]*\s*(\d+)\.(\d+)", re.IGNORECASE)
_FROM_RE = re.compile(r"(?im)^\s*FROM\s+(\S+)")
_PY_CLASSIFIER_RE = re.compile(r"Python\s*::\s*(3\.\d+)")
_PY_REQUIRES_RE = re.compile(r"(?:python_requires|requires-python)\s*=?\s*['\"]([^'\"]+)['\"]")
_BENCH_RE = re.compile(
    r"\b(ROC[- ]?AUC|AUROC|AUC|accuracy|F1(?:[- ]score)?|precision|recall|RMSD|Dice|IoU)\b"
    r"[^\n.]{0,40}?(\d+\.\d+)",
    re.IGNORECASE,
)

_DEFAULT_PY = {
    "tensorflow-1.x": "3.6",
    "tensorflow-2.x": "3.9",
    "pytorch": "3.9",
    "jax": "3.10",
    "keras": "3.7",
    "plain-python": "3.10",
}
_DEFAULT_TURNS = {
    "tensorflow-1.x": 25,
    "tensorflow-2.x": 15,
    "pytorch": 18,
    "jax": 18,
    "keras": 20,
    "plain-python": 10,
}

# Non-genomic output types, still keyword-matched: these are structural claims
# about scientific artefacts that GenomicFileType does not model.
_STRUCTURE_CHECKS: list[tuple[str, str]] = [
    ("pdb", "the output PDB file is valid and contains ≥ 1 model with atomic coordinates"),
    ("structure", "the output structure file is valid and contains ≥ 1 model with coordinates"),
]


def detect_framework(deps_text: str) -> str:
    """Return the canonical framework label from dependency/README text.

    Args:
        deps_text: Concatenated requirements/setup/pyproject/README text.

    Returns:
        One of ``tensorflow-1.x``, ``tensorflow-2.x``, ``pytorch``, ``jax``,
        ``keras``, or ``plain-python``.
    """
    low = deps_text.lower()
    if "tensorflow" in low:
        m = _TF_VER_RE.search(deps_text)
        if m and int(m.group(1)) < 2:
            return "tensorflow-1.x"
        if m:
            return "tensorflow-2.x"
        # No pinned version; a mention of "1.x" / "tf 1" implies the legacy line.
        if re.search(r"\btensorflow\s*1\b|\btf\s*1\.", low):
            return "tensorflow-1.x"
        return "tensorflow-2.x"
    if "torch" in low or "pytorch" in low:
        return "pytorch"
    if "jax" in low or "flax" in low:
        return "jax"
    if "keras" in low:
        return "keras"
    return "plain-python"


def detect_python_version(deps_text: str, framework: str) -> str:
    """Return the target Python version from repo metadata, or a framework default.

    Args:
        deps_text: Concatenated setup/pyproject/README text.
        framework: The detected framework label (used for the fallback).

    Returns:
        A ``"3.x"`` version string.
    """
    classifiers: list[str] = _PY_CLASSIFIER_RE.findall(deps_text)
    if classifiers:
        return min(classifiers, key=lambda v: tuple(int(x) for x in v.split(".")))
    m = _PY_REQUIRES_RE.search(deps_text)
    if m:
        ver = re.search(r"3\.\d+", m.group(1))
        if ver:
            return ver.group(0)
    return _DEFAULT_PY.get(framework, "3.10")


def select_base_image(framework: str, python_version: str, deps_text: str, dockerfile: str) -> str:
    """Select a base Docker image via the Scout's heuristics.

    A repo's own Dockerfile ``FROM`` line wins; otherwise the framework dictates
    the image family.

    Args:
        framework: Detected framework label.
        python_version: Target Python version.
        deps_text: Dependency text (used to read a pinned TF minor version).
        dockerfile: Dockerfile contents, if any.

    Returns:
        A Docker image reference.
    """
    from_line = _FROM_RE.search(dockerfile)
    if from_line and from_line.group(1).lower() != "scratch":
        return from_line.group(1)

    match framework:
        case "tensorflow-1.x":
            m = _TF_VER_RE.search(deps_text)
            minor = m.group(2) if m else "15"
            return f"tensorflow/tensorflow:1.{minor}.5-gpu"
        case "tensorflow-2.x":
            return "tensorflow/tensorflow:2.13.0-gpu"
        case "pytorch":
            return "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
        case _:
            return f"python:{python_version}-slim"


def extract_doi(text: str) -> str | None:
    """Return a DOI or arXiv id found in ``text``, or ``None``.

    Args:
        text: Text to scan (typically the README).

    Returns:
        A bare DOI, an ``arXiv:XXXX.XXXXX`` id, or ``None``.
    """
    m = _DOI_RE.search(text)
    if m:
        return m.group(0).rstrip(".)")
    m = _ARXIV_RE.search(text)
    if m:
        return f"arXiv:{m.group(1)}"
    return None


def _extract_capability(readme: str, name: str) -> str:
    """Return a 1-2 sentence capability summary from the README."""
    for raw in readme.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!", "[", "<", "=", "-", "*", ">")):
            continue
        if line.startswith("```"):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        summary = " ".join(sentences[:2]).strip()
        return summary[:280]
    return f"{name}: a research tool revived from its source repository."


def generate_sanity_check(snapshot: RepoSnapshot) -> str:
    """Generate a falsifiable sanity check for a repo.

    Prefers a benchmark threshold quoted in the README or paper abstract; falls
    back to a structural check on the *inferred output type*; finally to a
    type-agnostic non-empty/size check. The result is always measurable.

    Output-type inference is evidence-weighted (see
    :mod:`sanjeevini.contracts.output_type`) rather than first-mention-wins. When
    the evidence is ambiguous it deliberately declines to name a type and returns
    the generic check — asserting the wrong format is worse than asserting less.

    Args:
        snapshot: The repository snapshot.

    Returns:
        A falsifiable sanity-check string.
    """
    corpus = "\n".join([snapshot.get("README.md", "README.rst"), snapshot.paper_abstract])
    bench = _BENCH_RE.search(corpus)
    if bench:
        metric, value = bench.group(1), bench.group(2)
        floor = max(0.0, float(value) - 0.05)
        return (
            f"Reproduce the reported {metric} of {value}: measured {metric} ≥ "
            f"{floor:.2f} on the benchmark test input."
        )

    profile = infer_output_type(corpus)
    if profile is not None:
        return f"On the test input, {profile.check}."

    low = corpus.lower()
    for keyword, check in _STRUCTURE_CHECKS:
        if keyword in low:
            return f"On the test input, {check}."

    return GENERIC_CHECK


# ---------------------------------------------------------------------------
# The Scout
# ---------------------------------------------------------------------------


class PythonScout:
    """Reads a Python repo and writes a falsifiable :class:`PythonResurrectionPlan`."""

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
            snapshot: A pre-built snapshot to plan from (skips fetching); mainly
                for tests and offline use.
            fetcher: Async callable returning a :class:`RepoSnapshot`; defaults
                to fetching from GitHub.
        """
        self.github_url = github_url
        self._snapshot = snapshot
        self._fetcher: Fetcher = fetcher or fetch_snapshot

    async def plan(self, confirm: bool = True) -> PythonResurrectionPlan:
        """Read the repo, build the plan, and validate its sanity check.

        Args:
            confirm: If ``True``, print the plan and wait for user acknowledgement
                before returning.

        Returns:
            The resurrection plan.

        Raises:
            ValueError: If the generated sanity check is not falsifiable.
        """
        snapshot = self._snapshot or await self._fetcher(self.github_url)
        plan = self._build_plan(snapshot)
        ensure_falsifiable(plan.sanity_check)
        if confirm:
            self._confirm(plan)
        return plan

    def _build_plan(self, snapshot: RepoSnapshot) -> PythonResurrectionPlan:
        """Assemble a plan from a snapshot (pure)."""
        readme = snapshot.get("README.md", "README.rst")
        deps_text = "\n".join(
            [
                snapshot.get("requirements.txt"),
                snapshot.get("setup.py"),
                snapshot.get("setup.cfg"),
                snapshot.get("pyproject.toml"),
                readme,
            ]
        )
        framework = detect_framework(deps_text)
        python_version = detect_python_version(deps_text, framework)
        base_image = select_base_image(
            framework, python_version, deps_text, snapshot.get("Dockerfile")
        )
        capability = _extract_capability(readme, snapshot.name)
        sanity_check = generate_sanity_check(snapshot)
        known_issues = self._collect_known_issues(snapshot, readme)
        estimated_turns = min(_DEFAULT_TURNS.get(framework, 12) + 2 * len(known_issues), 60)
        goal = (
            f"Resurrect {snapshot.owner}/{snapshot.name} "
            f"({framework}, Python {python_version}). {capability} "
            f"Success criterion: {sanity_check}"
        )
        return PythonResurrectionPlan(
            capability=capability,
            base_image=base_image,
            goal=goal,
            sanity_check=sanity_check,
            test_input=(
                "The smallest representative input bundled with the repo (or the "
                "minimal example described in the README)."
            ),
            known_issues=known_issues,
            paper_doi=extract_doi(readme),
            estimated_turns=estimated_turns,
            framework=framework,
            python_version=python_version,
        )

    @staticmethod
    def _collect_known_issues(snapshot: RepoSnapshot, readme: str) -> list[str]:
        """Gather known issues from open GitHub issues and README warnings."""
        issues = [title.strip() for title, _ in snapshot.open_issues if title.strip()]
        for raw in readme.splitlines():
            line = raw.strip()
            if re.search(
                r"\b(deprecat|no longer maintained|unmaintained|broken|"
                r"does not work|warning:)\b",
                line,
                re.IGNORECASE,
            ):
                issues.append(line[:200])
        return issues

    def _confirm(self, plan: PythonResurrectionPlan) -> None:
        """Print the plan and pause for user acknowledgement."""
        print("── Python Resurrection Plan ──────────────────────────────")
        print(f"repo         : {self.github_url}")
        print(f"capability   : {plan.capability}")
        print(f"framework    : {plan.framework} (Python {plan.python_version})")
        print(f"base image   : {plan.base_image}")
        print(f"sanity check : {plan.sanity_check}")
        print(f"est. turns   : {plan.estimated_turns}")
        if plan.known_issues:
            print("known issues :")
            for issue in plan.known_issues:
                print(f"  - {issue}")
        with contextlib.suppress(EOFError):
            input("Press Enter to proceed (Ctrl-C to abort)…")
