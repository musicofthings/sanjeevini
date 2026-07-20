"""R / Bioconductor era pinner (new in Sanjeevini).

Bioconductor pins differently from PyPI/conda: a *release* (e.g. 3.14) fixes an
R version and a coherent set of package versions all at once. Given a target
date, this module finds the most recent Bioconductor release available on that
date, then reads that release's package versions from its ``VIEWS`` file (a DCF
document) and emits an R install script.

The release calendar is hardcoded from
https://bioconductor.org/about/release-announcements/ rather than scraped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from sanjeevini.pinners import cache_root

VIEWS_URL = "https://bioconductor.org/packages/{release}/bioc/VIEWS"
_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class BiocRelease:
    """A single Bioconductor release: version, its R version, and release date."""

    bioc_version: str
    r_version: str
    release_date: _dt.date


# Hardcoded release calendar (bioconductor.org/about/release-announcements/).
BIOC_RELEASES: list[BiocRelease] = [
    BiocRelease("3.12", "4.0", _dt.date(2020, 10, 28)),
    BiocRelease("3.13", "4.1", _dt.date(2021, 5, 19)),
    BiocRelease("3.14", "4.1", _dt.date(2021, 10, 27)),
    BiocRelease("3.15", "4.2", _dt.date(2022, 4, 27)),
    BiocRelease("3.16", "4.2", _dt.date(2022, 11, 2)),
    BiocRelease("3.17", "4.3", _dt.date(2023, 4, 26)),
    BiocRelease("3.18", "4.3", _dt.date(2023, 10, 25)),
    BiocRelease("3.19", "4.4", _dt.date(2024, 5, 1)),
    BiocRelease("3.20", "4.4", _dt.date(2024, 10, 30)),
    BiocRelease("3.21", "4.5", _dt.date(2025, 4, 16)),
]


@dataclass
class BiocPinResult:
    """The result of pinning a package set to a Bioconductor release.

    Attributes:
        bioc_version: Chosen Bioconductor release (e.g. ``"3.14"``).
        r_version: R version that release ships with (e.g. ``"4.1"``).
        package_versions: ``(package, version)`` pairs resolved from VIEWS;
            packages absent from the release get version ``"NOT_IN_BIOC"``.
        install_script: A runnable R script installing the release and packages.
    """

    bioc_version: str
    r_version: str
    package_versions: list[tuple[str, str]] = field(default_factory=list)
    install_script: str = ""


def resolve_release(date: _dt.date) -> BiocRelease:
    """Return the most recent Bioconductor release available on ``date``.

    Args:
        date: Target date.

    Returns:
        The newest :class:`BiocRelease` whose release date is on or before
        ``date``.

    Raises:
        ValueError: If ``date`` predates the earliest known release.
    """
    eligible = [r for r in BIOC_RELEASES if r.release_date <= date]
    if not eligible:
        earliest = BIOC_RELEASES[0]
        raise ValueError(
            f"date {date} predates the earliest known Bioconductor release "
            f"({earliest.bioc_version}, {earliest.release_date})"
        )
    return max(eligible, key=lambda r: r.release_date)


def parse_views(text: str) -> dict[str, str]:
    """Parse a Bioconductor ``VIEWS`` DCF document into ``{package: version}``.

    DCF records are separated by blank lines; each ``Package:`` / ``Version:``
    pair within a record is captured. Continuation lines (leading whitespace)
    are ignored since only the two scalar fields are needed.

    Args:
        text: The raw VIEWS file contents.

    Returns:
        A mapping of package name to version string.
    """
    versions: dict[str, str] = {}
    current_pkg: str | None = None
    current_ver: str | None = None
    for line in text.splitlines():
        if not line.strip():
            if current_pkg and current_ver:
                versions[current_pkg] = current_ver
            current_pkg = current_ver = None
            continue
        if line[0].isspace():
            continue  # continuation of a multi-line field
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Package":
            current_pkg = value
        elif key == "Version":
            current_ver = value
    if current_pkg and current_ver:  # final record with no trailing blank line
        versions[current_pkg] = current_ver
    return versions


def fetch_views(
    release: str,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> dict[str, str]:
    """Fetch and cache a release's VIEWS file, returning ``{package: version}``.

    Args:
        release: Bioconductor version string (e.g. ``"3.14"``).
        cache_dir: Cache directory; defaults to ``<cache_root>/bioc``.
        client: Optional shared :class:`httpx.Client`.

    Returns:
        The parsed package/version mapping for the release.

    Raises:
        httpx.HTTPError: On a request failure.
    """
    cache_dir = cache_dir or (cache_root() / "bioc")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"VIEWS_{release}.dcf"

    if cache_file.exists():
        return parse_views(cache_file.read_text(encoding="utf-8"))

    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    try:
        resp = client.get(VIEWS_URL.format(release=release))
        resp.raise_for_status()
        text = resp.text
    finally:
        if owns_client:
            client.close()

    cache_file.write_text(text, encoding="utf-8")
    return parse_views(text)


def _build_install_script(bioc_version: str, package_versions: list[tuple[str, str]]) -> str:
    """Render an R install script for a release and its resolved packages."""
    found = [pkg for pkg, ver in package_versions if ver != "NOT_IN_BIOC"]
    missing = [pkg for pkg, ver in package_versions if ver == "NOT_IN_BIOC"]
    lines = [
        'if (!requireNamespace("BiocManager", quietly = TRUE))',
        '    install.packages("BiocManager")',
        f'BiocManager::install(version = "{bioc_version}")',
    ]
    if found:
        pkg_vec = ", ".join(f'"{p}"' for p in found)
        lines.append(f"BiocManager::install(c({pkg_vec}))")
    for pkg in missing:
        lines.append(f'install.packages("{pkg}")  # not in Bioc {bioc_version}; CRAN fallback')
    return "\n".join(lines) + "\n"


def pin_bioc(
    packages: list[str],
    date: _dt.date,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> BiocPinResult:
    """Pin a package set to the Bioconductor release live on ``date``.

    Args:
        packages: R/Bioconductor package names to pin.
        date: Target date.
        cache_dir: Optional cache directory override.
        client: Optional shared :class:`httpx.Client`.

    Returns:
        A :class:`BiocPinResult` with the release, R version, resolved package
        versions, and a runnable install script.
    """
    release = resolve_release(date)
    views = fetch_views(release.bioc_version, cache_dir=cache_dir, client=client)
    package_versions = [(pkg, views.get(pkg, "NOT_IN_BIOC")) for pkg in packages]
    script = _build_install_script(release.bioc_version, package_versions)
    return BiocPinResult(
        bioc_version=release.bioc_version,
        r_version=release.r_version,
        package_versions=package_versions,
        install_script=script,
    )


class BiocPinner:
    """CLI handler for ``jeeva pin --bioc``."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: Namespace with ``packages``, ``date`` and ``json`` attributes.
        """
        self.args = args

    def run(self) -> None:
        """Resolve packages to a Bioconductor release and print the result.

        Emits the R install script by default, or a JSON object with the release
        metadata, package versions and script with ``--json``.
        """
        import json

        date = _dt.date.fromisoformat(self.args.date)
        result = pin_bioc(self.args.packages, date)
        if getattr(self.args, "json", False):
            payload = {
                "bioc_version": result.bioc_version,
                "r_version": result.r_version,
                "package_versions": [
                    {"package": p, "version": v} for p, v in result.package_versions
                ],
                "install_script": result.install_script,
            }
            print(json.dumps(payload, indent=2))
            return
        print(result.install_script, end="")
