"""PyPI commit-era pinner (ported from Lazarus).

The single biggest reason a stale repo won't even *install* is that ``pip`` now
resolves its unpinned dependencies to versions that did not exist when the repo
last worked. This pinner reconstructs the dependency universe *as it was* on a
target date: for each package, pick the newest release whose earliest file
upload time was on or before that date.

The network-free core is :func:`select_version`; :func:`fetch_release_history`
wraps the PyPI JSON API with an on-disk cache and retry/backoff, so repeated
runs are free and transient failures are absorbed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from sanjeevini.pinners import cache_root

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
NOT_FOUND = "NOT_FOUND"

_TIMEOUT_S = 10.0
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5


@dataclass(frozen=True)
class ReleaseInfo:
    """One released version of a package and when it first appeared.

    Attributes:
        version: The release string as PyPI reports it (e.g. ``"1.18.0"``).
        uploaded: Earliest file upload time for the release, in UTC.
        yanked: Whether every file in the release is yanked.
    """

    version: str
    uploaded: _dt.datetime
    yanked: bool = False


def as_cutoff(when: _dt.date | _dt.datetime) -> _dt.datetime:
    """Normalise a date/datetime to an inclusive UTC cutoff instant.

    A bare :class:`datetime.date` becomes end-of-day UTC so a release published
    on the cutoff date itself still counts as "on or before".

    Args:
        when: The target date or datetime.

    Returns:
        A timezone-aware UTC datetime.
    """
    if isinstance(when, _dt.datetime):
        dt = when
    else:
        dt = _dt.datetime.combine(when, _dt.time(23, 59, 59))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def _parse_upload_time(files: list[dict[str, Any]]) -> _dt.datetime | None:
    """Return the earliest upload time across a release's files (UTC), or None."""
    times: list[_dt.datetime] = []
    for f in files:
        raw = f.get("upload_time_iso_8601") or f.get("upload_time")
        if not raw:
            continue
        raw = raw.replace("Z", "+00:00")
        try:
            dt = _dt.datetime.fromisoformat(raw)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        times.append(dt.astimezone(_dt.timezone.utc))
    return min(times) if times else None


def _parse_releases(data: dict[str, Any]) -> list[ReleaseInfo]:
    """Convert a decoded PyPI JSON payload into :class:`ReleaseInfo` records."""
    out: list[ReleaseInfo] = []
    for version, files in (data.get("releases") or {}).items():
        if not files:
            continue
        uploaded = _parse_upload_time(files)
        if uploaded is None:
            continue
        yanked = all(bool(f.get("yanked")) for f in files)
        out.append(ReleaseInfo(version=version, uploaded=uploaded, yanked=yanked))
    return out


def _cache_path(cache_dir: Path, package: str, date: _dt.date) -> Path:
    """Return the cache file path for a package/date pair."""
    safe = package.replace("/", "_").lower()
    return cache_dir / f"{safe}_{date.isoformat()}.json"


def fetch_release_history(
    package: str,
    date: _dt.date,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
    _sleep: Callable[[float], object] = time.sleep,
) -> list[ReleaseInfo] | None:
    """Fetch every released version of ``package`` from PyPI, with caching.

    The raw PyPI payload is cached under ``cache_dir`` keyed by package + date so
    repeated runs are free. Transient network errors are retried with
    exponential backoff; a 404 is definitive and returns ``None`` immediately.

    Args:
        package: The distribution name to query.
        date: Target date (used only as the cache key).
        cache_dir: Directory for cached payloads; defaults to
            ``<cache_root>/pypi``.
        client: An :class:`httpx.Client` to reuse; one is created if omitted.
        _sleep: Sleep function, injected for tests.

    Returns:
        The parsed release history, or ``None`` if the package does not exist on
        PyPI.

    Raises:
        httpx.HTTPError: If the request keeps failing after all retries.
    """
    cache_dir = cache_dir or (cache_root() / "pypi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, package, date)

    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("not_found"):
            return None
        return _parse_releases(cached)

    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    url = PYPI_JSON_URL.format(package=package)
    try:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                last_exc = exc
                _sleep(_BACKOFF_BASE_S * (2**attempt))
                continue
            if resp.status_code == 404:
                cache_file.write_text(json.dumps({"not_found": True}), encoding="utf-8")
                return None
            if resp.status_code >= 500:
                last_exc = httpx.HTTPError(f"server error {resp.status_code}")
                _sleep(_BACKOFF_BASE_S * (2**attempt))
                continue
            data = resp.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return _parse_releases(data)
        raise last_exc if last_exc else httpx.HTTPError(f"failed to fetch {package}")
    finally:
        if owns_client:
            client.close()


def select_version(
    releases: Iterable[ReleaseInfo],
    cutoff: _dt.date | _dt.datetime,
    *,
    allow_prerelease: bool = False,
    allow_yanked: bool = False,
) -> str | None:
    """Return the newest release at or before ``cutoff``. Pure, network-free.

    PEP 440 ordering is used, so ``1.10`` correctly outranks ``1.9``.

    Args:
        releases: Candidate releases.
        cutoff: The target date/datetime (inclusive).
        allow_prerelease: Whether to consider prereleases.
        allow_yanked: Whether to consider fully-yanked releases.

    Returns:
        The chosen version string, or ``None`` if nothing qualifies.
    """
    cutoff_dt = as_cutoff(cutoff)
    best_ver: Version | None = None
    best_raw: str | None = None
    for r in releases:
        if r.uploaded > cutoff_dt:
            continue
        if r.yanked and not allow_yanked:
            continue
        try:
            v = Version(r.version)
        except InvalidVersion:
            continue
        if v.is_prerelease and not allow_prerelease:
            continue
        if best_ver is None or v > best_ver:
            best_ver, best_raw = v, r.version
    return best_raw


def _upload_date_of(releases: Iterable[ReleaseInfo], version: str) -> str | None:
    """Return the ISO date a specific version was uploaded, or None."""
    for r in releases:
        if r.version == version:
            return r.uploaded.date().isoformat()
    return None


def resolve_pypi(
    packages: list[str],
    date: _dt.date,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
    allow_prerelease: bool = False,
) -> list[dict[str, Any]]:
    """Resolve packages to full records including the upload date.

    Args:
        packages: Distribution names to pin.
        date: Target date; each package is pinned to its newest release on or
            before this date.
        cache_dir: Optional cache directory override.
        client: Optional shared :class:`httpx.Client`.
        allow_prerelease: Whether prereleases are eligible.

    Returns:
        A list of ``{"package", "version", "upload_date"}`` dicts, in input
        order. Unknown packages get version :data:`NOT_FOUND` and a ``None``
        upload date.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    records: list[dict[str, Any]] = []
    try:
        for pkg in packages:
            history = fetch_release_history(pkg, date, cache_dir=cache_dir, client=client)
            if history is None:
                records.append({"package": pkg, "version": NOT_FOUND, "upload_date": None})
                continue
            version = select_version(history, date, allow_prerelease=allow_prerelease)
            if version is None:
                records.append({"package": pkg, "version": NOT_FOUND, "upload_date": None})
                continue
            records.append(
                {
                    "package": pkg,
                    "version": version,
                    "upload_date": _upload_date_of(history, version),
                }
            )
    finally:
        if owns_client:
            client.close()
    return records


def pin_pypi(
    packages: list[str],
    date: _dt.date,
    emit_json: bool = False,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(package, version), ...]`` pinned to ``date``.

    Args:
        packages: Distribution names to pin.
        date: Target date.
        emit_json: Accepted for API symmetry; output formatting is the caller's
            responsibility (see :class:`PyPIPinner`).
        cache_dir: Optional cache directory override.
        client: Optional shared :class:`httpx.Client`.

    Returns:
        A list of ``(package, version)`` tuples in input order; unknown packages
        carry version :data:`NOT_FOUND`.
    """
    records = resolve_pypi(packages, date, cache_dir=cache_dir, client=client)
    return [(r["package"], r["version"]) for r in records]


class PyPIPinner:
    """CLI handler for ``jeeva pin`` (default PyPI mode)."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: Parsed argparse namespace with ``packages``, ``date`` and
                ``json`` attributes.
        """
        self.args = args

    def run(self) -> None:
        """Resolve the requested packages and print the result.

        Emits ``pip install package==version`` lines by default, or a JSON array
        of ``{package, version, upload_date}`` objects with ``--json``. Packages
        not found on PyPI are reported without failing the run.
        """
        date = _dt.date.fromisoformat(self.args.date)
        records = resolve_pypi(self.args.packages, date)
        if getattr(self.args, "json", False):
            print(json.dumps(records, indent=2))
            return
        for rec in records:
            if rec["version"] == NOT_FOUND:
                print(f"# {rec['package']}: NOT_FOUND on PyPI as of {date}", file=sys.stderr)
                continue
            print(f"pip install {rec['package']}=={rec['version']}")
