"""Conda / Bioconda commit-era pinner (new in Sanjeevini).

For each package, query the conda channel ``repodata.json`` archives to find the
most recent build published on or before the target date. Bioconda is checked
first (it holds the bioinformatics tools), then conda-forge, then defaults.

conda repodata records a ``timestamp`` in **milliseconds** since the Unix epoch;
:func:`select_conda_build` compares against the target date's end-of-day cutoff.
Repodata is large, so it is downloaded once per channel per day and cached
gzip-compressed under ``<cache_root>/conda``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from sanjeevini.pinners import cache_root

REPODATA_URL = "https://conda.anaconda.org/{channel}/{subdir}/repodata.json"
DEFAULT_CHANNELS = ["bioconda", "conda-forge", "defaults"]
DEFAULT_SUBDIR = "linux-64"

_TIMEOUT_S = 60.0


def _cutoff_ms(date: _dt.date) -> int:
    """Return the inclusive end-of-day UTC cutoff for ``date`` in epoch ms."""
    dt = _dt.datetime.combine(date, _dt.time(23, 59, 59), tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1000)


def select_conda_build(
    repodata: dict[str, Any], date: _dt.date, package: str
) -> tuple[str, str] | None:
    """Return the ``(version, build)`` of the newest ``package`` build on or before ``date``.

    Scans both the ``packages`` and ``packages.conda`` sections of a repodata
    document. Among all builds of ``package`` with a timestamp at or before the
    cutoff, the one with the highest timestamp wins; ties break on PEP 440
    version order.

    Args:
        repodata: A decoded ``repodata.json`` document.
        date: Target date (inclusive).
        package: The package name to resolve.

    Returns:
        The chosen ``(version, build)``, or ``None`` if no eligible build exists.
    """
    cutoff = _cutoff_ms(date)
    best: tuple[int, Version, str, str] | None = None  # (ts, version, version_raw, build)

    sections = (repodata.get("packages") or {}, repodata.get("packages.conda") or {})
    for section in sections:
        for entry in section.values():
            if entry.get("name") != package:
                continue
            ts = entry.get("timestamp", 0)
            if ts > cutoff:
                continue
            raw_version = str(entry.get("version", ""))
            try:
                version = Version(raw_version)
            except InvalidVersion:
                continue
            build = str(entry.get("build", ""))
            candidate = (ts, version, raw_version, build)
            if best is None or (ts, version) > (best[0], best[1]):
                best = candidate

    if best is None:
        return None
    return best[2], best[3]


def _cache_path(cache_dir: Path, channel: str, subdir: str, date: _dt.date) -> Path:
    """Return the gzip cache path for a channel/subdir/date triple."""
    return cache_dir / f"{channel}_{subdir}_{date.isoformat()}.json.gz"


def fetch_repodata(
    channel: str,
    date: _dt.date,
    *,
    subdir: str = DEFAULT_SUBDIR,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Download (or load from cache) a channel's ``repodata.json``.

    Args:
        channel: Conda channel name (e.g. ``"bioconda"``).
        date: Target date, used as part of the cache key.
        subdir: Platform subdir (default ``linux-64``).
        cache_dir: Cache directory; defaults to ``<cache_root>/conda``.
        client: Optional shared :class:`httpx.Client`.

    Returns:
        The decoded repodata document (empty dict if the channel 404s).

    Raises:
        httpx.HTTPError: On a non-404 request failure.
    """
    cache_dir = cache_dir or (cache_root() / "conda")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, channel, subdir, date)

    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
            cached: dict[str, Any] = json.load(fh)
        return cached

    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    url = REPODATA_URL.format(channel=channel, subdir=subdir)
    try:
        resp = client.get(url)
        if resp.status_code == 404:
            data: dict[str, Any] = {}
        else:
            resp.raise_for_status()
            data = resp.json()
    finally:
        if owns_client:
            client.close()

    with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def _warn_if_not_linux(channels: list[str]) -> None:
    """Warn (never fail) if bioconda is requested on a non-Linux host."""
    if "bioconda" not in channels:
        return
    system = platform.system()
    if system != "Linux":
        warnings.warn(
            f"bioconda packages are Linux-only; you are on {system} "
            f"({platform.machine()}). Pins resolve against linux-64 and may not "
            "install natively here.",
            RuntimeWarning,
            stacklevel=2,
        )


def pin_conda(
    packages: list[str],
    date: _dt.date,
    channels: list[str] | None = None,
    emit_json: bool = False,
    *,
    subdir: str = DEFAULT_SUBDIR,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``[(package, version, channel), ...]`` pinned to ``date``.

    Channels are searched in order; the first channel holding an eligible build
    of a package wins. User-supplied channels are prepended to the defaults.

    Args:
        packages: Package names to pin.
        date: Target date (inclusive).
        channels: Extra channels to search before the defaults.
        emit_json: Accepted for API symmetry; formatting is the caller's job.
        subdir: Platform subdir (default ``linux-64``).
        cache_dir: Optional cache directory override.
        client: Optional shared :class:`httpx.Client`.

    Returns:
        One tuple per package, in input order. Packages found in no channel
        carry version and channel ``"NOT_FOUND"``.
    """
    # Prepend user channels, then dedupe while preserving order.
    ordered = list(channels or []) + DEFAULT_CHANNELS
    search_channels = list(dict.fromkeys(ordered))
    _warn_if_not_linux(search_channels)

    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    repodata_by_channel: dict[str, dict[str, Any]] = {}
    results: list[tuple[str, str, str]] = []
    try:
        for pkg in packages:
            resolved: tuple[str, str, str] | None = None
            for channel in search_channels:
                if channel not in repodata_by_channel:
                    repodata_by_channel[channel] = fetch_repodata(
                        channel, date, subdir=subdir, cache_dir=cache_dir, client=client
                    )
                hit = select_conda_build(repodata_by_channel[channel], date, pkg)
                if hit is not None:
                    resolved = (pkg, hit[0], channel)
                    break
            results.append(resolved or (pkg, "NOT_FOUND", "NOT_FOUND"))
    finally:
        if owns_client:
            client.close()
    return results


class CondaPinner:
    """CLI handler for ``jeeva pin --conda``."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: Namespace with ``packages``, ``date``, ``channel`` and
                ``json`` attributes.
        """
        self.args = args

    def run(self) -> None:
        """Resolve packages against conda channels and print the result.

        Emits ``conda install package=version=build`` lines by default (build is
        included in the extended JSON), or a JSON array with ``--json``.
        """
        date = _dt.date.fromisoformat(self.args.date)
        channels = getattr(self.args, "channel", None)
        results = pin_conda(self.args.packages, date, channels=channels)
        if getattr(self.args, "json", False):
            payload = [{"package": p, "version": v, "channel": c} for p, v, c in results]
            print(json.dumps(payload, indent=2))
            return
        for pkg, version, channel in results:
            if version == "NOT_FOUND":
                print(
                    f"# {pkg}: NOT_FOUND in {', '.join(DEFAULT_CHANNELS)} as of {date}",
                    file=sys.stderr,
                )
                continue
            print(f"conda install -c {channel} {pkg}={version}")
