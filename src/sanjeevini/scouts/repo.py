"""Repository snapshot: the files a Scout reads before writing a plan.

A :class:`RepoSnapshot` is a small, in-memory view of the handful of files that
determine how a repo should be resurrected — README, dependency manifests, a
Dockerfile, CI workflows, an R ``DESCRIPTION``, and the first few open issues.

The Scouts take a snapshot and reason over it purely, which keeps their logic
unit-testable: tests construct a snapshot directly, while production fetches one
from GitHub's raw endpoints via :func:`fetch_snapshot`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

_GH_RE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s#?]+)")

# Files the Scouts care about, fetched from a repo's default branch.
_FETCH_PATHS = (
    "README.md",
    "README.rst",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "Dockerfile",
    "environment.yml",
    "DESCRIPTION",
)

_RAW_URL = "https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}"
_ISSUES_API = "https://api.github.com/repos/{owner}/{name}/issues"
_BRANCHES = ("main", "master")
_TIMEOUT_S = 15.0


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract ``(owner, name)`` from a GitHub URL.

    Args:
        url: A GitHub URL (https, ssh, with or without ``.git``/trailing path).

    Returns:
        The ``(owner, name)`` pair.

    Raises:
        ValueError: If ``url`` is not a recognisable GitHub repository URL.
    """
    match = _GH_RE.search(url)
    if not match:
        raise ValueError(f"not a GitHub repository URL: {url!r}")
    owner, name = match.group(1), match.group(2)
    if name.endswith(".git"):
        name = name[:-4]
    return owner, name


@dataclass
class RepoSnapshot:
    """The files and metadata a Scout reads to plan a resurrection.

    Attributes:
        url: The original repository URL.
        owner: GitHub owner/organisation.
        name: Repository name.
        files: Mapping of file path to its text content (present files only).
        open_issues: ``(title, body)`` pairs for the first few open issues.
        paper_abstract: Abstract text of an associated paper, if fetched.
    """

    url: str
    owner: str
    name: str
    files: dict[str, str] = field(default_factory=dict)
    open_issues: list[tuple[str, str]] = field(default_factory=list)
    paper_abstract: str = ""

    def get(self, *names: str) -> str:
        """Return the content of the first present file among ``names``.

        Args:
            *names: Candidate file paths, tried in order.

        Returns:
            The file content, or an empty string if none are present.
        """
        for n in names:
            if n in self.files:
                return self.files[n]
        return ""

    def has(self, name: str) -> bool:
        """Return whether ``name`` is present and non-empty in the snapshot."""
        return bool(self.files.get(name))


async def fetch_snapshot(url: str, *, client: httpx.AsyncClient | None = None) -> RepoSnapshot:
    """Fetch a :class:`RepoSnapshot` from GitHub's raw endpoints.

    Tries the ``main`` then ``master`` branch for each known file path and
    collects the first three open issues. Network failures degrade gracefully to
    a partial snapshot rather than raising.

    Args:
        url: The GitHub repository URL.
        client: An :class:`httpx.AsyncClient` to reuse; one is created if
            omitted.

    Returns:
        The assembled snapshot (possibly partial if some fetches failed).
    """
    owner, name = parse_repo_url(url)
    snapshot = RepoSnapshot(url=url, owner=owner, name=name)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True)
    try:
        for path in _FETCH_PATHS:
            for branch in _BRANCHES:
                raw = _RAW_URL.format(owner=owner, name=name, branch=branch, path=path)
                try:
                    resp = await client.get(raw)
                except httpx.HTTPError:
                    continue
                if resp.status_code == 200:
                    snapshot.files[path] = resp.text
                    break
        try:
            resp = await client.get(
                _ISSUES_API.format(owner=owner, name=name),
                params={"state": "open", "per_page": 3},
                headers={"Accept": "application/vnd.github+json"},
            )
            if resp.status_code == 200:
                for issue in resp.json():
                    if "pull_request" in issue:
                        continue
                    snapshot.open_issues.append((issue.get("title", ""), issue.get("body") or ""))
        except httpx.HTTPError:
            pass
    finally:
        if owns_client:
            await client.aclose()
    return snapshot
