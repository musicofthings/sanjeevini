"""The registry catalog — load, filter, and pull resurrected-tool contracts.

A catalog is a directory tree of per-tool ``contract.yaml`` bundles (exactly what
the repair loop's contract emitter writes under ``contracts/{slug}/``). Each
bundle yields one :class:`RegistryEntry`. :func:`load_catalog` reads every bundle
found under the given directories; :func:`pull_contract` copies a tool's bundle
(``contract.yaml`` + ``Dockerfile`` + the rest) to a destination directory.

The catalog is a thin data layer: entries carry the tool's typed
:class:`~sanjeevini.contracts.schema.ContractSchema`, its domain/platform tags
(for filtering), the GHCR pull URL, and the provenance/verification record.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sanjeevini.contracts.schema import ContractSchema

# The files that make up a contract bundle, copied by ``jeeva registry pull``.
CONTRACT_FILES = (
    "contract.yaml",
    "predict.py",
    "Dockerfile",
    "smoke_test.sh",
    "REPRODUCE.md",
    "PROVENANCE.json",
)

# Recognised registry domains (PRD §7 taxonomy).
DOMAINS = (
    "longread-ont",
    "longread-pacbio",
    "longread-agnostic",
    "variant-calling",
    "sv-calling",
    "methylation",
    "rna-seq",
    "proteomics",
    "docking",
    "single-cell",
    "assembly",
)

# Default directories a catalog is assembled from, in precedence order.
_DEFAULT_DIRS = ("contracts", "registry")

# Platform → domain fallback when a bundle omits an explicit domain.
_PLATFORM_DOMAIN = {
    "ont": "longread-ont",
    "pacbio_hifi": "longread-pacbio",
    "pacbio_clr": "longread-pacbio",
}


@dataclass
class RegistryEntry:
    """One revived tool in the registry.

    Attributes:
        slug: Filesystem-safe identifier (e.g. ``"sniffles2"``).
        name: Human-readable tool name.
        repo_url: The resurrected source repository.
        domain: One of :data:`DOMAINS` (or ``""`` if unclassified).
        platform: Sequencing platform tag (e.g. ``"ont"``, ``"pacbio_hifi"``,
            ``"any"``).
        image: GHCR pull URL for the resurrected image, if published.
        schema: The tool's typed I/O contract schema.
        provenance: The ``PROVENANCE.json`` record (bug classes, turns, cost).
        last_verified: ISO date of the last decay-check PASS.
        capability: One-line description of what the tool does (used by search).
    """

    slug: str
    name: str
    repo_url: str
    domain: str
    platform: str
    image: str
    schema: ContractSchema
    provenance: dict[str, Any] = field(default_factory=dict)
    last_verified: str = ""
    capability: str = ""

    def search_text(self) -> str:
        """Return the text a search engine embeds: name + capability + domain."""
        return " ".join(p for p in (self.name, self.capability, self.domain) if p)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this entry."""
        return {
            "slug": self.slug,
            "name": self.name,
            "repo_url": self.repo_url,
            "domain": self.domain,
            "platform": self.platform,
            "image": self.image,
            "schema": json.loads(self.schema.model_dump_json()),
            "provenance": self.provenance,
            "last_verified": self.last_verified,
            "capability": self.capability,
        }


def _coerce_schema(raw: Any) -> ContractSchema:
    """Return a :class:`ContractSchema` from a dict/JSON value, or an empty one."""
    if isinstance(raw, dict):
        return ContractSchema.model_validate(raw)
    if isinstance(raw, str) and raw.strip():
        return ContractSchema.from_json(raw)
    return ContractSchema()


def _entry_from_bundle(contract_path: Path) -> RegistryEntry:
    """Build a :class:`RegistryEntry` from a ``contract.yaml`` bundle.

    Missing fields are derived: the slug from the parent directory, the platform
    from the schema, the domain from the platform, and the provenance/verification
    date from a sibling ``PROVENANCE.json``.

    Args:
        contract_path: Path to a ``contract.yaml`` file.

    Returns:
        The assembled entry.
    """
    data: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    schema = _coerce_schema(data.get("schema"))

    bundle_dir = contract_path.parent
    slug = str(data.get("slug") or bundle_dir.name)

    provenance: dict[str, Any] = data.get("provenance") or {}
    prov_path = bundle_dir / "PROVENANCE.json"
    if not provenance and prov_path.is_file():
        with prov_path.open(encoding="utf-8") as fh:
            provenance = json.load(fh)

    platform = str(data.get("platform") or schema.platform.value)
    domain = str(data.get("domain") or _PLATFORM_DOMAIN.get(platform, ""))
    last_verified = str(data.get("last_verified") or provenance.get("resurrection_date", ""))

    return RegistryEntry(
        slug=slug,
        name=str(data.get("name") or slug),
        repo_url=str(data.get("repo_url") or provenance.get("repo_url", "")),
        domain=domain,
        platform=platform,
        image=str(data.get("image") or provenance.get("final_image", "")),
        schema=schema,
        provenance=provenance,
        last_verified=last_verified,
        capability=str(data.get("capability") or ""),
    )


def load_catalog(registry_dirs: list[Path]) -> list[RegistryEntry]:
    """Load every ``contract.yaml`` bundle found under ``registry_dirs``.

    Args:
        registry_dirs: Directories to scan; each is searched recursively for
            ``contract.yaml`` files. Missing directories are skipped.

    Returns:
        The entries, de-duplicated by slug (first occurrence wins) and sorted by
        name.
    """
    entries: dict[str, RegistryEntry] = {}
    for directory in registry_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for contract_path in sorted(directory.rglob("contract.yaml")):
            entry = _entry_from_bundle(contract_path)
            entries.setdefault(entry.slug, entry)
    return sorted(entries.values(), key=lambda e: e.name.lower())


def default_registry_dirs() -> list[Path]:
    """Return the default catalog directories that exist in the CWD."""
    return [Path(d) for d in _DEFAULT_DIRS if Path(d).is_dir()]


def find_bundle_dir(slug: str, registry_dirs: list[Path]) -> Path | None:
    """Return the bundle directory for ``slug``, or ``None`` if not found."""
    for directory in registry_dirs:
        for contract_path in Path(directory).rglob("contract.yaml"):
            data = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
            found = str(data.get("slug") or contract_path.parent.name)
            if found == slug:
                return contract_path.parent
    return None


def pull_contract(
    slug: str, dest: Path, *, registry_dirs: list[Path] | None = None
) -> Path:
    """Copy a tool's contract bundle into ``dest/{slug}/``.

    Args:
        slug: The tool slug to pull.
        dest: Destination directory (a per-slug subdirectory is created under it).
        registry_dirs: Directories to search; defaults to :func:`default_registry_dirs`.

    Returns:
        The destination bundle directory.

    Raises:
        KeyError: If no bundle for ``slug`` is found.
    """
    dirs = registry_dirs if registry_dirs is not None else default_registry_dirs()
    source = find_bundle_dir(slug, dirs)
    if source is None:
        raise KeyError(f"no contract bundle for {slug!r} in {[str(d) for d in dirs]}")

    out = Path(dest) / slug
    out.mkdir(parents=True, exist_ok=True)
    for fname in CONTRACT_FILES:
        src = source / fname
        if src.is_file():
            shutil.copyfile(src, out / fname)
    return out


class RegistryCommand:
    """CLI handler for ``jeeva registry`` (``list`` / ``search`` / ``pull``)."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: The ``registry`` subparser namespace (with ``registry_cmd``).
        """
        self.args = args

    def run(self) -> None:
        """Dispatch to the ``list``, ``search``, or ``pull`` subcommand."""
        cmd = self.args.registry_cmd
        if cmd == "list":
            self._list()
        elif cmd == "search":
            self._search()
        elif cmd == "pull":
            self._pull()

    def _catalog(self) -> list[RegistryEntry]:
        return load_catalog(default_registry_dirs())

    def _list(self) -> None:
        entries = self._catalog()
        domain = getattr(self.args, "domain", None)
        platform = getattr(self.args, "platform", None)
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if platform:
            entries = [e for e in entries if e.platform == platform]

        if getattr(self.args, "json", False):
            print(json.dumps([e.to_dict() for e in entries], indent=2))
            return
        if not entries:
            print("no tools in the registry (looked in: "
                  f"{', '.join(str(d) for d in default_registry_dirs()) or 'contracts, registry'})")
            return
        for e in entries:
            verified = f" · verified {e.last_verified}" if e.last_verified else ""
            print(f"{e.slug:<24} {e.domain or '—':<18} {e.platform:<12} {e.name}{verified}")

    def _search(self) -> None:
        from sanjeevini.registry.search import RegistrySearchEngine

        engine = RegistrySearchEngine(self._catalog())
        engine.build_index()
        if engine.backend == "lexical":
            print(
                "note: semantic search backend not installed; using lexical fallback. "
                "For embeddings: pip install 'sanjeevini-bio[search]'"
            )
        results = engine.search(self.args.query, top_k=self.args.top)
        if not results:
            print("no matches.")
            return
        for entry, score in results:
            print(f"{score:6.3f}  {entry.slug:<24} {entry.domain or '—':<18} {entry.name}")

    def _pull(self) -> None:
        out = pull_contract(self.args.tool, Path("."))
        print(f"pulled {self.args.tool} → {out}")
