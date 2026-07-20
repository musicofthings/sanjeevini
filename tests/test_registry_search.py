"""Tests for sanjeevini.registry.search (target: 80% branch coverage).

These exercise the dependency-free lexical backend; the semantic backend
(``sentence-transformers``) is marked no-cover and tested only when installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanjeevini.contracts.schema import ContractSchema
from sanjeevini.registry.catalog import RegistryEntry
from sanjeevini.registry.search import RegistrySearchEngine, _tokenize, index_path


def _entry(slug: str, domain: str, platform: str, capability: str) -> RegistryEntry:
    return RegistryEntry(
        slug=slug,
        name=slug,
        repo_url=f"https://github.com/x/{slug}",
        domain=domain,
        platform=platform,
        image="",
        schema=ContractSchema(),
        capability=capability,
    )


def _catalog() -> list[RegistryEntry]:
    return [
        _entry(
            "sniffles2", "longread-ont", "ont", "structural variant SV caller for ONT long reads"
        ),
        _entry(
            "pbsv",
            "longread-pacbio",
            "pacbio_hifi",
            "structural variant caller for PacBio HiFi reads",
        ),
        _entry("deseq2", "rna-seq", "illumina", "differential gene expression from RNA-seq counts"),
    ]


def _engine() -> RegistrySearchEngine:
    engine = RegistrySearchEngine(_catalog())
    engine.build_index()
    return engine


def test_tokenize() -> None:
    assert _tokenize("Longread-ONT SV_caller!") == ["longread", "ont", "sv", "caller"]


def test_build_index_uses_lexical_without_deps() -> None:
    engine = _engine()
    assert engine.backend == "lexical"


def test_lexical_search_finds_ont_sv_caller() -> None:
    results = _engine().search("SV caller for ONT")
    assert results, "expected at least one result"
    top_entry, top_score = results[0]
    assert top_entry.domain == "longread-ont"
    assert top_score > 0.0


def test_search_domain_filter() -> None:
    results = _engine().search("structural variant caller", domain_filter="longread-pacbio")
    assert [e.slug for e, _ in results] == ["pbsv"]


def test_search_platform_filter() -> None:
    results = _engine().search("caller", platform_filter="ont")
    assert all(e.platform == "ont" for e, _ in results)


def test_search_top_k_limits_results() -> None:
    results = _engine().search("caller", top_k=1)
    assert len(results) == 1


def test_search_empty_query_returns_nothing() -> None:
    assert _engine().search("") == []


def test_search_no_match_returns_empty() -> None:
    assert _engine().search("quantum chromodynamics tensor") == []


def test_search_empty_catalog() -> None:
    engine = RegistrySearchEngine([])
    engine.build_index()
    assert engine.search("anything") == []


def test_index_path_honours_cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    assert index_path() == tmp_path / "registry" / "index.faiss"
