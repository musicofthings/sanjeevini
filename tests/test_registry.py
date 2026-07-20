"""Tests for sanjeevini.registry.catalog (target: 80% branch coverage)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from sanjeevini.contracts.schema import (
    ContractSchema,
    GenomicFileType,
    IOPort,
    SequencingPlatform,
)
from sanjeevini.registry.catalog import (
    RegistryCommand,
    default_registry_dirs,
    find_bundle_dir,
    load_catalog,
    pull_contract,
)


def _write_bundle(root: Path, slug: str, payload: dict) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    (d / "contract.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return d


def _schema_dict(**kw) -> dict:
    schema = ContractSchema(
        inputs=[IOPort(name="bam_in", type=GenomicFileType.BAM)],
        outputs=[IOPort(name="vcf_out", type=GenomicFileType.VCF)],
        **kw,
    )
    return json.loads(schema.model_dump_json())


# ---- load_catalog ---------------------------------------------------------


def test_load_catalog_reads_all_entries(sample_registry: Path) -> None:
    entries = load_catalog([sample_registry])
    slugs = {e.slug for e in entries}
    assert slugs == {"sniffles2", "minimap2"}
    assert all(isinstance(e.schema, ContractSchema) for e in entries)
    # sorted by name
    assert [e.name for e in entries] == sorted(e.name for e in entries)


def test_domain_inferred_from_platform(sample_registry: Path) -> None:
    entries = {e.slug: e for e in load_catalog([sample_registry])}
    assert entries["sniffles2"].platform == "ont"
    assert entries["sniffles2"].domain == "longread-ont"
    assert entries["minimap2"].domain == ""  # platform "any" → no inference


def test_explicit_domain_and_platform_win(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        "clair3",
        {
            "slug": "clair3",
            "name": "Clair3",
            "domain": "longread-agnostic",
            "platform": "any",
            "schema": _schema_dict(platform=SequencingPlatform.ONT),
        },
    )
    (entry,) = load_catalog([tmp_path])
    assert entry.domain == "longread-agnostic"
    assert entry.platform == "any"


def test_load_catalog_reads_provenance_sibling(tmp_path: Path) -> None:
    d = _write_bundle(
        tmp_path,
        "sniffles2",
        {"slug": "sniffles2", "name": "Sniffles2", "schema": _schema_dict()},
    )
    (d / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "repo_url": "https://github.com/fritzsedlazeck/Sniffles",
                "resurrection_date": "2026-07-19",
                "final_image": "ghcr.io/x/sniffles2:latest",
            }
        ),
        encoding="utf-8",
    )
    (entry,) = load_catalog([tmp_path])
    assert entry.repo_url.endswith("Sniffles")
    assert entry.last_verified == "2026-07-19"
    assert entry.image == "ghcr.io/x/sniffles2:latest"


def test_load_catalog_dedups_by_slug_and_skips_missing(tmp_path: Path) -> None:
    schema = _schema_dict()
    _write_bundle(tmp_path / "a", "dup", {"slug": "dup", "name": "First", "schema": schema})
    _write_bundle(tmp_path / "b", "dup", {"slug": "dup", "name": "Second", "schema": schema})
    entries = load_catalog([tmp_path / "a", tmp_path / "b", tmp_path / "missing"])
    assert len(entries) == 1
    assert entries[0].name == "First"  # first occurrence wins


def test_entry_to_dict_and_search_text(sample_registry: Path) -> None:
    entry = {e.slug: e for e in load_catalog([sample_registry])}["sniffles2"]
    d = entry.to_dict()
    assert set(d) >= {"slug", "name", "domain", "platform", "schema", "provenance"}
    assert "longread-ont" in entry.search_text()


# ---- pull_contract --------------------------------------------------------


def test_pull_contract_copies_contract_and_dockerfile(tmp_path: Path) -> None:
    d = _write_bundle(
        tmp_path / "reg",
        "sniffles2",
        {"slug": "sniffles2", "name": "Sniffles2", "schema": _schema_dict()},
    )
    (d / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    dest = tmp_path / "out"
    result = pull_contract("sniffles2", dest, registry_dirs=[tmp_path / "reg"])
    assert result == dest / "sniffles2"
    assert (result / "contract.yaml").is_file()
    assert (result / "Dockerfile").is_file()


def test_pull_contract_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        pull_contract("nope", tmp_path / "out", registry_dirs=[tmp_path])


def test_find_bundle_dir(sample_registry: Path) -> None:
    found = find_bundle_dir("minimap2", [sample_registry])
    assert found is not None and found.name == "minimap2"
    assert find_bundle_dir("ghost", [sample_registry]) is None


# ---- RegistryCommand ------------------------------------------------------


def _chdir_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = _write_bundle(
        tmp_path / "contracts",
        "sniffles2",
        {
            "slug": "sniffles2",
            "name": "Sniffles2",
            "domain": "longread-ont",
            "platform": "ont",
            "capability": "structural variant SV caller for ONT long reads",
            "schema": _schema_dict(platform=SequencingPlatform.ONT),
        },
    )
    (d / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_registry_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _chdir_registry(tmp_path, monkeypatch)
    assert default_registry_dirs() == [Path("contracts")]


def test_registry_command_list_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _chdir_registry(tmp_path, monkeypatch)
    args = argparse.Namespace(registry_cmd="list", domain=None, platform=None, json=False)
    RegistryCommand(args).run()
    out = capsys.readouterr().out
    assert "sniffles2" in out and "longread-ont" in out


def test_registry_command_list_json_and_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _chdir_registry(tmp_path, monkeypatch)
    args = argparse.Namespace(registry_cmd="list", domain="methylation", platform=None, json=True)
    RegistryCommand(args).run()
    assert json.loads(capsys.readouterr().out) == []  # filtered out


def test_registry_command_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _chdir_registry(tmp_path, monkeypatch)
    args = argparse.Namespace(registry_cmd="search", query="SV caller for ONT", top=5)
    RegistryCommand(args).run()
    out = capsys.readouterr().out
    assert "sniffles2" in out


def test_registry_command_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _chdir_registry(tmp_path, monkeypatch)
    args = argparse.Namespace(registry_cmd="pull", tool="sniffles2")
    RegistryCommand(args).run()
    assert (tmp_path / "sniffles2" / "contract.yaml").is_file()
    assert "pulled sniffles2" in capsys.readouterr().out
