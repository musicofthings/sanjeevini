"""Tests for sanjeevini.pinners.bioc (target: 85% branch coverage)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from sanjeevini.pinners.bioc import (
    VIEWS_URL,
    BiocPinner,
    parse_views,
    pin_bioc,
    resolve_release,
)

_VIEWS_314 = """\
Package: DESeq2
Version: 1.34.0
Depends: S4Vectors, IRanges
biocViews: RNASeq, DifferentialExpression

Package: edgeR
Version: 3.36.0
Depends: limma

Package: limma
Version: 3.50.0
"""


class FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        pass


class FakeClient:
    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(200, self.routes[url])

    def close(self) -> None:
        pass


# ---- release resolution ---------------------------------------------------


def test_resolve_release_picks_most_recent() -> None:
    assert resolve_release(dt.date(2021, 11, 1)).bioc_version == "3.14"
    assert resolve_release(dt.date(2021, 11, 1)).r_version == "4.1"


def test_resolve_release_on_exact_release_date() -> None:
    # 3.14 released 2021-10-27; that date resolves to 3.14, not 3.13.
    assert resolve_release(dt.date(2021, 10, 27)).bioc_version == "3.14"


def test_resolve_release_raises_before_earliest() -> None:
    with pytest.raises(ValueError):
        resolve_release(dt.date(2019, 1, 1))


# ---- VIEWS parsing --------------------------------------------------------


def test_parse_views_extracts_versions() -> None:
    parsed = parse_views(_VIEWS_314)
    assert parsed["DESeq2"] == "1.34.0"
    assert parsed["edgeR"] == "3.36.0"
    assert parsed["limma"] == "3.50.0"


def test_parse_views_handles_final_record_without_trailing_blank() -> None:
    text = "Package: foo\nVersion: 1.0.0"
    assert parse_views(text) == {"foo": "1.0.0"}


# ---- pin_bioc -------------------------------------------------------------


def test_pin_bioc_deseq2(tmp_path: Path) -> None:
    client = FakeClient({VIEWS_URL.format(release="3.14"): _VIEWS_314})
    result = pin_bioc(["DESeq2"], dt.date(2021, 11, 1), cache_dir=tmp_path, client=client)
    assert result.bioc_version == "3.14"
    assert result.r_version == "4.1"
    assert result.package_versions == [("DESeq2", "1.34.0")]
    assert 'BiocManager::install(version = "3.14")' in result.install_script
    assert '"DESeq2"' in result.install_script


def test_pin_bioc_unknown_package_marked(tmp_path: Path) -> None:
    client = FakeClient({VIEWS_URL.format(release="3.14"): _VIEWS_314})
    result = pin_bioc(["notapkg"], dt.date(2021, 11, 1), cache_dir=tmp_path, client=client)
    assert result.package_versions == [("notapkg", "NOT_IN_BIOC")]
    assert "CRAN fallback" in result.install_script


def test_pin_bioc_caches_views(tmp_path: Path) -> None:
    client = FakeClient({VIEWS_URL.format(release="3.14"): _VIEWS_314})
    pin_bioc(["DESeq2"], dt.date(2021, 11, 1), cache_dir=tmp_path, client=client)
    pin_bioc(["edgeR"], dt.date(2021, 11, 1), cache_dir=tmp_path, client=client)
    assert len(client.calls) == 1  # VIEWS fetched once, then cached


def test_bioc_pinner_json_output(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "bioc"
    cache_dir.mkdir(parents=True)
    (cache_dir / "VIEWS_3.14.dcf").write_text(_VIEWS_314)

    args = type("A", (), {"packages": ["DESeq2"], "date": "2021-11-01", "json": True})()
    BiocPinner(args).run()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["bioc_version"] == "3.14"
    assert parsed["package_versions"][0] == {"package": "DESeq2", "version": "1.34.0"}


def test_bioc_pinner_script_output(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "bioc"
    cache_dir.mkdir(parents=True)
    (cache_dir / "VIEWS_3.14.dcf").write_text(_VIEWS_314)

    args = type("A", (), {"packages": ["DESeq2"], "date": "2021-11-01", "json": False})()
    BiocPinner(args).run()
    out = capsys.readouterr().out
    assert "BiocManager::install" in out
