"""Tests for sanjeevini.pinners.conda (target: 85% branch coverage)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sanjeevini.pinners.conda import (
    REPODATA_URL,
    CondaPinner,
    pin_conda,
    select_conda_build,
)


def _ms(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp() * 1000)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class FakeClient:
    def __init__(self, routes: dict[str, dict]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        if url in self.routes:
            return FakeResponse(200, self.routes[url])
        return FakeResponse(404, {})

    def close(self) -> None:
        pass


def _url(channel: str) -> str:
    return REPODATA_URL.format(channel=channel, subdir="linux-64")


def _samtools_repodata() -> dict:
    return {
        "packages": {
            "samtools-1.12-h9aed4be_1.tar.bz2": {
                "name": "samtools", "version": "1.12", "build": "h9aed4be_1",
                "timestamp": _ms(2021, 3, 25),
            },
            "samtools-1.13-h8c37831_0.tar.bz2": {
                "name": "samtools", "version": "1.13", "build": "h8c37831_0",
                "timestamp": _ms(2021, 7, 19),
            },
        }
    }


# ---- pure core ------------------------------------------------------------


def test_select_conda_build_respects_cutoff() -> None:
    repodata = _samtools_repodata()
    hit = select_conda_build(repodata, dt.date(2021, 6, 1), "samtools")
    assert hit == ("1.12", "h9aed4be_1")  # 1.13 is after the cutoff


def test_select_conda_build_none_for_missing_package() -> None:
    assert select_conda_build(_samtools_repodata(), dt.date(2021, 6, 1), "bcftools") is None


def test_select_conda_build_reads_packages_conda_section() -> None:
    repodata = {
        "packages.conda": {
            "minimap2-2.24-h7132678_1.conda": {
                "name": "minimap2", "version": "2.24", "build": "h7132678_1",
                "timestamp": _ms(2022, 1, 15),
            }
        }
    }
    assert select_conda_build(repodata, dt.date(2022, 6, 1), "minimap2") == ("2.24", "h7132678_1")


# ---- pin_conda ------------------------------------------------------------


def test_pin_bioconda_package(tmp_path: Path) -> None:
    client = FakeClient({_url("bioconda"): _samtools_repodata()})
    result = pin_conda(
        ["samtools"], dt.date(2021, 6, 1), cache_dir=tmp_path, client=client
    )
    pkg, version, channel = result[0]
    assert pkg == "samtools"
    assert version in ("1.12", "1.13")
    assert channel == "bioconda"


def test_pin_conda_forge_fallback(tmp_path: Path) -> None:
    forge = {
        "packages": {
            "scipy-1.7.0-py39_0.tar.bz2": {
                "name": "scipy", "version": "1.7.0", "build": "py39_0",
                "timestamp": _ms(2021, 6, 20),
            }
        }
    }
    client = FakeClient({_url("bioconda"): {}, _url("conda-forge"): forge})
    result = pin_conda(["scipy"], dt.date(2021, 8, 1), cache_dir=tmp_path, client=client)
    pkg, version, channel = result[0]
    assert channel == "conda-forge"
    assert version == "1.7.0"


def test_channel_order_prefers_bioconda(tmp_path: Path) -> None:
    shared_bioconda = {
        "packages": {
            "shared-1.0-0.tar.bz2": {
                "name": "shared", "version": "1.0", "build": "0", "timestamp": _ms(2021, 1, 1),
            }
        }
    }
    shared_forge = {
        "packages": {
            "shared-2.0-0.tar.bz2": {
                "name": "shared", "version": "2.0", "build": "0", "timestamp": _ms(2021, 1, 1),
            }
        }
    }
    client = FakeClient({_url("bioconda"): shared_bioconda, _url("conda-forge"): shared_forge})
    result = pin_conda(["shared"], dt.date(2021, 6, 1), cache_dir=tmp_path, client=client)
    _, version, channel = result[0]
    assert channel == "bioconda"
    assert version == "1.0"


def test_cache_repodata_fetched_once(tmp_path: Path) -> None:
    client = FakeClient({_url("bioconda"): _samtools_repodata()})
    pin_conda(["samtools"], dt.date(2021, 6, 1), cache_dir=tmp_path, client=client)
    pin_conda(["samtools"], dt.date(2021, 6, 1), cache_dir=tmp_path, client=client)
    # bioconda repodata fetched on the first run, served from gz cache on the second
    assert client.calls.count(_url("bioconda")) == 1


def test_not_found_across_channels(tmp_path: Path) -> None:
    client = FakeClient({})  # every channel 404s -> empty repodata
    result = pin_conda(["ghostpkg"], dt.date(2021, 6, 1), cache_dir=tmp_path, client=client)
    assert result == [("ghostpkg", "NOT_FOUND", "NOT_FOUND")]


def test_output_json(tmp_path, monkeypatch, capsys) -> None:
    import gzip

    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "conda"
    cache_dir.mkdir(parents=True)
    with gzip.open(cache_dir / "bioconda_linux-64_2021-06-01.json.gz", "wt") as fh:
        json.dump(_samtools_repodata(), fh)

    args = type(
        "A", (), {"packages": ["samtools"], "date": "2021-06-01", "channel": None, "json": True}
    )()
    CondaPinner(args).run()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["package"] == "samtools"
    assert {"package", "version", "channel"} <= set(parsed[0])
