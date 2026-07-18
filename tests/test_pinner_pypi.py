"""Tests for sanjeevini.pinners.pypi (target: 85% branch coverage)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx

from sanjeevini.pinners.pypi import (
    NOT_FOUND,
    PyPIPinner,
    ReleaseInfo,
    pin_pypi,
    resolve_pypi,
    select_version,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeClient:
    """Counting fake httpx client keyed by URL."""

    def __init__(self, routes: dict[str, FakeResponse]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(404, {}))

    def close(self) -> None:
        pass


def _numpy_payload() -> dict:
    return {
        "releases": {
            "1.17.0": [{"upload_time_iso_8601": "2019-07-26T00:00:00Z"}],
            "1.17.5": [{"upload_time_iso_8601": "2020-01-01T12:00:00Z"}],
            "1.18.0": [{"upload_time_iso_8601": "2019-12-22T00:00:00Z"}],
            "1.18.1": [{"upload_time_iso_8601": "2020-01-06T00:00:00Z"}],
            "1.19.0": [{"upload_time_iso_8601": "2020-06-20T00:00:00Z"}],
        }
    }


def _url(pkg: str) -> str:
    return f"https://pypi.org/pypi/{pkg}/json"


# ---- pure core ------------------------------------------------------------


def test_select_version_picks_newest_before_cutoff() -> None:
    releases = [
        ReleaseInfo("1.18.0", dt.datetime(2019, 12, 22, tzinfo=dt.timezone.utc)),
        ReleaseInfo("1.18.1", dt.datetime(2020, 1, 6, tzinfo=dt.timezone.utc)),
        ReleaseInfo("1.10.0", dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)),
    ]
    # PEP 440 ordering: 1.18.0 beats 1.10.0; 1.18.1 is after the cutoff.
    assert select_version(releases, dt.date(2020, 1, 1)) == "1.18.0"


def test_select_version_skips_prerelease_and_yanked() -> None:
    releases = [
        ReleaseInfo("2.0.0rc1", dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)),
        ReleaseInfo("1.9.0", dt.datetime(2019, 6, 1, tzinfo=dt.timezone.utc), yanked=True),
        ReleaseInfo("1.8.0", dt.datetime(2019, 5, 1, tzinfo=dt.timezone.utc)),
    ]
    assert select_version(releases, dt.date(2020, 6, 1)) == "1.8.0"
    assert select_version(releases, dt.date(2020, 6, 1), allow_prerelease=True) == "2.0.0rc1"


def test_select_version_none_when_all_after_cutoff() -> None:
    releases = [ReleaseInfo("1.0.0", dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc))]
    assert select_version(releases, dt.date(2020, 1, 1)) is None


# ---- pin_pypi / resolve ---------------------------------------------------


def test_pin_known_package(tmp_path: Path) -> None:
    client = FakeClient({_url("numpy"): FakeResponse(200, _numpy_payload())})
    result = pin_pypi(["numpy"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client)
    assert len(result) == 1
    pkg, version = result[0]
    assert pkg == "numpy"
    assert version.startswith("1.17.") or version.startswith("1.18.")


def test_pin_not_found_does_not_raise(tmp_path: Path) -> None:
    client = FakeClient({_url("definitely-not-a-real-pkg"): FakeResponse(404, {})})
    result = pin_pypi(
        ["definitely-not-a-real-pkg"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client
    )
    assert result == [("definitely-not-a-real-pkg", NOT_FOUND)]


def test_cache_hit_second_call_uses_cache(tmp_path: Path) -> None:
    client = FakeClient({_url("numpy"): FakeResponse(200, _numpy_payload())})
    pin_pypi(["numpy"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client)
    pin_pypi(["numpy"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client)
    assert len(client.calls) == 1  # second run served from disk cache


def test_resolve_includes_upload_date(tmp_path: Path) -> None:
    client = FakeClient({_url("numpy"): FakeResponse(200, _numpy_payload())})
    records = resolve_pypi(["numpy"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client)
    assert records[0]["upload_date"] is not None
    dt.date.fromisoformat(records[0]["upload_date"])  # parseable ISO date


def test_output_json_via_cache(tmp_path, monkeypatch, capsys) -> None:
    # Pre-seed the on-disk cache so run() needs no network.
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "pypi"
    cache_dir.mkdir(parents=True)
    (cache_dir / "numpy_2020-01-01.json").write_text(json.dumps(_numpy_payload()))

    args = type("A", (), {"packages": ["numpy"], "date": "2020-01-01", "json": True})()
    PyPIPinner(args).run()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed[0]["package"] == "numpy"
    assert {"package", "version", "upload_date"} <= set(parsed[0])


def test_output_pip_lines_via_cache(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "pypi"
    cache_dir.mkdir(parents=True)
    (cache_dir / "numpy_2020-01-01.json").write_text(json.dumps(_numpy_payload()))

    args = type("A", (), {"packages": ["numpy"], "date": "2020-01-01", "json": False})()
    PyPIPinner(args).run()
    out = capsys.readouterr().out
    assert out.startswith("pip install numpy==")


def test_as_cutoff_accepts_naive_datetime() -> None:
    from sanjeevini.pinners.pypi import as_cutoff

    cutoff = as_cutoff(dt.datetime(2020, 1, 1, 10, 30))
    assert cutoff.tzinfo is not None
    assert cutoff.year == 2020


def test_resolve_not_found_when_no_eligible_version(tmp_path: Path) -> None:
    # numpy exists, but every release is after the (very old) cutoff.
    client = FakeClient({_url("numpy"): FakeResponse(200, _numpy_payload())})
    result = pin_pypi(["numpy"], dt.date(2010, 1, 1), cache_dir=tmp_path, client=client)
    assert result == [("numpy", NOT_FOUND)]


def test_parse_upload_time_ignores_unparseable(tmp_path: Path) -> None:
    payload = {
        "releases": {
            "1.0.0": [{"upload_time_iso_8601": "not-a-date"}, {}],
            "2.0.0": [{"upload_time_iso_8601": "2019-01-01T00:00:00Z"}],
        }
    }
    client = FakeClient({_url("pkg"): FakeResponse(200, payload)})
    result = pin_pypi(["pkg"], dt.date(2020, 1, 1), cache_dir=tmp_path, client=client)
    # 1.0.0 has no usable upload time and is dropped; 2.0.0 wins.
    assert result == [("pkg", "2.0.0")]


def test_run_not_found_reports_to_stderr(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SANJEEVINI_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "pypi"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ghost_2020-01-01.json").write_text(json.dumps({"not_found": True}))

    args = type("A", (), {"packages": ["ghost"], "date": "2020-01-01", "json": False})()
    PyPIPinner(args).run()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "NOT_FOUND" in captured.err


def test_retry_then_succeed(tmp_path: Path) -> None:
    from sanjeevini.pinners import pypi

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("boom")
            return FakeResponse(200, _numpy_payload())

        def close(self) -> None:
            pass

    flaky = FlakyClient()
    history = pypi.fetch_release_history(
        "numpy", dt.date(2020, 1, 1), cache_dir=tmp_path, client=flaky, _sleep=lambda _s: None
    )
    assert history is not None
    assert flaky.calls == 2
