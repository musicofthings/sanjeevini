"""Tests for sanjeevini.mcp.server (stdio JSON-RPC transport)."""

from __future__ import annotations

import io
import json

import pytest

from sanjeevini.mcp import server


def test_initialize_returns_server_info() -> None:
    resp = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp is not None
    assert resp["result"]["serverInfo"] == server.SERVER_INFO
    assert resp["result"]["protocolVersion"] == server.PROTOCOL_VERSION


def test_tools_list_exposes_all_verbs() -> None:
    resp = server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert resp is not None
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"resurrect", "pin", "decay_check", "registry_search", "run_pipeline"}


def test_notifications_get_no_reply() -> None:
    assert server.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_returns_method_not_found() -> None:
    resp = server.dispatch({"jsonrpc": "2.0", "id": 9, "method": "does/not/exist"})
    assert resp is not None
    assert resp["error"]["code"] == -32601


def test_tools_call_dispatches_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_jeeva(*argv: str, timeout: int = 30) -> str:
        captured["argv"] = argv
        return json.dumps({"verdict": "naive_runs"})

    monkeypatch.setattr(server, "_run_jeeva", fake_run_jeeva)
    resp = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "decay_check", "arguments": {"url": "https://github.com/x/y"}},
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["verdict"] == "naive_runs"
    assert captured["argv"][0] == "decay-check"


@pytest.mark.parametrize(
    ("name", "arguments", "expected_head"),
    [
        ("resurrect", {"url": "https://github.com/x/y", "turns": 10}, ["resurrect"]),
        (
            "pin",
            {
                "packages": ["pysam"],
                "date": "2021-01-01",
                "ecosystem": "conda",
                "channels": ["bioconda"],
            },
            ["pin", "--date", "2021-01-01", "pysam", "--conda"],
        ),
        (
            "registry_search",
            {"query": "SV caller", "domain": "longread-ont"},
            ["registry", "search", "SV caller"],
        ),
        ("run_pipeline", {"pipeline_yaml": "p.yaml", "dry_run": True}, ["run", "p.yaml"]),
    ],
)
def test_tool_argv_builders(
    monkeypatch: pytest.MonkeyPatch, name: str, arguments: dict, expected_head: list[str]
) -> None:
    seen: dict[str, tuple] = {}

    def fake_run_jeeva(*argv: str, timeout: int = 30) -> str:
        seen["argv"] = argv
        return "{}"

    monkeypatch.setattr(server, "_run_jeeva", fake_run_jeeva)
    server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    argv = list(seen["argv"])
    assert argv[: len(expected_head)] == expected_head


def test_unknown_tool_returns_error() -> None:
    resp = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert resp is not None
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert "Unknown tool" in payload["error"]


def test_serve_stdio_round_trip() -> None:
    lines = [
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n',
        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n',
        "\n",  # blank line ignored
        "not json\n",  # malformed line ignored
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',
    ]
    out = io.StringIO()
    server.serve_stdio(lines, out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in responses] == [1, 2]
    assert {t["name"] for t in responses[1]["result"]["tools"]} >= {"resurrect", "pin"}
