"""Sanjeevini MCP server — Jeeva as a tool provider.

Exposes the core Sanjeevini verbs as MCP tools so any AI coding agent
(Claude Code, Copilot Workspace, etc.) can call resurrect, pin, compose,
decay-check, and registry-search directly, without driving the CLI.

Transport
---------
  stdio (default):  ``jeeva mcp``
  SSE (web):        ``jeeva mcp --host sse --port 8765``

Usage from Claude Code
----------------------
  Add to .mcp.json:

      {
        "mcpServers": {
          "jeeva": {
            "command": "jeeva",
            "args": ["mcp"]
          }
        }
      }

  Then in a Claude Code session:

      Use jeeva.resurrect with url=https://github.com/nanoporetech/medaka
      Use jeeva.pin with packages=["pysam","pod5"] date="2022-06-01"
      Use jeeva.registry_search with query="SV caller for ONT long reads"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from typing import Any


def _run_jeeva(*argv: str, timeout: int = 30) -> str:
    """Run a jeeva CLI subcommand and return its stdout as a string.

    This delegates to the same binary so the MCP server doesn't need to
    import every module eagerly — it stays lightweight.
    """
    cmd = [sys.executable, "-m", "sanjeevini.cli", *argv]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return json.dumps({
                "error": result.stderr.strip() or "non-zero exit",
                "stdout": result.stdout.strip(),
            })
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"jeeva timed out after {timeout}s"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool definitions and handlers
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "resurrect",
        "description": (
            "Revive a dead research repository into a callable integration contract. "
            "Jeeva reads the repo URL, writes its own resurrection plan and sanity check, "
            "then runs an autonomous build→repair loop in a Docker sandbox. "
            "Returns the path to the emitted contract directory or a progress message."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub URL of the target repo, e.g. https://github.com/nanoporetech/medaka",
                },
                "docker_host": {
                    "type": "string",
                    "description": "Optional remote Docker host, e.g. ssh://user@gpu-box",
                },
                "gpus": {
                    "type": "string",
                    "description": "GPU spec forwarded to docker run, e.g. 'all'",
                },
                "turns": {
                    "type": "integer",
                    "description": "Maximum repair-loop turns (default 60)",
                    "default": 60,
                },
                "budget_usd": {
                    "type": "number",
                    "description": "Hard cost cap in USD",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "pin",
        "description": (
            "Resolve a list of packages to the versions that were live on a given date. "
            "Supports PyPI (default), conda-forge/bioconda (--conda), and Bioconductor (--bioc). "
            "Returns install-ready version strings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Package names to pin, e.g. ['pysam', 'pod5', 'medaka']",
                },
                "date": {
                    "type": "string",
                    "description": "Target date in YYYY-MM-DD format",
                },
                "ecosystem": {
                    "type": "string",
                    "enum": ["pypi", "conda", "bioc"],
                    "description": "Package ecosystem (default: pypi)",
                    "default": "pypi",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra conda channels (only used when ecosystem=conda)",
                },
            },
            "required": ["packages", "date"],
        },
    },
    {
        "name": "decay_check",
        "description": (
            "Agent-free check: does this repo still install and run today? "
            "Clones the repo, attempts install (no repair), runs a smoke test, "
            "and returns a reason-coded verdict: naive_runs | install_fails | run_fails | unknown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub URL to check",
                },
                "sandbox": {
                    "type": "string",
                    "enum": ["host", "docker"],
                    "description": "host = venv/tmp; docker = strict benchmark parity",
                    "default": "docker",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "registry_search",
        "description": (
            "Semantic search over the Sanjeevini/Lazarus registry of revived tools. "
            "Use natural-language queries like 'SV caller for ONT', "
            "'chromatin accessibility from DNA sequence', or 'protein binding-site predictor'. "
            "Returns a ranked list of matching contracts with pull instructions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "top": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter: longread, proteomics, variant-calling, etc.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_pipeline",
        "description": (
            "Execute a Sanjeevini Compose pipeline from a YAML spec. "
            "Pass the YAML content directly or a path to a local file. "
            "Returns pipeline run status and output artifact paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pipeline_yaml": {
                    "type": "string",
                    "description": "Path to pipeline YAML file OR inline YAML content",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input overrides as key=value pairs",
                    "additionalProperties": {"type": "string"},
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate and type-check without running",
                    "default": False,
                },
                "docker_host": {
                    "type": "string",
                    "description": "Remote Docker host",
                },
            },
            "required": ["pipeline_yaml"],
        },
    },
]


async def _handle_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call and return a JSON string result."""

    if name == "resurrect":
        argv = ["resurrect", arguments["url"], "--json"]
        if arguments.get("docker_host"):
            argv += ["--docker-host", arguments["docker_host"]]
        if arguments.get("gpus"):
            argv += ["--gpus", arguments["gpus"]]
        if arguments.get("turns"):
            argv += ["--turns", str(arguments["turns"])]
        if arguments.get("budget_usd"):
            argv += ["--budget-usd", str(arguments["budget_usd"])]
        return _run_jeeva(*argv, timeout=3600)

    if name == "pin":
        eco = arguments.get("ecosystem", "pypi")
        argv = ["pin", "--date", arguments["date"], *arguments["packages"]]
        if eco == "conda":
            argv.append("--conda")
            for ch in arguments.get("channels", []):
                argv += ["--channel", ch]
        elif eco == "bioc":
            argv.append("--bioc")
        argv.append("--json")
        return _run_jeeva(*argv, timeout=120)

    if name == "decay_check":
        argv = ["decay-check", arguments["url"], "--json",
                "--sandbox", arguments.get("sandbox", "docker")]
        return _run_jeeva(*argv, timeout=600)

    if name == "registry_search":
        argv = ["registry", "search", arguments["query"],
                "--top", str(arguments.get("top", 5))]
        if arguments.get("domain"):
            argv += ["--domain", arguments["domain"]]
        return _run_jeeva(*argv, timeout=30)

    if name == "run_pipeline":
        yaml_arg = arguments["pipeline_yaml"]
        argv = ["run", yaml_arg]
        for k, v in arguments.get("inputs", {}).items():
            argv += ["--input", f"{k}={v}"]
        if arguments.get("dry_run"):
            argv.append("--dry-run")
        if arguments.get("docker_host"):
            argv += ["--docker-host", arguments["docker_host"]]
        return _run_jeeva(*argv, timeout=7200)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# MCP protocol (stdio transport — minimal, spec-compliant)
# ---------------------------------------------------------------------------

async def _stdio_server() -> None:
    """Run the MCP server over stdio (JSON-RPC 2.0 framing)."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.BaseProtocol, sys.stdout.buffer
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    def _send(obj: dict) -> None:
        line = json.dumps(obj) + "\n"
        writer.write(line.encode())

    async for raw in reader:
        try:
            msg = json.loads(raw.decode())
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            _send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "jeeva", "version": "0.1.0"},
                },
            })

        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": TOOLS},
            })

        elif method == "tools/call":
            params  = msg.get("params", {})
            name    = params.get("name", "")
            args    = params.get("arguments", {})
            content = await _handle_tool_call(name, args)
            _send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": content}],
                    "isError": False,
                },
            })

        elif method == "notifications/initialized":
            pass  # no response needed

        else:
            if msg_id is not None:
                _send({
                    "jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })

    await writer.drain()


def serve(args: argparse.Namespace) -> None:
    """Entry point called by the CLI 'jeeva mcp' subcommand."""
    if getattr(args, "host", "stdio") == "sse":
        # SSE transport — requires mcp[sse] extra
        try:
            from mcp.server.sse import SseServerTransport  # type: ignore[import]
            from mcp import Server  # type: ignore[import]
        except ImportError:
            print(
                "SSE transport requires: pip install 'sanjeevini-bio[mcp]'",
                file=sys.stderr,
            )
            sys.exit(1)
        # SSE server wiring would go here
        raise NotImplementedError("SSE transport not yet implemented")

    asyncio.run(_stdio_server())
