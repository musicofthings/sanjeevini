"""jeeva — the Sanjeevini command-line interface.

Entrypoint for all Sanjeevini subcommands.  Jeeva is the agent persona;
the CLI is the shell you use when you don't want the autonomous loop.

Usage
-----
    jeeva resurrect <github-url>           # Scout + autonomous repair loop
    jeeva pin --date 2021-03-01 <pkgs…>   # commit-era PyPI pin
    jeeva pin --conda --date 2021-03-01 <pkgs…>   # Bioconda/conda-forge pin
    jeeva run <pipeline.yaml>              # Compose: wire + execute bricks
    jeeva registry list                    # browse revived-tool catalog
    jeeva registry search "<query>"        # semantic search over catalog
    jeeva decay-check <github-url>         # agent-free "does it still run?"
    jeeva mcp                              # start MCP server (stdio)
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]


def _add_resurrect(sub: SubParsers) -> None:
    p = sub.add_parser("resurrect", help="Revive a dead research repo from a GitHub URL.")
    p.add_argument("url", help="GitHub URL of the target repo.")
    p.add_argument("--image", help="Override the base Docker image (skips Scout image selection).")
    p.add_argument(
        "--goal-file",
        metavar="FILE",
        help=(
            "Override the Scout's goal and/or sanity check. Plain text sets the goal; "
            "YAML with 'goal:' and/or 'sanity_check:' keys sets either or both. A "
            "sanity check supplied here must still be falsifiable."
        ),
    )
    p.add_argument(
        "--workdir", default="/workspace", help="Working directory inside the container."
    )
    p.add_argument(
        "--keep", action="store_true", help="Keep container after success (for inspection)."
    )
    p.add_argument("--turns", type=int, default=60, help="Maximum repair-loop turns (default: 60).")
    p.add_argument(
        "--escalate",
        type=int,
        default=1,
        metavar="N",
        help="On failure, retry on up to N alternate base images when the run's own "
        "errors justify one — e.g. Python 2 sources on a Python 3 image "
        "(default: 1; 0 disables). Total work is bounded by --turns x (1 + N).",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        help="Hard cost cap in USD (requires claude-agent-sdk ≥ 0.3).",
    )
    p.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Directory for turn-level checkpoints (enables resume).",
    )
    p.add_argument(
        "--no-scout",
        action="store_true",
        help="Skip the Scout entirely; requires --image and --goal-file.",
    )
    p.add_argument(
        "--docker-host",
        default=None,
        metavar="HOST",
        help="Remote Docker host, e.g. ssh://user@gpu-box.  Overrides DOCKER_HOST.",
    )
    p.add_argument("--gpus", default=None, help="GPU spec forwarded to docker run, e.g. 'all'.")


def _add_pin(sub: SubParsers) -> None:
    p = sub.add_parser("pin", help="Resolve packages to commit-era versions.")
    p.add_argument("packages", nargs="+", help="Package names to pin.")
    p.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Target date — packages will be pinned to the latest version live on this date.",
    )
    p.add_argument(
        "--conda",
        action="store_true",
        help="Pin against conda-forge + bioconda instead of PyPI.",
    )
    p.add_argument(
        "--bioc",
        action="store_true",
        help="Pin against a Bioconductor release (resolves from CRAN + Bioc).",
    )
    p.add_argument(
        "--channel", action="append", metavar="CH", help="Extra conda channels (repeatable)."
    )
    p.add_argument(
        "--json", action="store_true", help="Emit JSON instead of pip/conda install lines."
    )


def _add_run(sub: SubParsers) -> None:
    p = sub.add_parser("run", help="Execute a Sanjeevini Compose pipeline.")
    p.add_argument("pipeline", help="Path to pipeline YAML.")
    p.add_argument("--input", action="append", metavar="K=V", help="Input overrides (repeatable).")
    p.add_argument(
        "--registry", action="append", metavar="DIR", help="Local registry paths (repeatable)."
    )
    p.add_argument("--docker-host", default=None, metavar="HOST")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate YAML and type-check I/O without running.",
    )


def _add_registry(sub: SubParsers) -> None:
    p = sub.add_parser("registry", help="Browse or search the revived-tool registry.")
    rsub = p.add_subparsers(dest="registry_cmd", required=True)

    ls = rsub.add_parser("list", help="List all tools in the catalog.")
    ls.add_argument(
        "--domain", help="Filter by domain (e.g. longread-ont, proteomics, variant-calling)."
    )
    ls.add_argument(
        "--platform", help="Filter by sequencing platform (ont, pacbio_hifi, illumina)."
    )
    ls.add_argument("--json", action="store_true")

    sr = rsub.add_parser("search", help="Semantic search over the catalog.")
    sr.add_argument("query", help="Natural-language query, e.g. 'SV caller for ONT'.")
    sr.add_argument("--top", type=int, default=5)

    pull = rsub.add_parser("pull", help="Pull a tool's integration contract locally.")
    pull.add_argument("tool", help="Tool name or slug (e.g. sniffles2, dorado-align).")


def _add_decay_check(sub: SubParsers) -> None:
    p = sub.add_parser(
        "decay-check", help="Agent-free 'does this repo still install and run today?'"
    )
    p.add_argument("url", help="GitHub URL to check.")
    p.add_argument(
        "--sandbox",
        choices=["host", "docker"],
        default="docker",
        help="host = venv/temp; docker = strict benchmark parity (default).",
    )
    p.add_argument("--fail-on-decay", action="store_true", help="Exit 1 if repo is decayed.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable verdict.")


def _add_mcp(sub: SubParsers) -> None:
    p = sub.add_parser("mcp", help="Start Jeeva as an MCP server (stdio transport).")
    p.add_argument(
        "--host", default="stdio", choices=["stdio", "sse"], help="Transport (default: stdio)."
    )
    p.add_argument("--port", type=int, default=8765, help="Port for SSE transport.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jeeva",
        description=(
            "Jeeva — the Sanjeevini agent.\n"
            "Revives dead bioinformatics tools; first-class support for ONT/PacBio long-read\n"
            "tools and Nextflow/Snakemake/WDL workflows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="sanjeevini-bio 0.1.0")

    sub = parser.add_subparsers(dest="command", required=True)
    _add_resurrect(sub)
    _add_pin(sub)
    _add_run(sub)
    _add_registry(sub)
    _add_decay_check(sub)
    _add_mcp(sub)

    args = parser.parse_args(argv)

    # Dispatch — each module is imported lazily so startup stays fast.
    if args.command == "resurrect":
        from sanjeevini.repair.loop import ResurrectCommand

        ResurrectCommand(args).run()

    elif args.command == "pin":
        if args.conda:
            from sanjeevini.pinners.conda import CondaPinner

            CondaPinner(args).run()
        elif args.bioc:
            from sanjeevini.pinners.bioc import BiocPinner

            BiocPinner(args).run()
        else:
            from sanjeevini.pinners.pypi import PyPIPinner

            PyPIPinner(args).run()

    elif args.command == "run":
        from sanjeevini.compose.pipeline import ComposeCommand

        ComposeCommand(args).run()

    elif args.command == "registry":
        from sanjeevini.registry.catalog import RegistryCommand

        RegistryCommand(args).run()

    elif args.command == "decay-check":
        from sanjeevini.repair.loop import DecayCheckCommand

        DecayCheckCommand(args).run()

    elif args.command == "mcp":
        from sanjeevini.mcp.server import serve

        serve(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
