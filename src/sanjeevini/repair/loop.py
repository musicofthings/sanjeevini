"""The repair loop — Jeeva's core agentic organ (Phase 4).

The repair loop drives a bounded ``build → run → read-traceback → patch → retry``
cycle inside a :class:`~sanjeevini.sandbox.docker_sandbox.DockerSandbox` toward a
Scout's goal, then emits a verified integration contract. It extends the Lazarus
loop with four Sanjeevini additions:

* **Checkpoint-aware resume** — a run pointed at an existing ``checkpoint_dir``
  continues from the last known-good snapshot rather than from turn zero.
* **Workflow-aware dispatch** — :func:`select_plan` returns a
  :class:`~sanjeevini.scouts.workflow_scout.WorkflowResurrectionPlan` for
  Nextflow/Snakemake/WDL/CWL repos and a Python/R plan otherwise.
* **Long-read model-bundle pre-fetch** — hardware-heavy long-read tools declare
  GPU requirements in the emitted contract.
* **Cost tracking** — per-turn USD cost accumulates into ``PROVENANCE.json``.

Design for testability
-----------------------
The loop never declares ``PASS`` on the agent's word: a resurrection passes only
when a command the agent *marks as the sanity check* actually exits 0 inside the
sandbox. This keeps the scientific-correctness principle enforceable and makes
the whole organ unit-testable — the agent (:class:`RepairAgent`) and the sandbox
(:class:`SandboxProtocol`) are both injected, so :class:`RepairLoop` runs to
completion with fakes and no Docker daemon.

The autonomous LLM agent that plays :class:`RepairAgent` in production is driven
by ``claude-agent-sdk`` through the MCP server (``jeeva mcp``); the loop can also
be hand-driven with a :class:`ScriptedAgent`. The loop itself, the plan
dispatch, the contract emitter, and the decay check are all dependency-free.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import yaml

from sanjeevini import __version__ as JEEVA_VERSION
from sanjeevini.contracts.schema import ContractSchema, GenomicFileType, IOPort
from sanjeevini.sandbox.checkpoint import CheckpointStore, TurnRecord
from sanjeevini.sandbox.docker_sandbox import DockerError, DockerSandbox, ExecResult
from sanjeevini.scouts.python_scout import PythonResurrectionPlan, PythonScout
from sanjeevini.scouts.r_scout import RResurrectionPlan, RScout
from sanjeevini.scouts.repo import RepoSnapshot, parse_repo_url
from sanjeevini.scouts.workflow_scout import (
    WorkflowResurrectionPlan,
    build_resurrection_plan,
    detect_workflow_language,
)

Verdict = Literal["PASS", "TIMEOUT", "FAILED"]
ResurrectionPlan = PythonResurrectionPlan | RResurrectionPlan | WorkflowResurrectionPlan

# Consecutive unrecoverable container errors before the loop aborts (a dead
# container can't be repaired in-place, so spinning to the turn limit is waste).
_MAX_SANDBOX_ERRORS = 3
# Exit codes synthesised for sandbox failures so the agent sees them as tracebacks.
_RC_TIMEOUT = 124
_RC_SANDBOX_ERROR = 125

# Local files a Scout reads when a repo is already cloned to disk.
_LOCAL_FETCH_PATHS = (
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

# Starting runner image per workflow language (the "base image" for a workflow).
_WORKFLOW_BASE_IMAGES: dict[str, str] = {
    "nextflow": "nextflow/nextflow:23.10.1",
    "snakemake": "snakemake/snakemake:v7.32.4",
    "wdl": "ubuntu:22.04",
    "cwl": "ubuntu:22.04",
}


# ---------------------------------------------------------------------------
# Slugs and snapshots
# ---------------------------------------------------------------------------


def tool_slug(url_or_name: str) -> str:
    """Return a filesystem-safe slug for a repo URL or tool name.

    Args:
        url_or_name: A GitHub URL or a bare tool name.

    Returns:
        A lowercase ``[a-z0-9-]`` slug derived from the repository name.
    """
    try:
        _, name = parse_repo_url(url_or_name)
    except ValueError:
        name = url_or_name
    slug = name.strip().lower()
    slug = "".join(c if c.isalnum() else "-" for c in slug)
    return slug.strip("-") or "tool"


def snapshot_from_dir(repo_dir: Path, url: str) -> RepoSnapshot:
    """Build a :class:`RepoSnapshot` from an already-cloned repo directory.

    Args:
        repo_dir: Path to the checked-out repository.
        url: The repository URL (recorded on the snapshot).

    Returns:
        A snapshot populated from the files present on disk.
    """
    try:
        owner, name = parse_repo_url(url)
    except ValueError:
        owner, name = "local", repo_dir.name
    snapshot = RepoSnapshot(url=url, owner=owner, name=name)
    for path in _LOCAL_FETCH_PATHS:
        candidate = repo_dir / path
        if candidate.is_file():
            snapshot.files[path] = candidate.read_text(encoding="utf-8", errors="replace")
    return snapshot


async def select_plan(repo_dir: Path, url: str, *, confirm: bool = True) -> ResurrectionPlan:
    """Detect the repo kind and return the appropriate resurrection plan.

    Workflow repos (a Nextflow ``main.nf``, a ``Snakefile``, a ``*.wdl``/``*.cwl``)
    dispatch to the Workflow Scout; an R ``DESCRIPTION`` dispatches to the R
    Scout; everything else to the Python Scout.

    Args:
        repo_dir: Path to the cloned repository.
        url: The repository URL.
        confirm: Whether the Scout should pause for confirmation (Python/R only).

    Returns:
        A workflow, R, or Python resurrection plan.
    """
    language = detect_workflow_language(repo_dir)
    if language != "unknown":
        return build_resurrection_plan(repo_dir)

    snapshot = snapshot_from_dir(repo_dir, url)
    if snapshot.has("DESCRIPTION"):
        return await RScout(url, snapshot=snapshot).plan(confirm=confirm)
    return await PythonScout(url, snapshot=snapshot).plan(confirm=confirm)


# ---------------------------------------------------------------------------
# Normalised resurrection spec (bridges the three plan shapes)
# ---------------------------------------------------------------------------


@dataclass
class ResurrectionSpec:
    """The subset of a plan the repair loop and contract emitter need.

    Attributes:
        tool_slug: Filesystem-safe tool identifier.
        goal: Full goal statement for the loop.
        sanity_check: Falsifiable pass criterion.
        base_image: Docker image the resurrection starts from.
        repo_url: Source repository URL.
        repo_commit: Resolved commit the resurrection ran against.
        workflow_type: Contract ``workflow_type`` (python/r/nextflow/…).
        gpu_required: Whether the tool needs a GPU.
        entry_command: The tool's run command (workflow entry point; empty for
            Python/R, where the passing sanity command is used instead).
    """

    tool_slug: str
    goal: str
    sanity_check: str
    base_image: str
    repo_url: str = ""
    repo_commit: str = ""
    workflow_type: str = "python"
    gpu_required: bool = False
    entry_command: str = ""


def _gpu_from_image(base_image: str) -> bool:
    low = base_image.lower()
    return "cuda" in low or low.endswith("-gpu") or "-gpu" in low


def spec_from_plan(plan: ResurrectionPlan, *, url: str, repo_commit: str = "") -> ResurrectionSpec:
    """Normalise any Scout plan into a :class:`ResurrectionSpec`.

    Args:
        plan: A Python, R, or Workflow resurrection plan.
        url: The repository URL.
        repo_commit: Resolved commit hash, if known.

    Returns:
        The normalised spec used by :class:`RepairLoop` and the contract emitter.
    """
    slug = tool_slug(url)
    if isinstance(plan, WorkflowResurrectionPlan):
        base_image = _WORKFLOW_BASE_IMAGES.get(plan.language, "ubuntu:22.04")
        return ResurrectionSpec(
            tool_slug=slug,
            goal=f"Resurrect {slug} as a {plan.language} workflow. {plan.entry_point}",
            sanity_check=plan.sanity_check,
            base_image=base_image,
            repo_url=url,
            repo_commit=repo_commit,
            workflow_type=plan.language,
            gpu_required=False,
            entry_command=plan.entry_point,
        )
    if isinstance(plan, RResurrectionPlan):
        return ResurrectionSpec(
            tool_slug=slug,
            goal=plan.goal,
            sanity_check=plan.sanity_check,
            base_image=plan.base_image,
            repo_url=url,
            repo_commit=repo_commit,
            workflow_type="r",
            gpu_required=False,
        )
    return ResurrectionSpec(
        tool_slug=slug,
        goal=plan.goal,
        sanity_check=plan.sanity_check,
        base_image=plan.base_image,
        repo_url=url,
        repo_commit=repo_commit,
        workflow_type="python",
        gpu_required=_gpu_from_image(plan.base_image),
    )


# ---------------------------------------------------------------------------
# Agent / sandbox protocols and per-turn types
# ---------------------------------------------------------------------------


@dataclass
class RepairAction:
    """One action chosen by the agent for a turn.

    Attributes:
        kind: ``"exec"`` to run a command; ``"give_up"`` to abort as unresolvable.
        cmd: Argv list to execute (``kind == "exec"``).
        is_sanity_check: If ``True`` and the command exits 0, the run passes.
        patch: Unified-diff string applied this turn (recorded in provenance).
        bug_class: Classification of the bug the patch fixes (e.g. ``dead_endpoint``).
        bug_description: Human-readable description of the fix.
        reason: Why the agent gave up (``kind == "give_up"``).
        cost_usd: Agent cost accrued producing this action.
        timeout: Per-command timeout in seconds.
    """

    kind: Literal["exec", "give_up"]
    cmd: list[str] = field(default_factory=list)
    is_sanity_check: bool = False
    patch: str | None = None
    bug_class: str | None = None
    bug_description: str = ""
    reason: str = ""
    cost_usd: float = 0.0
    timeout: int = 300


@dataclass
class TurnOutcome:
    """The action taken on a turn and the sandbox result it produced."""

    action: RepairAction
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        """Whether the command succeeded (exit code 0)."""
        return self.returncode == 0


@dataclass
class LoopState:
    """The state handed to the agent before it chooses the next action.

    Attributes:
        turn: 1-based number of the turn about to run.
        max_turns: Hard turn ceiling for the run.
        goal: The resurrection goal.
        sanity_check: The falsifiable pass criterion.
        base_image: The starting Docker image.
        last_returncode: Exit code of the previous turn, or ``None`` on turn 1.
        last_stdout: Stdout of the previous turn.
        last_stderr: Stderr of the previous turn (the traceback to read).
        patch_history: Diffs applied so far this run.
        history: Prior :class:`TurnOutcome` records this run.
    """

    turn: int
    max_turns: int
    goal: str
    sanity_check: str
    base_image: str
    last_returncode: int | None
    last_stdout: str
    last_stderr: str
    patch_history: list[str]
    history: list[TurnOutcome]


class RepairAgent(Protocol):
    """Chooses the next action given the current loop state."""

    def next_action(self, state: LoopState) -> RepairAction:
        """Return the action to run for ``state.turn``."""
        ...


class SandboxProtocol(Protocol):
    """The sandbox surface the repair loop drives (satisfied by DockerSandbox)."""

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult: ...
    def snapshot(self, tag: str) -> str: ...
    def last_successful_snapshot(self) -> str | None: ...
    @property
    def previous_turns(self) -> list[TurnRecord]: ...


class ScriptedAgent:
    """A :class:`RepairAgent` that replays a fixed list of actions.

    Used by tests and for deterministic reruns. When the script is exhausted it
    gives up, which drives the loop to a ``FAILED`` verdict rather than looping
    forever.
    """

    def __init__(self, actions: list[RepairAction]) -> None:
        """Store the actions to replay in order.

        Args:
            actions: The actions to return, one per :meth:`next_action` call.
        """
        self._actions = list(actions)
        self._index = 0

    def next_action(self, state: LoopState) -> RepairAction:
        """Return the next scripted action, or a give-up once exhausted."""
        if self._index >= len(self._actions):
            return RepairAction(kind="give_up", reason="script exhausted")
        action = self._actions[self._index]
        self._index += 1
        return action


# ---------------------------------------------------------------------------
# Run outcome
# ---------------------------------------------------------------------------


@dataclass
class RepairOutcome:
    """The terminal result of a repair run.

    Attributes:
        verdict: ``PASS``, ``TIMEOUT``, or ``FAILED``.
        turns: Total turns executed (including any resumed from checkpoint).
        cost_usd: Total agent cost across the run.
        reason: Explanation for a non-``PASS`` verdict.
        bugs_fixed: One dict per applied patch (class/description/patch).
        sanity_cmd: The command that proved the sanity check, on ``PASS``.
        final_image: The committed image tag banking the passing state.
        contract_dir: Directory the contract/provenance was written to.
    """

    verdict: Verdict
    turns: int
    cost_usd: float
    reason: str = ""
    bugs_fixed: list[dict[str, str]] = field(default_factory=list)
    sanity_cmd: list[str] = field(default_factory=list)
    reproduction: list[list[str]] = field(default_factory=list)
    final_image: str = ""
    contract_dir: Path | None = None


# ---------------------------------------------------------------------------
# The repair loop
# ---------------------------------------------------------------------------


def _utc_today() -> str:
    """Return today's date (UTC) as an ISO 8601 ``YYYY-MM-DD`` string."""
    return datetime.now(timezone.utc).date().isoformat()


class RepairLoop:
    """Drives a bounded resurrection loop and emits the contract on completion."""

    def __init__(
        self,
        spec: ResurrectionSpec,
        sandbox: SandboxProtocol,
        agent: RepairAgent,
        *,
        max_turns: int = 60,
        contracts_root: Path | str = "contracts",
        today: str | None = None,
    ) -> None:
        """Configure a repair loop.

        Args:
            spec: The normalised resurrection spec.
            sandbox: A started sandbox to execute inside.
            agent: The agent choosing actions each turn.
            max_turns: Hard ceiling on total turns (including resumed turns).
            contracts_root: Directory contracts are emitted under.
            today: Override for the resurrection date (defaults to today, UTC).
        """
        self.spec = spec
        self.sandbox = sandbox
        self.agent = agent
        self.max_turns = max_turns
        self.contracts_root = Path(contracts_root)
        self.today = today or _utc_today()

    def run(self) -> RepairOutcome:
        """Run the loop to a terminal verdict and emit the contract.

        Returns:
            The :class:`RepairOutcome`. ``PASS`` emits the full contract set;
            ``TIMEOUT``/``FAILED`` emit ``PROVENANCE.json`` only.
        """
        prior = list(self.sandbox.previous_turns)
        turn = len(prior)
        last_rc: int | None = prior[-1].returncode if prior else None
        last_out = prior[-1].stdout if prior else ""
        last_err = prior[-1].stderr if prior else ""

        history: list[TurnOutcome] = []
        patch_history: list[str] = []
        bugs_fixed: list[dict[str, str]] = []
        cost = 0.0
        verdict: Verdict | None = None
        reason = ""
        sanity_cmd: list[str] = []
        reproduction: list[list[str]] = []
        sandbox_errors = 0

        while turn < self.max_turns:
            state = LoopState(
                turn=turn + 1,
                max_turns=self.max_turns,
                goal=self.spec.goal,
                sanity_check=self.spec.sanity_check,
                base_image=self.spec.base_image,
                last_returncode=last_rc,
                last_stdout=last_out,
                last_stderr=last_err,
                patch_history=list(patch_history),
                history=list(history),
            )
            # A hard agent/API failure (after the agent's own retries) must not
            # crash the run — end it gracefully so a contract is still emitted.
            try:
                action = self.agent.next_action(state)
            except Exception as exc:
                verdict = "FAILED"
                reason = f"agent call failed: {type(exc).__name__}: {exc}"
                break
            cost += action.cost_usd

            if action.kind == "give_up":
                verdict = "FAILED"
                reason = action.reason or "agent signalled the resurrection is unresolvable"
                break

            result, container_error = self._exec(action)
            turn += 1
            history.append(
                TurnOutcome(
                    action=action,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_s=result.duration_s,
                )
            )
            last_rc, last_out, last_err = result.returncode, result.stdout, result.stderr

            # Record the successful commands, in order — this is the reproduction
            # recipe emitted as a self-contained smoke test.
            if result.ok:
                reproduction.append(list(action.cmd))

            # A dead container can't be repaired in place; bail after a few in a row.
            if container_error:
                sandbox_errors += 1
                if sandbox_errors >= _MAX_SANDBOX_ERRORS:
                    verdict = "FAILED"
                    reason = "sandbox became unrecoverable (repeated container errors)"
                    break
            else:
                sandbox_errors = 0

            if action.patch:
                patch_history.append(action.patch)
                bugs_fixed.append(
                    {
                        "class": action.bug_class or "unknown",
                        "description": action.bug_description,
                        "patch": action.patch,
                    }
                )

            if result.ok and action.is_sanity_check:
                verdict = "PASS"
                sanity_cmd = list(action.cmd)
                break

        if verdict is None:
            verdict = "TIMEOUT"
            reason = f"turn limit ({self.max_turns}) reached before the sanity check passed"

        final_image = ""
        if verdict == "PASS":
            final_tag = f"sanjeevini/{self.spec.tool_slug}:resurrected"
            try:
                final_image = self.sandbox.snapshot(final_tag)
            except DockerError:
                final_image = final_tag

        outcome = RepairOutcome(
            verdict=verdict,
            turns=turn,
            cost_usd=round(cost, 6),
            reason=reason,
            bugs_fixed=bugs_fixed,
            sanity_cmd=sanity_cmd,
            reproduction=reproduction,
            final_image=final_image,
        )
        outcome.contract_dir = self._emit(outcome)
        return outcome

    def _exec(self, action: RepairAction) -> tuple[ExecResult, bool]:
        """Run one action, turning sandbox failures into a failed result.

        A command timeout or container error is converted into a non-zero
        :class:`ExecResult` (surfaced to the agent as the next turn's traceback)
        instead of propagating, so the loop self-corrects rather than crashing.

        Args:
            action: The exec action to run.

        Returns:
            ``(result, container_error)`` — ``container_error`` is ``True`` only
            for an unrecoverable container fault (not a plain timeout).
        """
        try:
            return self.sandbox.exec(action.cmd, timeout=action.timeout), False
        except TimeoutError as exc:
            stderr = f"[command timed out after {action.timeout}s] {exc}"
            return ExecResult(_RC_TIMEOUT, "", stderr, float(action.timeout)), False
        except DockerError as exc:
            return ExecResult(_RC_SANDBOX_ERROR, "", f"[sandbox error] {exc}", 0.0), True

    # ---- contract emission -------------------------------------------------

    def _slug_dir(self) -> Path:
        d = self.contracts_root / self.spec.tool_slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _emit(self, outcome: RepairOutcome) -> Path:
        """Write the contract for ``outcome`` and return its directory.

        On ``PASS`` the full integration package is written; otherwise only
        ``PROVENANCE.json`` (a resurrection record exists for every completed
        run, pass or fail).
        """
        slug_dir = self._slug_dir()
        (slug_dir / "PROVENANCE.json").write_text(
            json.dumps(self._provenance(outcome), indent=2) + "\n", encoding="utf-8"
        )
        if outcome.verdict != "PASS":
            return slug_dir

        entry = outcome.sanity_cmd or (
            shlex.split(self.spec.entry_command) if self.spec.entry_command else []
        )
        (slug_dir / "contract.yaml").write_text(self._contract_yaml(), encoding="utf-8")
        (slug_dir / "Dockerfile").write_text(self._dockerfile(entry), encoding="utf-8")
        (slug_dir / "predict.py").write_text(self._predict_py(entry), encoding="utf-8")
        (slug_dir / "smoke_test.sh").write_text(
            self._smoke_test(outcome), encoding="utf-8"
        )
        (slug_dir / "REPRODUCE.md").write_text(
            self._reproduce_md(outcome), encoding="utf-8"
        )
        return slug_dir

    def _schema(self) -> ContractSchema:
        return ContractSchema(
            inputs=[
                IOPort(name="input", type=GenomicFileType.ANY, description="primary input")
            ],
            outputs=[
                IOPort(name="output", type=GenomicFileType.ANY, description="primary output")
            ],
            gpu_required=self.spec.gpu_required,
            workflow_type=self.spec.workflow_type,  # type: ignore[arg-type]
        )

    def _contract_yaml(self) -> str:
        payload = {
            "slug": self.spec.tool_slug,
            "repo_url": self.spec.repo_url,
            "base_image": self.spec.base_image,
            "schema": json.loads(self._schema().model_dump_json()),
        }
        dumped: str = yaml.safe_dump(payload, sort_keys=False)
        return dumped

    def _dockerfile(self, entry: list[str]) -> str:
        cmd_json = json.dumps(entry) if entry else '["bash", "smoke_test.sh"]'
        return (
            f"# Pinned, verified Dockerfile for {self.spec.tool_slug} "
            f"(emitted by Jeeva {JEEVA_VERSION}).\n"
            f"FROM {self.spec.base_image}\n"
            f"WORKDIR /work\n"
            f"COPY predict.py smoke_test.sh ./\n"
            f'CMD {cmd_json}\n'
        )

    def _predict_py(self, entry: list[str]) -> str:
        return _PREDICT_TEMPLATE.format(
            slug=self.spec.tool_slug,
            version=JEEVA_VERSION,
            cmd=json.dumps(entry),
        )

    def _smoke_test(self, outcome: RepairOutcome) -> str:
        """Emit a self-contained reproduction script.

        Replays every command that succeeded during the run, in order, ending
        with the sanity check. Each command runs in its own subshell starting
        from the script's directory — exactly as the sandbox executed it (a
        fresh working directory per exec) — so ``cd repo && …`` steps replay
        correctly. The repo is cloned first if it isn't already present, making
        the script runnable from a bare base image, not just the final image.
        """
        header = (
            "#!/usr/bin/env bash\n"
            f"# Self-contained reproduction of {self.spec.tool_slug} "
            f"(emitted by Jeeva {JEEVA_VERSION}).\n"
            f"# Replays the verified command chain; success criterion:\n"
            f"#   {self.spec.sanity_check}\n"
            "set -euo pipefail\n"
            'cd "$(cd "$(dirname "$0")" && pwd)"\n'
        )
        if not outcome.reproduction:
            return header + "echo 'no reproduction was captured' >&2\nexit 1\n"

        lines = []
        if self.spec.repo_url:
            lines.append(f"[ -d repo ] || git clone --depth 1 -- {self.spec.repo_url} repo")
        lines += [f"( {shlex.join(cmd)} )" for cmd in outcome.reproduction]
        return header + "\n".join(lines) + "\n"

    def _reproduce_md(self, outcome: RepairOutcome) -> str:
        verdict_line = {
            "PASS": "PASS — the sanity-check command exited 0 on the test input.",
            "TIMEOUT": f"OFF — {outcome.reason}",
            "FAILED": f"OFF — {outcome.reason}",
        }[outcome.verdict]
        return (
            f"# Reproduction certificate — {self.spec.tool_slug}\n\n"
            f"- **Verdict:** {outcome.verdict}\n"
            f"- **Repository:** {self.spec.repo_url}\n"
            f"- **Base image:** {self.spec.base_image}\n"
            f"- **Final image:** {outcome.final_image or '—'}\n"
            f"- **Jeeva version:** {JEEVA_VERSION}\n"
            f"- **Resurrection date:** {self.today}\n"
            f"- **Turns:** {outcome.turns}\n"
            f"- **Cost (USD):** {outcome.cost_usd}\n\n"
            f"## Sanity check\n\n{self.spec.sanity_check}\n\n"
            f"## Result\n\n{verdict_line}\n"
            f"{self._reproduction_section(outcome)}"
        )

    def _reproduction_section(self, outcome: RepairOutcome) -> str:
        """Return the Markdown block listing the verified reproduction commands."""
        if not outcome.reproduction:
            return ""
        steps = "\n".join(shlex.join(cmd) for cmd in outcome.reproduction)
        return (
            "\n## Reproduce\n\n"
            "`bash smoke_test.sh` replays the verified command chain "
            "(clones the repo if needed, then):\n\n"
            f"```bash\n{steps}\n```\n"
        )

    def _provenance(self, outcome: RepairOutcome) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "tool": self.spec.tool_slug,
            "repo_url": self.spec.repo_url,
            "repo_commit": self.spec.repo_commit,
            "resurrection_date": self.today,
            "jeeva_version": JEEVA_VERSION,
            "turn_count": outcome.turns,
            "cost_usd": outcome.cost_usd,
            "base_image": self.spec.base_image,
            "final_image": outcome.final_image,
            "bugs_fixed": outcome.bugs_fixed,
            "sanity_check_verdict": outcome.verdict,
            "sanity_check_metric": self.spec.sanity_check,
        }


_PREDICT_TEMPLATE = '''#!/usr/bin/env python3
"""Minimal runnable entry point for {slug}, emitted by Jeeva {version}.

Usage:
    INPUT=<path> OUTDIR=<dir> python predict.py
"""
from __future__ import annotations

import os
import subprocess
import sys

# The minimal command Jeeva verified reproduces the tool's headline output.
COMMAND = {cmd}


def main() -> int:
    outdir = os.environ.get("OUTDIR", "out")
    os.makedirs(outdir, exist_ok=True)
    if not COMMAND:
        print("no verified command was recorded for this tool", file=sys.stderr)
        return 1
    print(f"[jeeva:{slug}] running: {{' '.join(COMMAND)}}", file=sys.stderr)
    return subprocess.run(COMMAND, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# Decay check (agent-free "does it still install and run today?")
# ---------------------------------------------------------------------------

DecayVerdictName = Literal["naive_runs", "install_fails", "run_fails", "unknown"]

# probe(url, sandbox_mode, docker_host) -> (naive_runs, stage, reason)
DecayProbe = Callable[[str, str, str | None], "tuple[bool | None, str, str]"]


@dataclass
class DecayVerdict:
    """The result of a decay check.

    Attributes:
        url: The repository checked.
        verdict: ``naive_runs``, ``install_fails``, ``run_fails``, or ``unknown``.
        stage: The protocol stage the verdict was decided at.
        reason: A short machine-readable reason token.
    """

    url: str
    verdict: DecayVerdictName
    stage: str
    reason: str

    @property
    def decayed(self) -> bool:
        """Whether the repo failed to run naively (a conclusive non-run)."""
        return self.verdict in ("install_fails", "run_fails")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict with keys verdict, stage, reason, url."""
        return {
            "verdict": self.verdict,
            "stage": self.stage,
            "reason": self.reason,
            "url": self.url,
        }


def classify_decay(naive_runs: bool | None, stage: str) -> DecayVerdictName:
    """Map a ``(naive_runs, stage)`` probe result to a decay verdict.

    Args:
        naive_runs: Whether the shipped example ran (``None`` if inconclusive).
        stage: The protocol stage (``clone``/``install``/``example``/…).

    Returns:
        The decay verdict name.
    """
    if naive_runs is True:
        return "naive_runs"
    if naive_runs is None:
        return "unknown"
    if stage == "example":
        return "run_fails"
    if stage in ("clone", "install"):
        return "install_fails"
    return "unknown"


# The fixed decay protocol: clone, install from the repo's own files, run a
# shipped example. Prints one machine-readable ===DECAY=== line.
_DECAY_SCRIPT = r"""
set +e
export PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive
verdict() { echo "===DECAY=== naive_runs=$1 stage=$2 reason=$3"; exit 0; }
TO="timeout 1200"; command -v timeout >/dev/null 2>&1 || TO=""
command -v git >/dev/null 2>&1 || verdict 0 clone needs_git
REPODIR=$(mktemp -d)/repo
git clone --depth 1 -- "REPO_URL" "$REPODIR" >/tmp/clone.log 2>&1 \
  || verdict 0 clone clone_failed
cd "$REPODIR"
if [ -f environment.yml ] && command -v conda >/dev/null 2>&1; then
  $TO conda env create -q -f environment.yml -p ./.env >/tmp/inst.log 2>&1 \
    || verdict 0 install conda_env_failed
  PY="./.env/bin/python"
elif [ -f requirements.txt ]; then
  $TO python -m pip install -q -r requirements.txt >/tmp/inst.log 2>&1 \
    || verdict 0 install pip_requirements_failed
  PY=python
elif [ -f setup.py ] || [ -f pyproject.toml ]; then
  $TO python -m pip install -q . >/tmp/inst.log 2>&1 \
    || verdict 0 install pip_install_failed
  PY=python
elif [ -f DESCRIPTION ] && command -v Rscript >/dev/null 2>&1; then
  $TO Rscript -e "
    if (!requireNamespace('remotes', quietly=TRUE))
      install.packages('remotes', repos='https://cloud.r-project.org');
    remotes::install_local('.', dependencies=TRUE, upgrade='never')
  " >/tmp/inst.log 2>&1 || verdict 0 install R_install_failed
  PY=""
else
  verdict 0 install no_install_manifest
fi
EX=""
for d in example examples demo demos tutorial tutorials quickstart; do
  f=$(ls $d/*.py $d/*.R $d/*.sh 2>/dev/null | grep -vi test | head -1)
  if [ -n "$f" ]; then
    case "$f" in
      *.py) EX="$PY $f";;
      *.R) EX="Rscript $f";;
      *.sh) EX="bash $f";;
    esac
    break
  fi
done
if [ -z "$EX" ]; then
  if [ -f DESCRIPTION ]; then
    P=$(grep -i '^Package:' DESCRIPTION | head -1 \
      | sed 's/[Pp]ackage:[[:space:]]*//' | tr -d '\r')
    EX="Rscript -e library($P)"
  else
    P=$(basename "$(pwd)" | tr '-' '_'); EX="$PY -c import_$P"
  fi
fi
$TO bash -lc "$EX" >/tmp/ex.log 2>&1 \
  && verdict 1 example ran_ok || verdict 0 example example_failed
"""

_DECAY_LINE = "===DECAY=== naive_runs="

# A repo URL is interpolated into the decay shell script; allow only characters
# that cannot break out of the double-quoted context or inject a command.
_SAFE_URL_RE = re.compile(r"^[A-Za-z0-9._:/@~+-]+$")


def _parse_decay_output(output: str) -> tuple[bool | None, str, str]:
    """Parse a ``===DECAY===`` line from probe output into (runs, stage, reason)."""
    for line in output.splitlines():
        if not line.startswith(_DECAY_LINE):
            continue
        parts: dict[str, str] = {}
        for token in line.split():
            if token.startswith("==="):
                continue
            key, sep, value = token.partition("=")
            if sep:
                parts[key] = value
        runs = parts.get("naive_runs")
        return (
            (runs == "1") if runs in ("0", "1") else None,
            parts.get("stage", "unknown"),
            parts.get("reason", "no_reason"),
        )
    return None, "unknown", "no_verdict_parsed"


def _docker_decay_probe(
    url: str, sandbox_mode: str, docker_host: str | None
) -> tuple[bool | None, str, str]:
    """Default probe: run the decay protocol on the host or in a fresh container."""
    if not _SAFE_URL_RE.match(url):
        return None, "error", "unsafe_url"
    script = _DECAY_SCRIPT.replace("REPO_URL", url)
    if sandbox_mode == "host":
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=1900,
            check=False,
        )
        return _parse_decay_output((proc.stdout or "") + "\n" + (proc.stderr or ""))

    box = DockerSandbox("continuumio/miniconda3", workdir="/", docker_host=docker_host)
    box.start()
    try:
        result = box.exec(["bash", "-lc", script], timeout=1900)
    finally:
        box.stop(force=True)
    return _parse_decay_output(result.stdout + "\n" + result.stderr)


def run_decay_check(
    url: str,
    *,
    sandbox_mode: str = "docker",
    docker_host: str | None = None,
    probe: DecayProbe | None = None,
) -> DecayVerdict:
    """Run the fixed decay protocol against ``url`` and classify the verdict.

    Args:
        url: The GitHub repository URL to check.
        sandbox_mode: ``"docker"`` (fresh container) or ``"host"``.
        docker_host: Remote Docker endpoint, if any.
        probe: Injected probe returning ``(naive_runs, stage, reason)``; defaults
            to the real docker/host protocol runner.

    Returns:
        The :class:`DecayVerdict`.
    """
    probe = probe or _docker_decay_probe
    naive_runs, stage, reason = probe(url, sandbox_mode, docker_host)
    return DecayVerdict(
        url=url,
        verdict=classify_decay(naive_runs, stage),
        stage=stage,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


class ResurrectCommand:
    """CLI handler for ``jeeva resurrect``.

    Clones the repo, runs the appropriate Scout, starts a sandbox (resuming from
    a checkpoint if present), drives the repair loop with the autonomous agent,
    and emits the integration contract.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: The ``resurrect`` subparser namespace.
        """
        self.args = args

    def run(self) -> None:
        """Execute the full resurrection pipeline and print the contract path."""
        import asyncio

        url = self.args.url
        with tempfile.TemporaryDirectory(prefix="jeeva-resurrect-") as tmp:
            repo_dir = Path(tmp) / "repo"
            commit = self._clone(url, repo_dir)
            plan = asyncio.run(
                select_plan(repo_dir, url, confirm=not self.args.no_scout)
            )
            spec = spec_from_plan(plan, url=url, repo_commit=commit)
            if self.args.image:
                spec.base_image = self.args.image

            checkpoint_dir = (
                Path(self.args.checkpoint_dir) if self.args.checkpoint_dir else None
            )
            # Resume from the last good snapshot when this checkpoint has prior turns.
            resume_image: str | None = None
            if checkpoint_dir is not None:
                resume_image = CheckpointStore(checkpoint_dir).last_successful_snapshot()

            agent = self._build_agent(spec)  # fail fast before starting a container
            sandbox = DockerSandbox(
                resume_image or spec.base_image,
                workdir=self.args.workdir,
                docker_host=self.args.docker_host,
                gpus=self.args.gpus,
                checkpoint_dir=checkpoint_dir,
            )
            sandbox.start()
            try:
                if resume_image is None:
                    # Fresh run: seed the container with the checkout. On resume the
                    # snapshot already carries the repo and any in-container edits.
                    sandbox.copy_in(repo_dir, self.args.workdir)
                loop = RepairLoop(spec, sandbox, agent, max_turns=self.args.turns)
                outcome = loop.run()
            finally:
                if not self.args.keep:
                    sandbox.stop(force=True)

        self._report(outcome)

    @staticmethod
    def _clone(url: str, dest: Path) -> str:
        """Shallow-clone ``url`` into ``dest`` and return the resolved commit.

        Exits cleanly (status 1) rather than dumping a traceback if the clone
        fails (e.g. the URL is wrong or the repo is private/unreachable).
        """
        clone = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "clone", "--depth", "1", "--", url, str(dest)],
            check=False,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            tail = (clone.stderr or clone.stdout).strip().splitlines()[-1:] or ["unknown error"]
            print(f"could not clone {url}: {tail[0]}", file=sys.stderr)
            sys.exit(1)
        head = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return head.stdout.strip()

    def _build_agent(self, spec: ResurrectionSpec) -> RepairAgent:
        """Build the autonomous LLM agent that plays :class:`RepairAgent`.

        Args:
            spec: The resurrection spec to plan against.

        Returns:
            An :class:`~sanjeevini.repair.agent.LLMRepairAgent`.

        Raises:
            RuntimeError: If the ``anthropic`` package (the ``[agent]`` extra) is
                not installed.
        """
        from sanjeevini.repair.agent import LLMRepairAgent

        return LLMRepairAgent(spec)

    def _report(self, outcome: RepairOutcome) -> None:
        """Print the verdict, contract directory, and smoke-test command."""
        print(f"verdict      : {outcome.verdict}")
        print(f"turns        : {outcome.turns}")
        print(f"cost (usd)   : {outcome.cost_usd}")
        if outcome.reason:
            print(f"reason       : {outcome.reason}")
        if outcome.contract_dir is not None:
            print(f"contract     : {outcome.contract_dir}")
            if outcome.verdict == "PASS":
                print(f"smoke test   : bash {outcome.contract_dir / 'smoke_test.sh'}")


class DecayCheckCommand:
    """CLI handler for ``jeeva decay-check``."""

    def __init__(self, args: argparse.Namespace, *, probe: DecayProbe | None = None) -> None:
        """Store parsed CLI arguments and an optional injected probe.

        Args:
            args: The ``decay-check`` subparser namespace.
            probe: Injected decay probe (for tests); defaults to the real one.
        """
        self.args = args
        self._probe = probe

    def run(self) -> None:
        """Run the decay check, print the verdict, and honour ``--fail-on-decay``."""
        result = run_decay_check(
            self.args.url,
            sandbox_mode=self.args.sandbox,
            probe=self._probe,
        )
        if self.args.json:
            print(json.dumps(result.to_dict()))
        else:
            print(f"{result.verdict}  ({result.stage}: {result.reason})  {result.url}")
        if self.args.fail_on_decay and result.verdict != "naive_runs":
            sys.exit(1)
