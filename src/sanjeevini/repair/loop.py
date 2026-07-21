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
import contextlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import yaml

from sanjeevini import __version__ as JEEVA_VERSION
from sanjeevini.contracts.output_type import extensions_for_check
from sanjeevini.contracts.schema import ContractSchema, GenomicFileType, IOPort
from sanjeevini.repair.escalation import AttemptRecord, EscalatingResurrection
from sanjeevini.repair.knowledge import (
    KnowledgeStore,
    default_store,
    error_signature,
    lessons_from_bugs,
)
from sanjeevini.sandbox.checkpoint import CheckpointStore, TurnRecord
from sanjeevini.sandbox.docker_sandbox import DockerError, DockerSandbox, ExecResult
from sanjeevini.scouts.python_scout import (
    PythonResurrectionPlan,
    PythonScout,
    ensure_falsifiable,
)
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
        framework: Canonical framework label (e.g. ``tensorflow-1.x``), used to
            score which prior lessons are relevant to this resurrection.
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
    framework: str = ""


@dataclass
class GoalOverride:
    """A goal and/or sanity check supplied by ``--goal-file``.

    Attributes:
        goal: Replacement goal statement, or ``""`` to keep the Scout's.
        sanity_check: Replacement pass criterion, or ``""`` to keep the Scout's.
    """

    goal: str = ""
    sanity_check: str = ""


def parse_goal_file(path: Path) -> GoalOverride:
    """Parse a ``--goal-file`` into a :class:`GoalOverride`.

    Accepts either plain text (the whole file is the goal) or YAML with ``goal``
    and/or ``sanity_check`` keys. A supplied sanity check is held to the same
    falsifiability bar as a Scout-generated one — overriding the criterion must
    not be a way to smuggle in an unfalsifiable claim.

    Args:
        path: Path to the goal file.

    Returns:
        The parsed override.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the YAML is malformed, both keys are absent from a
            mapping, or the sanity check is not falsifiable.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")

    parsed: object = None
    with contextlib.suppress(yaml.YAMLError):
        parsed = yaml.safe_load(text)

    if isinstance(parsed, dict):
        goal = str(parsed.get("goal", "") or "").strip()
        sanity_check = str(parsed.get("sanity_check", "") or "").strip()
        if not goal and not sanity_check:
            raise ValueError(f"{path} has neither a 'goal:' nor a 'sanity_check:' key")
    else:
        goal, sanity_check = text, ""

    if sanity_check:
        ensure_falsifiable(sanity_check)
    return GoalOverride(goal=goal, sanity_check=sanity_check)


def _apply_override(spec: ResurrectionSpec, override: GoalOverride) -> None:
    """Apply a :class:`GoalOverride` in place, leaving empty fields untouched."""
    if override.goal:
        spec.goal = override.goal
    if override.sanity_check:
        spec.sanity_check = override.sanity_check


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
            framework=plan.language,
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
            framework="r",
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
        framework=plan.framework,
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
        notes: Durable findings the agent wants carried into later turns — its
            only memory of this run besides the previous command's output.
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
    notes: str = ""


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
        notes: Findings the agent recorded on earlier turns — its working memory.
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
    notes: list[str] = field(default_factory=list)


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
class SanityEvidence:
    """Whether the sanity check's structural claim is backed by real artefacts.

    Attributes:
        status: ``supported`` (matching files found), ``unsupported`` (the check
            names a type but no such file exists), ``untyped`` (the check makes
            no structural claim to verify), or ``unknown`` (the probe failed).
        claimed_extensions: Extensions implied by the check's claimed type.
        found: Paths of matching non-empty files (capped).
    """

    status: Literal["supported", "unsupported", "untyped", "unknown"] = "untyped"
    claimed_extensions: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)

    @property
    def contradicts_claim(self) -> bool:
        """Whether the check named an output type that never materialised."""
        return self.status == "unsupported"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict for the provenance record."""
        return {
            "status": self.status,
            "claimed_extensions": self.claimed_extensions,
            "found": self.found,
        }


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
        evidence: Whether the sanity check's structural claim is backed by files
            the run actually produced (a qualifier on PASS, never a verdict).
        blockers: One compact signature per failing turn, oldest first. Lossy by
            design — enough to diagnose *what kind* of wall the run hit, which is
            what self-escalation needs to pick a better base image.
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
    evidence: SanityEvidence = field(default_factory=SanityEvidence)
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The repair loop
# ---------------------------------------------------------------------------


# Commands that only inspect state. Re-running one with no intervening change is
# always a no-op, which makes it safe to suppress — unlike a build or a test,
# where re-running after a fix is the whole point.
_INSPECT_VERBS = frozenset(
    {
        "ls",
        "cat",
        "sed",
        "grep",
        "head",
        "tail",
        "find",
        "wc",
        "file",
        "stat",
        "awk",
        "less",
        "more",
        "nl",
        "tree",
        "diff",
        "readlink",
        "basename",
        "dirname",
    }
)
# Verbs that neither inspect nor change anything, so they never decide the answer.
_NEUTRAL_VERBS = frozenset({"cd", "echo", "pwd", "which", "true", "type", "command"})
# Any redirection makes a command a writer, however innocent its verb looks —
# `cat > prove.py << EOF` is how the agent writes files.
_REDIRECT_RE = re.compile(r"(?<![0-9<>])>{1,2}(?!&)")
_SEGMENT_RE = re.compile(r"&&|\|\||;|\|")

# Exit code synthesised for a suppressed repeat.
_RC_NOOP_REPEAT = 126

# Return codes the loop itself invents. Their text is our own prose, not the
# tool's, so it must never be mistaken for evidence about the environment.
_SYNTHETIC_RCS = frozenset({_RC_TIMEOUT, _RC_SANDBOX_ERROR, _RC_NOOP_REPEAT})


def is_read_only(cmd: list[str]) -> bool:
    """Return whether ``cmd`` only inspects state and cannot change anything.

    Handles the shapes agents actually emit — ``cd /repo && grep …``, pipelines,
    and ``&&`` chains — by requiring *every* segment to be inspective or neutral.
    A single redirection anywhere disqualifies the whole command, so a heredoc
    like ``cat > prove.py << EOF`` is correctly treated as a write.

    Args:
        cmd: The argv list to classify.

    Returns:
        ``True`` only if running the command again could change nothing.
    """
    if not cmd:
        return False
    payload = cmd[-1]
    if _REDIRECT_RE.search(payload):
        return False

    saw_inspection = False
    for segment in _SEGMENT_RE.split(payload):
        words = segment.strip().split()
        if not words:
            continue
        verb = words[0]
        if verb in _INSPECT_VERBS:
            saw_inspection = True
        elif verb not in _NEUTRAL_VERBS:
            return False
    return saw_inspection


# Working-memory bounds: enough to remember a repo's shape, small enough that the
# prompt cannot grow without limit over a long run.
_MAX_NOTES = 40
_MAX_NOTE_CHARS = 6000
# Enough distinct walls to diagnose a failure; more is repetition of the same one.
_MAX_BLOCKERS = 12


def _record_blocker(blockers: list[str], result: ExecResult) -> None:
    """Append this failed turn's error signature, deduplicated and bounded.

    Falls back to stdout because agents routinely pipe ``2>&1``, which leaves
    stderr empty on exactly the turns whose failure matters most.
    """
    signature = error_signature(result.stderr) or error_signature(result.stdout)
    if result.returncode in _SYNTHETIC_RCS or not signature or signature in blockers:
        return
    blockers.append(signature)
    if len(blockers) > _MAX_BLOCKERS:
        del blockers[0]


def _record_note(notes: list[str], note: str) -> list[str]:
    """Return ``notes`` with ``note`` appended, deduplicated and bounded.

    Oldest notes are dropped first when either bound is exceeded — recent
    findings are the ones still relevant to where the run has got to.

    Args:
        notes: The notes accumulated so far.
        note: The new note to record.

    Returns:
        The updated note list.
    """
    note = note.strip()
    if not note or note in notes:
        return notes
    updated = [*notes, note]
    while len(updated) > _MAX_NOTES or sum(len(n) for n in updated) > _MAX_NOTE_CHARS:
        if len(updated) == 1:
            updated[0] = updated[0][:_MAX_NOTE_CHARS]
            break
        updated.pop(0)
    return updated


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
        knowledge: KnowledgeStore | None = None,
        prior_attempts: Sequence[AttemptRecord] = (),
    ) -> None:
        """Configure a repair loop.

        Args:
            spec: The normalised resurrection spec.
            sandbox: A started sandbox to execute inside.
            agent: The agent choosing actions each turn.
            max_turns: Hard ceiling on total turns (including resumed turns).
            contracts_root: Directory contracts are emitted under.
            today: Override for the resurrection date (defaults to today, UTC).
            knowledge: Cross-run lesson store to record this run's fixes into;
                ``None`` disables learning (the default for tests).
            prior_attempts: Earlier attempts on other base images, recorded in the
                provenance so a contract shows which images were ruled out first.
        """
        self.spec = spec
        self.sandbox = sandbox
        self.agent = agent
        self.max_turns = max_turns
        self.contracts_root = Path(contracts_root)
        self.today = today or _utc_today()
        self.knowledge = knowledge
        self.prior_attempts = list(prior_attempts)

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
        notes: list[str] = []
        inspected: dict[tuple[str, ...], int] = {}
        bugs_fixed: list[dict[str, str]] = []
        blockers: list[str] = []
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
                notes=list(notes),
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
            if action.notes:
                notes = _record_note(notes, action.notes)

            if action.kind == "give_up":
                verdict = "FAILED"
                reason = action.reason or "agent signalled the resurrection is unresolvable"
                break

            # An inspection command already run cannot tell us anything new, and
            # a model that repeats one is stuck. Spend the turn saying so rather
            # than re-running it.
            key = tuple(action.cmd)
            if is_read_only(action.cmd) and key in inspected:
                result, container_error = self._noop_repeat(inspected[key]), False
            else:
                if is_read_only(action.cmd):
                    inspected[key] = turn + 1
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
            if not result.ok:
                _record_blocker(blockers, result)

            # Record the successful commands, in order — this is the reproduction
            # recipe emitted as a smoke test. Inspection commands are dropped:
            # they are how the agent learned, not steps in reproducing the result.
            if result.ok and (not is_read_only(action.cmd) or action.is_sanity_check):
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

            # A repair does not have to be a source diff. Rewriting a dead apt
            # mirror or installing a missing compiler fixes real decay and is the
            # most reusable knowledge a run produces — requiring a patch here is
            # what left the lesson store empty after a successful resurrection.
            if action.patch or action.bug_class:
                if action.patch:
                    patch_history.append(action.patch)
                bugs_fixed.append(
                    {
                        "class": action.bug_class or "unknown",
                        "description": action.bug_description,
                        "patch": action.patch or "",
                        # The failure this fix was a response to — the "symptom"
                        # half of the lesson. Falls back to stdout because agents
                        # routinely pipe `2>&1`, which leaves stderr empty and
                        # would otherwise store a lesson with nothing to match on.
                        "symptom": error_signature(state.last_stderr)
                        or error_signature(state.last_stdout),
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
            notes=notes,
            blockers=blockers,
        )
        if verdict == "PASS":
            outcome.evidence = self._verify_sanity_claim()
        outcome.contract_dir = self._emit(outcome)
        self._learn(outcome)
        return outcome

    def _verify_sanity_claim(self) -> SanityEvidence:
        """Check the sanity check's structural claim against real produced files.

        The exit code proves a command succeeded; it does not prove the command
        proved what the plan *said* it would. A check reading "the BAM output
        passes samtools quickcheck" is only meaningful if a BAM was actually
        written. This probes the sandbox for files matching the claimed type's
        extensions and reports what it found.

        This never overturns the verdict. A real exit code outranks a filesystem
        heuristic — a tool may stream to stdout, write outside the working
        directory, or emit a type this module does not track — so an unsupported
        claim is recorded as a qualifier on the PASS, not a failure.

        Returns:
            The :class:`SanityEvidence` for this run.
        """
        extensions = extensions_for_check(self.spec.sanity_check)
        if not extensions:
            return SanityEvidence(status="untyped", claimed_extensions=[], found=[])

        # -newer is unavailable portably; list candidates by extension instead.
        patterns: list[str] = []
        for ext in extensions:
            patterns += ["-o", "-iname", f"*{ext}"]
        cmd = ["find", ".", "-type", "f", "(", *patterns[1:], ")", "-size", "+0c"]
        try:
            result = self.sandbox.exec(cmd, timeout=60)
        except (DockerError, TimeoutError):
            return SanityEvidence(status="unknown", claimed_extensions=list(extensions), found=[])
        if result.returncode != 0:
            return SanityEvidence(status="unknown", claimed_extensions=list(extensions), found=[])

        found = [line.strip() for line in result.stdout.splitlines() if line.strip()][:20]
        return SanityEvidence(
            status="supported" if found else "unsupported",
            claimed_extensions=list(extensions),
            found=found,
        )

    def _learn(self, outcome: RepairOutcome) -> None:
        """Record this run's fixes as lessons for future resurrections.

        Learning is best-effort: an unwritable store must never turn a successful
        resurrection into a crash.
        """
        if self.knowledge is None:
            return
        lessons = lessons_from_bugs(
            outcome.bugs_fixed,
            framework=self.spec.framework,
            tool=self.spec.tool_slug,
        )
        if not lessons:
            return
        with contextlib.suppress(OSError):
            self.knowledge.extend(lessons)

    @staticmethod
    def _noop_repeat(first_turn: int) -> ExecResult:
        """Return the synthetic result fed back for a suppressed repeat.

        Args:
            first_turn: The turn the identical command was first run on.

        Returns:
            A failed :class:`ExecResult` explaining the suppression.
        """
        return ExecResult(
            _RC_NOOP_REPEAT,
            "",
            f"[not executed] You already ran this exact inspection command on turn "
            f"{first_turn}. Nothing has changed since, so the output would be "
            "identical. You are repeating yourself instead of making progress. "
            "Write what you already know into notes, then do something DIFFERENT: "
            "install a dependency, build the package, write a script, or run the "
            "code. Do not inspect this again.",
            0.0,
        )

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
        (slug_dir / "smoke_test.sh").write_text(self._smoke_test(outcome), encoding="utf-8")
        (slug_dir / "REPRODUCE.md").write_text(self._reproduce_md(outcome), encoding="utf-8")
        return slug_dir

    def _schema(self) -> ContractSchema:
        return ContractSchema(
            inputs=[IOPort(name="input", type=GenomicFileType.ANY, description="primary input")],
            outputs=[IOPort(name="output", type=GenomicFileType.ANY, description="primary output")],
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
            f"CMD {cmd_json}\n"
        )

    def _predict_py(self, entry: list[str]) -> str:
        return _PREDICT_TEMPLATE.format(
            slug=self.spec.tool_slug,
            version=JEEVA_VERSION,
            cmd=json.dumps(entry),
        )

    def _smoke_test(self, outcome: RepairOutcome) -> str:
        """Emit the reproduction script for a passing run.

        Replays the state-changing commands that succeeded, in order, ending with
        the sanity check; each runs in its own subshell, exactly as the sandbox
        executed it. Inspection commands are already excluded — they are how the
        agent learned, not steps in reproducing the result.

        Commands that reference an absolute container path only make sense inside
        the resurrected image, so the script guards on that path rather than
        cloning and pretending it will work from a bare base image.
        """
        header = (
            "#!/usr/bin/env bash\n"
            f"# Reproduction of {self.spec.tool_slug} (emitted by Jeeva {JEEVA_VERSION}).\n"
            f"# Replays the verified command chain; success criterion:\n"
            f"#   {self.spec.sanity_check}\n"
            "set -euo pipefail\n"
        )
        if not outcome.reproduction:
            return header + "echo 'no reproduction was captured' >&2\nexit 1\n"

        joined = " ".join(shlex.join(cmd) for cmd in outcome.reproduction)
        anchor = self._container_anchor(joined)
        lines: list[str] = []
        if anchor:
            # The recipe is written against the image's own layout.
            header += (
                f"# Run this INSIDE the resurrected image ({outcome.final_image or 'see above'}):\n"
                f"#   docker run --rm {outcome.final_image or '<image>'} bash smoke_test.sh\n"
            )
            lines.append(
                f'[ -d {anchor} ] || {{ echo "{anchor} not found — run this inside the '
                f'resurrected image, not a bare base image" >&2; exit 1; }}'
            )
        else:
            header += 'cd "$(cd "$(dirname "$0")" && pwd)"\n'
            if self.spec.repo_url:
                lines.append(f"[ -d repo ] || git clone --depth 1 -- {self.spec.repo_url} repo")
        lines += [f"( {shlex.join(cmd)} )" for cmd in outcome.reproduction]
        return header + "\n".join(lines) + "\n"

    @staticmethod
    def _container_anchor(commands: str) -> str:
        """Return the absolute container directory the recipe depends on, if any.

        Args:
            commands: The reproduction commands joined into one string.

        Returns:
            The matched path (e.g. ``/workspace/repo``), or ``""`` if the recipe
            uses only relative paths and can run anywhere.
        """
        match = re.search(r"(/(?:workspace|work|opt|srv)/[A-Za-z0-9._-]+)", commands)
        return match.group(1) if match else ""

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
            f"{self._evidence_section(outcome)}"
            f"{self._reproduction_section(outcome)}"
        )

    def _evidence_section(self, outcome: RepairOutcome) -> str:
        """Return the Markdown block reporting how well the check's claim held up."""
        evidence = outcome.evidence
        if evidence.status == "untyped":
            return ""
        if evidence.status == "unknown":
            return (
                "\n## Evidence\n\nThe output-type probe could not run, so the sanity "
                "check's structural claim was not independently verified.\n"
            )
        claimed = ", ".join(evidence.claimed_extensions)
        if evidence.status == "supported":
            files = "\n".join(f"- `{path}`" for path in evidence.found)
            return (
                f"\n## Evidence\n\nThe check claims a `{claimed}` output, and matching "
                f"non-empty files were produced:\n\n{files}\n"
            )
        return (
            f"\n## Evidence\n\n**The check claims a `{claimed}` output, but no such file "
            "was found.** The sanity command exited 0, so the verdict stands — but the "
            "check may be proving less than its wording asserts. Treat this contract "
            "with more scepticism than one whose claim is backed by artefacts.\n"
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
            "sanity_check_evidence": outcome.evidence.to_dict(),
            # What the agent worked out about the repo — the run's audit trail.
            "agent_notes": outcome.notes,
            # Walls this attempt hit, and the images ruled out before it.
            "blockers": outcome.blockers,
            "escalation": [a.to_dict() for a in self.prior_attempts],
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
            override = self._goal_override()
            if self.args.no_scout:
                spec = self._spec_without_scout(url, commit, override)
            else:
                plan = asyncio.run(select_plan(repo_dir, url, confirm=True))
                spec = spec_from_plan(plan, url=url, repo_commit=commit)
                _apply_override(spec, override)
            if self.args.image:
                spec.base_image = self.args.image

            checkpoint_dir = Path(self.args.checkpoint_dir) if self.args.checkpoint_dir else None
            # Resume from the last good snapshot when this checkpoint has prior turns.
            resume_image: str | None = None
            if checkpoint_dir is not None:
                resume_image = CheckpointStore(checkpoint_dir).last_successful_snapshot()

            # One store, both directions: the agent reads prior lessons from it,
            # the loop writes this run's fixes back into it.
            knowledge = default_store()
            agent = self._build_agent(spec, knowledge)  # fail fast, before any container

            def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
                # Only the first attempt may resume a checkpoint: a snapshot taken
                # on a ruled-out image would drag that image's state into the retry.
                resume = resume_image if not prior else None
                spec.base_image = image
                sandbox = DockerSandbox(
                    resume or image,
                    workdir=self.args.workdir,
                    docker_host=self.args.docker_host,
                    gpus=self.args.gpus,
                    checkpoint_dir=checkpoint_dir if not prior else None,
                )
                sandbox.start()
                try:
                    if resume is None:
                        # Fresh container: seed it with the checkout. On resume the
                        # snapshot already carries the repo and in-container edits.
                        sandbox.copy_in(repo_dir, self.args.workdir)
                    return RepairLoop(
                        spec,
                        sandbox,
                        agent,
                        max_turns=self.args.turns,
                        knowledge=knowledge,
                        prior_attempts=prior,
                    ).run()
                finally:
                    if not self.args.keep:
                        sandbox.stop(force=True)

            runner = EscalatingResurrection(
                base_image=spec.base_image,
                run_attempt=attempt,
                max_extra_attempts=self.args.escalate,
                announce=lambda msg: print(msg, file=sys.stderr),
            )
            outcome = runner.run()

        self._report(outcome, runner.attempts)

    def _goal_override(self) -> GoalOverride:
        """Read and validate ``--goal-file``, or return an empty override.

        Exits cleanly (status 1) on an unreadable file, malformed YAML, or a
        sanity check that is not falsifiable — a bad criterion must be rejected
        before a container starts, not discovered after a meaningless PASS.
        """
        if not self.args.goal_file:
            return GoalOverride()
        try:
            override = parse_goal_file(Path(self.args.goal_file))
        except (OSError, ValueError) as exc:
            print(f"could not use --goal-file: {exc}", file=sys.stderr)
            sys.exit(1)
        return override

    def _spec_without_scout(
        self, url: str, commit: str, override: GoalOverride
    ) -> ResurrectionSpec:
        """Build a spec straight from the CLI, with no Scout involved.

        Requires ``--image`` and a ``--goal-file`` supplying both a goal and a
        sanity check: with the Scout skipped there is nothing else to derive
        them from, and a resurrection without a falsifiable criterion cannot
        mean anything.
        """
        if not self.args.image or not override.goal or not override.sanity_check:
            print(
                "--no-scout requires --image and a --goal-file providing both "
                "'goal:' and 'sanity_check:'.",
                file=sys.stderr,
            )
            sys.exit(1)
        return ResurrectionSpec(
            tool_slug=tool_slug(url),
            goal=override.goal,
            sanity_check=override.sanity_check,
            base_image=self.args.image,
            repo_url=url,
            repo_commit=commit,
        )

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

    def _build_agent(
        self, spec: ResurrectionSpec, knowledge: KnowledgeStore | None = None
    ) -> RepairAgent:
        """Build the autonomous LLM agent that plays :class:`RepairAgent`.

        Args:
            spec: The resurrection spec to plan against.
            knowledge: Lesson store the agent retrieves prior fixes from.

        Returns:
            An :class:`~sanjeevini.repair.agent.LLMRepairAgent`.

        Raises:
            RuntimeError: If the ``anthropic`` package (the ``[agent]`` extra) is
                not installed.
        """
        from sanjeevini.repair.agent import LLMRepairAgent

        return LLMRepairAgent(spec, knowledge=knowledge)

    def _report(self, outcome: RepairOutcome, attempts: Sequence[AttemptRecord] = ()) -> None:
        """Print the verdict, contract directory, and smoke-test command."""
        print(f"verdict      : {outcome.verdict}")
        print(f"turns        : {outcome.turns}")
        print(f"cost (usd)   : {outcome.cost_usd}")
        if outcome.reason:
            print(f"reason       : {outcome.reason}")
        # Only worth showing when the run actually escalated; a single attempt
        # is already fully described by the lines above.
        for i, attempt in enumerate(attempts if len(attempts) > 1 else (), start=1):
            detail = f" ({attempt.rule})" if attempt.rule else ""
            print(
                f"attempt {i}    : {attempt.base_image} -> {attempt.verdict} "
                f"in {attempt.turns} turns{detail}"
            )
        if outcome.evidence.contradicts_claim:
            exts = ", ".join(outcome.evidence.claimed_extensions)
            print(
                f"WARNING      : the sanity check claims a {exts} output, but no such "
                "file was found. The command exited 0, so this is still a PASS — but "
                "the check may be proving less than it says.",
                file=sys.stderr,
            )
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
