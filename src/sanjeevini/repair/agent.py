"""The autonomous repair agent — a single-step planner for the repair loop.

:class:`~sanjeevini.repair.loop.RepairLoop` asks a :class:`RepairAgent` for one
action per turn. :class:`LLMRepairAgent` answers each request with a single
Claude call: it renders the current loop state (goal, sanity check, last
traceback, patch history) into a prompt and parses the model's reply into one
:class:`~sanjeevini.repair.loop.RepairAction` — a shell command to run in the
sandbox, or a decision to give up.

Deliberately, the model never declares success: PASS is decided by the loop when
a command the model *marks as the sanity check* actually exits 0 in the sandbox.
The model only chooses the next command. This keeps the scientific-correctness
guarantee in tested code and makes the agent's own logic — prompt construction
and reply parsing — unit-testable behind an injected ``complete`` callable.

The default backend uses the Anthropic Messages API (``anthropic`` package, the
``[agent]`` extra). Cost tracking is best-effort: pass per-million-token prices
to :class:`AnthropicClient` to accumulate real USD into the provenance record.

Set ``$JEEVA_BACKEND=subscription`` to drive the loop through :class:`SubscriptionClient`
instead — it shells out to the local Claude Code CLI via ``claude-agent-sdk`` and bills
against the caller's Claude subscription rather than a separate Anthropic API key.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable
from typing import TYPE_CHECKING

from sanjeevini.repair.knowledge import KnowledgeStore, Lesson
from sanjeevini.repair.loop import RepairAction, is_read_only

if TYPE_CHECKING:
    from sanjeevini.repair.loop import LoopState, ResurrectionSpec, TurnOutcome

# complete(system, user) -> (reply_text, cost_usd)
Completion = Callable[[str, str], "tuple[str, float]"]

DEFAULT_MODEL = "claude-sonnet-5"

# How many times to re-ask when a reply is malformed/empty before giving up.
_MAX_REPLY_ATTEMPTS = 3

# Forced structured-output tool: the model must call this with a valid action,
# so the reply is always schema-valid JSON rather than free text we hope parses.
_ACTION_TOOL = {
    "name": "next_action",
    "description": "Choose the single next action for the resurrection loop.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["exec", "give_up"]},
            "cmd": {
                "type": "array",
                "items": {"type": "string"},
                "description": "argv to run in the sandbox (required for exec).",
            },
            "is_sanity_check": {"type": "boolean"},
            "patch": {"type": "string"},
            "bug_class": {"type": "string"},
            "bug_description": {"type": "string"},
            "reason": {"type": "string"},
            "timeout": {"type": "integer"},
            "notes": {
                "type": "string",
                "description": (
                    "Durable findings to carry forward — the ONLY thing you will "
                    "still know next turn besides the last command's output. Record "
                    "what you learned (file layout, real API signatures, what a file "
                    "does NOT contain, what you already ruled out)."
                ),
            },
        },
        "required": ["action"],
    },
}

# Output budgets. These were 800/2000, which was far too small: a 7 KB file read
# left the agent with 11% of what it saw and nothing at all one turn later, so it
# re-read the same files indefinitely instead of making progress.
_MAX_STDOUT_CHARS = 6000
_MAX_STDERR_CHARS = 4000

RESURRECTION_SYSTEM_PROMPT = """\
You are Jeeva: you resurrect dead research code and make a buried capability
callable, then PROVE it runs on a fresh input.

You work by issuing ONE shell command at a time inside a disposable Docker
sandbox. The target repository is already checked out in your working directory
(run `ls` first to orient). Container state persists across your commands, so an
install in one turn is still there the next. You cannot touch the host.

YOU HAVE NO MEMORY BETWEEN TURNS except (a) the last command's output and (b) the
"notes" you write. Everything else you read is GONE next turn. So:
- Put every durable finding in "notes": the file layout, real function/class
  signatures, what you have already ruled out, and what a file does NOT contain.
  Notes are FACTS, not narration. Write "core.py is 261 lines; defines only
  MetaSegment and Segment; no Event class" — never "continuing to inspect
  core.py". A note that does not state a fact is wasted.
- Never re-read a file you have already read — check your notes first. If a file
  is shorter than you expected, record its true length and move on.
- Notes are cheap and re-reading is expensive. Write them on EVERY turn that
  taught you something.

Each turn you receive the goal, the falsifiable sanity check, the last command's
exit code and stderr (the traceback to read), your accumulated notes, and the
patches applied so far. Reply with a SINGLE JSON object and nothing else:

  {"action": "exec",
   "cmd": ["bash", "-lc", "<one shell command>"],
   "is_sanity_check": false,
   "patch": "<unified diff, ONLY if this command changed source code>",
   "bug_class": "<dead_mirror|missing_toolchain|dep_conflict|build_failure|
                  dead_endpoint|missing_binary|api_drift|...>",
   "bug_description": "<one line: what was broken and what fixed it>",
   "notes": "<durable findings to carry forward — see above>"}

Set "bug_class" and "bug_description" on EVERY turn that repairs something,
whether or not there is a diff. A repair is not only a source edit: repointing a
dead apt mirror, installing a missing compiler, pinning a dependency to its
commit era, or fixing a build flag are all repairs of real decay, and they are
the most reusable things you will discover. Omit "patch" when no source file
changed — but still name the bug.

or, only when an environmental blocker makes success impossible:

  {"action": "give_up", "reason": "<why no in-sandbox fix exists>"}

Method:
1. ORIENT — explore the repo; find the real code path from a fresh input to the
   headline output. Budget a handful of turns for this, not dozens: read a file
   ONCE, record what matters in notes, then move to REVIVE. If you have spent
   several turns reading without running anything, stop reading and run something.
2. REVIVE — run it. On failure, read the ACTUAL traceback and fix THAT cause with
   the smallest change that works. Re-run to confirm.
3. CARVE — reduce to the minimal command from one input to the output.
4. PROVE — run the sanity check on the known input. Set "is_sanity_check": true
   ONLY on the command whose exit code, being 0, actually demonstrates the
   measurable criterion. Do not claim success in prose — the loop decides PASS
   from the real exit code.

Repair heuristics:
- A missing binary is often present but off PATH — find it (`which`, `find /`) and
  symlink or export PATH rather than reinstalling.
- A rotted download URL: prefer feeding a local input file directly, bypassing the
  network.
- When pip pulls too-new deps, pin them to the repo's commit era and constrain
  transitive pins (numpy/protobuf).
- Snapshot-worthy successes are banked by the loop automatically; keep each command
  small and incremental.
- If a prebuilt binary aborts with SIGILL / an illegal-instruction / CPU-feature
  error under emulation and cannot be replaced or downgraded, the HOST lacks the
  native CPU support: verify concisely, then give_up naming the offending binary.
"""


def render_state(
    spec: ResurrectionSpec, state: LoopState, lessons: list[Lesson] | None = None
) -> str:
    """Render the loop state into the user prompt for one turn.

    Args:
        spec: The resurrection spec (goal, sanity check, repo, base image).
        state: The current loop state (last result, patch history, recent turns).
        lessons: Relevant fixes learned in earlier resurrections, injected as
            hints so each run starts smarter than the last.

    Returns:
        The prompt text handed to the model.
    """
    lines = [
        f"Repository: {spec.repo_url}",
        f"Base image: {spec.base_image}",
        f"Goal: {spec.goal}",
        f"Sanity check (must be provably met): {spec.sanity_check}",
        f"Turn {state.turn} of at most {state.max_turns}.",
    ]

    if state.last_returncode is None:
        lines.append("\nThis is the first turn. Start by orienting (`ls`, read the README).")
    else:
        lines.append(f"\nLast command exit code: {state.last_returncode}")
        stderr = state.last_stderr.strip()
        if stderr:
            lines.append("Last stderr (tail):\n" + _tail(stderr, _MAX_STDERR_CHARS))
        stdout = state.last_stdout.strip()
        if stdout:
            lines.append("Last stdout (tail):\n" + _tail(stdout, _MAX_STDOUT_CHARS))
        elif state.last_returncode == 0:
            # An empty result is information: the file ended, the pattern is absent.
            lines.append(
                "Last stdout was EMPTY. The command succeeded but produced nothing — "
                "the range is past end-of-file, or the pattern does not occur. Do not "
                "re-probe the same file; record this in notes and move on."
            )

    if state.notes:
        lines.append(
            "\nWhat you have already established (your notes — this is your only "
            "memory of earlier turns):"
        )
        lines.extend(f"  - {note}" for note in state.notes)

    if lessons:
        lines.append(
            "\nPrior experience from earlier resurrections (fixes that worked on "
            "similar errors — treat as hints, verify before trusting):"
        )
        lines.extend(f"  - {lesson.as_hint()}" for lesson in lessons)

    if state.patch_history:
        lines.append("\nPatches applied so far:")
        lines.extend(f"  - {_first_line(p)}" for p in state.patch_history)

    if state.history:
        lines.append("\nCommands already run this run (do not repeat these):")
        for turn in state.history[-12:]:
            status = "ok" if turn.ok else f"exit {turn.returncode}"
            size = len(turn.stdout)
            lines.append(f"  [{status}, {size}b out] {' '.join(turn.action.cmd)}")

    stalled = _reading_without_progress(state.history)
    if stalled:
        lines.append(
            f"\nSTOP: your last {stalled} commands only inspected files without "
            "changing or running anything. You are not making progress. Write what "
            "you know into notes and take a concrete action now — install a "
            "dependency, build the package, or run the code."
        )

    lines.append("\nReply with the single JSON action object for your next command.")
    return "\n".join(lines)


# Consecutive read-only turns tolerated before the prompt calls it out.
_STALL_LIMIT = 4


def _reading_without_progress(history: list[TurnOutcome]) -> int:
    """Return the length of the trailing run of purely inspective commands.

    Orienting is necessary; orienting forever is the failure mode this catches —
    an agent that re-reads files instead of acting will otherwise burn its whole
    turn budget looking busy.

    Args:
        history: The turns taken so far this run.

    Returns:
        The trailing read-only streak, or ``0`` if below :data:`_STALL_LIMIT`.
    """
    streak = 0
    for turn in reversed(history):
        if turn.action.patch or not is_read_only(turn.action.cmd):
            break
        streak += 1
    return streak if streak >= _STALL_LIMIT else 0


def _tail(text: str, limit: int) -> str:
    """Return the last ``limit`` characters of ``text``."""
    return text if len(text) <= limit else "…" + text[-limit:]


def _first_line(text: str) -> str:
    """Return the first non-empty line of ``text`` (for compact patch summaries)."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:100]
    return "(empty patch)"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Give-up reasons that signal a malformed/empty model reply (retry), as opposed
# to the model deliberately choosing to give up (honour immediately).
_REASON_NO_JSON = "model reply was not valid JSON"
_REASON_NO_CMD = "model returned an exec action with no command"
_RETRYABLE_REASONS = frozenset({_REASON_NO_JSON, _REASON_NO_CMD})


def parse_action(text: str) -> RepairAction:
    """Parse a model reply into a :class:`RepairAction`.

    Accepts a bare JSON object, one wrapped in prose, or one inside a ```` ```json ````
    fence. A reply with no usable JSON, or an ``exec`` with no command, becomes a
    ``give_up`` so the loop terminates cleanly rather than looping on noise.

    Args:
        text: The model's raw reply.

    Returns:
        The parsed action.
    """
    data = _extract_json(text)
    if data is None:
        return RepairAction(kind="give_up", reason=_REASON_NO_JSON)

    action = data.get("action") or ("give_up" if data.get("reason") else "exec")
    if action == "give_up":
        return RepairAction(kind="give_up", reason=str(data.get("reason", "unspecified")))

    cmd = _coerce_cmd(data.get("cmd"))
    if not cmd:
        return RepairAction(kind="give_up", reason=_REASON_NO_CMD)

    patch = data.get("patch")
    timeout = data.get("timeout", 300)
    return RepairAction(
        kind="exec",
        cmd=cmd,
        is_sanity_check=bool(data.get("is_sanity_check")),
        patch=str(patch) if patch else None,
        bug_class=str(data["bug_class"]) if data.get("bug_class") else None,
        bug_description=str(data.get("bug_description", "")),
        timeout=int(timeout) if isinstance(timeout, (int, str)) else 300,
        notes=str(data.get("notes", "")).strip(),
    )


def _extract_json(text: str) -> dict[str, object] | None:
    """Return the first JSON object embedded in ``text``, or ``None``."""
    for candidate in (text, _JSON_OBJECT_RE.search(text)):
        raw = candidate if isinstance(candidate, str) else (candidate.group(0) if candidate else "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_cmd(value: object) -> list[str]:
    """Coerce a command value (list or shell string) into an argv list."""
    if isinstance(value, list):
        return [str(part) for part in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return []


def _default_completion(model: str) -> Completion:
    """Return the default completion backend for ``model``.

    ``$JEEVA_BACKEND=subscription`` opts into :class:`SubscriptionClient`; any
    other value (including unset) keeps the direct-API :class:`AnthropicClient`.
    """
    if os.environ.get("JEEVA_BACKEND") == "subscription":
        return SubscriptionClient(model)
    return AnthropicClient(model)


class LLMRepairAgent:
    """A :class:`RepairAgent` that plans one action per turn with a Claude call."""

    def __init__(
        self,
        spec: ResurrectionSpec,
        *,
        complete: Completion | None = None,
        model: str | None = None,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        """Configure the agent.

        Args:
            spec: The resurrection spec to plan against.
            complete: Injected ``(system, user) -> (text, cost_usd)`` callable;
                defaults to an Anthropic-backed client, or a subscription-backed
                one when ``$JEEVA_BACKEND=subscription`` (see module docstring).
            model: Model id override (defaults to ``$JEEVA_MODEL`` or
                :data:`DEFAULT_MODEL`).
            knowledge: Store of lessons from earlier runs; relevant ones are
                injected into each turn's prompt. ``None`` disables retrieval.
        """
        self.spec = spec
        self.model = model or os.environ.get("JEEVA_MODEL", DEFAULT_MODEL)
        self._complete = complete or _default_completion(self.model)
        self._knowledge = knowledge

    def next_action(self, state: LoopState) -> RepairAction:
        """Ask the model for the next action given ``state``.

        Retries a malformed or empty reply a few times (a transient hiccup must
        not end the whole resurrection); a deliberate ``give_up`` from the model
        is honoured immediately.
        """
        user = render_state(self.spec, state, self._lessons_for(state))
        action = RepairAction(kind="give_up", reason=_REASON_NO_JSON)
        for attempt in range(_MAX_REPLY_ATTEMPTS):
            nudge = "" if attempt == 0 else "\n\nReturn exactly one next_action now."
            text, cost = self._complete(RESURRECTION_SYSTEM_PROMPT, user + nudge)
            action = parse_action(text)
            action.cost_usd = cost
            if not (action.kind == "give_up" and action.reason in _RETRYABLE_REASONS):
                return action
        return action

    def _lessons_for(self, state: LoopState) -> list[Lesson]:
        """Return prior lessons relevant to the traceback the agent is facing."""
        if self._knowledge is None:
            return []
        return self._knowledge.relevant(
            framework=self.spec.framework,
            error_text=state.last_stderr,
            top_k=4,
        )


class AnthropicClient:
    """Default completion backend using the Anthropic Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 2048,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
    ) -> None:
        """Create the client (raises if the ``anthropic`` package is absent).

        Args:
            model: Model id to call.
            max_tokens: Max output tokens per turn.
            price_in_per_mtok: Optional input price ($/1M tokens) for cost tracking.
            price_out_per_mtok: Optional output price ($/1M tokens) for cost tracking.

        Raises:
            RuntimeError: If ``anthropic`` is not installed.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "autonomous resurrection needs the 'anthropic' package: "
                "pip install 'sanjeevini-bio[agent]' (and set ANTHROPIC_API_KEY)."
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self._price_in = price_in_per_mtok
        self._price_out = price_out_per_mtok
        # Let the SDK ride out transient rate-limit / overloaded (429/529) errors
        # with backoff before an exception ever reaches the loop.
        self._client = anthropic.Anthropic(max_retries=6)

    def __call__(self, system: str, user: str) -> tuple[str, float]:
        """Send one turn to the model and return ``(action_json, cost_usd)``.

        Forces the model to call the ``next_action`` tool, so the returned text is
        always the schema-valid action serialised as JSON (never free-form prose
        we have to hope parses).
        """
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_ACTION_TOOL],
            tool_choice={"type": "tool", "name": "next_action"},
        )
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "next_action":
                return json.dumps(block.input), self._cost(response)
        # Fallback: no tool call (unexpected under forced tool_choice) — return any text.
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return text, self._cost(response)

    def _cost(self, response: object) -> float:
        """Best-effort USD cost from token usage, when prices were provided."""
        if self._price_in is None or self._price_out is None:
            return 0.0
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        in_tok = getattr(usage, "input_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0)
        return (in_tok * self._price_in + out_tok * self._price_out) / 1_000_000


class SubscriptionClient:
    """Completion backend that drives the local Claude Code CLI via ``claude-agent-sdk``.

    Unlike :class:`AnthropicClient`, this bills against the caller's Claude
    subscription (Pro/Max/Team) rather than a separate Anthropic API key — it
    spawns the ``claude`` binary already authenticated in this environment
    instead of calling ``anthropic.Anthropic()`` directly.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Create the client (raises if ``claude-agent-sdk`` is absent).

        Args:
            model: Model id to request from the CLI.

        Raises:
            RuntimeError: If ``claude_agent_sdk`` is not installed.
        """
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "subscription-backed resurrection needs claude-agent-sdk: "
                "pip install 'sanjeevini-bio[agent]'."
            ) from exc
        self.model = model

    def __call__(self, system: str, user: str) -> tuple[str, float]:
        """Send one turn through the CLI and return ``(reply_text, cost_usd)``."""
        import anyio

        return anyio.run(self._aquery, system, user)

    async def _aquery(self, system: str, user: str) -> tuple[str, float]:
        """Run a single tool-free query and collect its text and reported cost."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self.model,
            tools=[],
            permission_mode="bypassPermissions",
            max_turns=1,
            # An ANTHROPIC_API_KEY inherited from the parent shell would otherwise
            # take billing precedence over the CLI's own subscription login.
            env={"ANTHROPIC_API_KEY": ""},
        )
        text_parts: list[str] = []
        cost = 0.0
        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                text_parts.extend(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
        return "".join(text_parts), cost
