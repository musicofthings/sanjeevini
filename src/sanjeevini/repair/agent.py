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
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable
from typing import TYPE_CHECKING

from sanjeevini.repair.loop import RepairAction

if TYPE_CHECKING:
    from sanjeevini.repair.loop import LoopState, ResurrectionSpec

# complete(system, user) -> (reply_text, cost_usd)
Completion = Callable[[str, str], "tuple[str, float]"]

DEFAULT_MODEL = "claude-sonnet-5"

RESURRECTION_SYSTEM_PROMPT = """\
You are Jeeva: you resurrect dead research code and make a buried capability
callable, then PROVE it runs on a fresh input.

You work by issuing ONE shell command at a time inside a disposable Docker
sandbox. The target repository is already checked out in your working directory
(run `ls` first to orient). Container state persists across your commands, so an
install in one turn is still there the next. You cannot touch the host.

Each turn you receive the goal, the falsifiable sanity check, the last command's
exit code and stderr (the traceback to read), and the patches applied so far.
Reply with a SINGLE JSON object and nothing else:

  {"action": "exec",
   "cmd": ["bash", "-lc", "<one shell command>"],
   "is_sanity_check": false,
   "patch": "<unified diff, if this command applied a fix, else omit>",
   "bug_class": "<dead_endpoint|dep_conflict|missing_binary|api_drift|...>",
   "bug_description": "<one line, if a fix was applied>"}

or, only when an environmental blocker makes success impossible:

  {"action": "give_up", "reason": "<why no in-sandbox fix exists>"}

Method:
1. ORIENT — explore the repo; find the real code path from a fresh input to the
   headline output.
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


def render_state(spec: ResurrectionSpec, state: LoopState) -> str:
    """Render the loop state into the user prompt for one turn.

    Args:
        spec: The resurrection spec (goal, sanity check, repo, base image).
        state: The current loop state (last result, patch history, recent turns).

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
            lines.append("Last stderr (tail):\n" + _tail(stderr, 2000))
        stdout = state.last_stdout.strip()
        if stdout:
            lines.append("Last stdout (tail):\n" + _tail(stdout, 800))

    if state.patch_history:
        lines.append("\nPatches applied so far:")
        lines.extend(f"  - {_first_line(p)}" for p in state.patch_history)

    if state.history:
        lines.append("\nRecent commands this run:")
        for turn in state.history[-5:]:
            status = "ok" if turn.ok else f"exit {turn.returncode}"
            lines.append(f"  [{status}] {' '.join(turn.action.cmd)}")

    lines.append("\nReply with the single JSON action object for your next command.")
    return "\n".join(lines)


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
        return RepairAction(kind="give_up", reason="model reply was not valid JSON")

    action = data.get("action") or ("give_up" if data.get("reason") else "exec")
    if action == "give_up":
        return RepairAction(kind="give_up", reason=str(data.get("reason", "unspecified")))

    cmd = _coerce_cmd(data.get("cmd"))
    if not cmd:
        return RepairAction(kind="give_up", reason="model returned an exec action with no command")

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


class LLMRepairAgent:
    """A :class:`RepairAgent` that plans one action per turn with a Claude call."""

    def __init__(
        self,
        spec: ResurrectionSpec,
        *,
        complete: Completion | None = None,
        model: str | None = None,
    ) -> None:
        """Configure the agent.

        Args:
            spec: The resurrection spec to plan against.
            complete: Injected ``(system, user) -> (text, cost_usd)`` callable;
                defaults to an Anthropic-backed client.
            model: Model id override (defaults to ``$JEEVA_MODEL`` or
                :data:`DEFAULT_MODEL`).
        """
        self.spec = spec
        self.model = model or os.environ.get("JEEVA_MODEL", DEFAULT_MODEL)
        self._complete = complete or AnthropicClient(self.model)

    def next_action(self, state: LoopState) -> RepairAction:
        """Ask the model for the next action given ``state``."""
        user = render_state(self.spec, state)
        text, cost = self._complete(RESURRECTION_SYSTEM_PROMPT, user)
        action = parse_action(text)
        action.cost_usd = cost
        return action


class AnthropicClient:
    """Default completion backend using the Anthropic Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 1024,
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
        self._client = anthropic.Anthropic()

    def __call__(self, system: str, user: str) -> tuple[str, float]:
        """Send one turn to the model and return ``(reply_text, cost_usd)``."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
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
