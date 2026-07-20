"""Tests for within-run working memory — the notes scratchpad, output budgets,
and stall detection.

Regression origin: on the PyPore run the agent burned 38 turns re-reading the
same two files. It saw only the last 800 chars of stdout and no earlier output at
all, so a 7 KB file read left it with 11% of what it had seen and nothing one turn
later. These tests pin the fix.
"""

from __future__ import annotations

from pathlib import Path

from sanjeevini.repair.agent import (
    _MAX_STDERR_CHARS,
    _MAX_STDOUT_CHARS,
    LLMRepairAgent,
    parse_action,
    render_state,
)
from sanjeevini.repair.loop import (
    _MAX_NOTE_CHARS,
    _MAX_NOTES,
    LoopState,
    RepairAction,
    RepairLoop,
    ResurrectionSpec,
    ScriptedAgent,
    TurnOutcome,
    _record_note,
    is_read_only,
)
from sanjeevini.sandbox.docker_sandbox import ExecResult


def _spec() -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="pypore",
        goal="revive pypore",
        sanity_check="the JSON output parses and contains ≥ 10 events",
        base_image="python:2.7-slim",
    )


def _state(**kw: object) -> LoopState:
    base: dict[str, object] = {
        "turn": 3,
        "max_turns": 40,
        "goal": "revive pypore",
        "sanity_check": "≥ 10 events",
        "base_image": "python:2.7-slim",
        "last_returncode": 0,
        "last_stdout": "",
        "last_stderr": "",
        "patch_history": [],
        "history": [],
        "notes": [],
    }
    base.update(kw)
    return LoopState(**base)  # type: ignore[arg-type]


def _turn(cmd: str, rc: int = 0, stdout: str = "") -> TurnOutcome:
    return TurnOutcome(
        action=RepairAction(kind="exec", cmd=["bash", "-lc", cmd]),
        returncode=rc,
        stdout=stdout,
        stderr="",
        duration_s=0.1,
    )


class _Sandbox:
    def __init__(self) -> None:
        self.previous_turns: list[object] = []

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        return ExecResult(0, "", "", 0.1)

    def snapshot(self, tag: str) -> str:
        return tag

    def last_successful_snapshot(self) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Notes survive across turns
# ---------------------------------------------------------------------------


def test_notes_are_parsed_from_a_reply() -> None:
    action = parse_action('{"action":"exec","cmd":["ls"],"notes":"core.py is 261 lines"}')
    assert action.notes == "core.py is 261 lines"


def test_notes_default_to_empty() -> None:
    assert parse_action('{"action":"exec","cmd":["ls"]}').notes == ""


def test_notes_are_rendered_as_the_agents_memory() -> None:
    prompt = render_state(_spec(), _state(notes=["core.py is 261 lines, no class Event"]))
    assert "already established" in prompt
    assert "core.py is 261 lines, no class Event" in prompt


def test_no_notes_section_when_there_are_none() -> None:
    assert "already established" not in render_state(_spec(), _state())


def test_the_loop_carries_notes_between_turns(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class Recorder:
        def __init__(self) -> None:
            self._turn = 0

        def next_action(self, state: LoopState) -> RepairAction:
            seen.append(list(state.notes))
            self._turn += 1
            if self._turn == 1:
                return RepairAction(kind="exec", cmd=["ls"], notes="core.py is 261 lines")
            return RepairAction(kind="exec", cmd=["run"], is_sanity_check=True)

    RepairLoop(_spec(), _Sandbox(), Recorder(), contracts_root=tmp_path).run()

    assert seen[0] == []
    assert seen[1] == ["core.py is 261 lines"]


def test_duplicate_notes_are_not_recorded_twice() -> None:
    notes = _record_note(["a finding"], "a finding")
    assert notes == ["a finding"]


def test_blank_notes_are_ignored() -> None:
    assert _record_note(["a"], "   ") == ["a"]


def test_notes_are_bounded_by_count() -> None:
    notes: list[str] = []
    for i in range(_MAX_NOTES + 10):
        notes = _record_note(notes, f"note {i}")
    assert len(notes) == _MAX_NOTES
    # Oldest dropped first — recent findings are the relevant ones.
    assert notes[-1] == f"note {_MAX_NOTES + 9}"


def test_notes_are_bounded_by_total_size() -> None:
    notes: list[str] = []
    for i in range(20):
        notes = _record_note(notes, f"{i}:" + "x" * 1000)
    assert sum(len(n) for n in notes) <= _MAX_NOTE_CHARS


def test_a_single_oversized_note_is_truncated_not_dropped() -> None:
    notes = _record_note([], "y" * (_MAX_NOTE_CHARS * 2))
    assert len(notes) == 1
    assert len(notes[0]) == _MAX_NOTE_CHARS


# ---------------------------------------------------------------------------
# Output budgets
# ---------------------------------------------------------------------------


def test_stdout_budget_is_large_enough_for_a_source_file() -> None:
    # The PyPore failure: a 7 KB read must not be cut to a few hundred chars.
    assert _MAX_STDOUT_CHARS >= 6000
    body = "\n".join(f"line {i}" for i in range(900))
    prompt = render_state(_spec(), _state(last_stdout=body))
    assert len(prompt) > 5000


def test_stderr_budget_is_generous() -> None:
    assert _MAX_STDERR_CHARS >= 4000


def test_an_empty_successful_result_is_called_out() -> None:
    # Empty output is information — it means end-of-file or no match, and the
    # agent must not re-probe the same range.
    prompt = render_state(_spec(), _state(last_returncode=0, last_stdout=""))
    assert "EMPTY" in prompt
    assert "re-probe" in prompt


def test_an_empty_failed_result_is_not_called_out() -> None:
    prompt = render_state(_spec(), _state(last_returncode=1, last_stdout=""))
    assert "EMPTY" not in prompt


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------


def test_a_long_read_only_streak_is_flagged() -> None:
    history = [_turn(f"sed -n '{i},{i + 50}p' core.py") for i in range(8)]
    prompt = render_state(_spec(), _state(history=history))
    assert "STOP" in prompt
    assert "not making progress" in prompt


def test_a_short_read_only_streak_is_not_flagged() -> None:
    history = [_turn("ls"), _turn("cat README.md")]
    assert "STOP" not in render_state(_spec(), _state(history=history))


def test_a_real_action_resets_the_streak() -> None:
    history = [_turn(f"grep x f{i}.py") for i in range(8)]
    history.append(_turn("pip install numpy==1.16.6"))
    assert "STOP" not in render_state(_spec(), _state(history=history))


def test_a_patch_counts_as_progress() -> None:
    history = [_turn(f"cat f{i}.py") for i in range(8)]
    history[-1].action.patch = "--- a\n+++ b\n+fix"
    assert "STOP" not in render_state(_spec(), _state(history=history))


def test_history_shows_output_sizes_so_repeats_are_visible() -> None:
    prompt = render_state(_spec(), _state(history=[_turn("cat core.py", stdout="x" * 7016)]))
    assert "7016b out" in prompt
    assert "do not repeat" in prompt


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_the_agent_passes_notes_into_the_prompt() -> None:
    prompts: list[str] = []

    def fake(system: str, user: str) -> tuple[str, float]:
        prompts.append(user)
        return '{"action":"exec","cmd":["ls"]}', 0.0

    LLMRepairAgent(_spec(), complete=fake).next_action(_state(notes=["numpy 1.16.6 installed"]))
    assert "numpy 1.16.6 installed" in prompts[0]


def test_the_system_prompt_warns_about_amnesia() -> None:
    from sanjeevini.repair.agent import RESURRECTION_SYSTEM_PROMPT

    assert "NO MEMORY BETWEEN TURNS" in RESURRECTION_SYSTEM_PROMPT
    assert "Never re-read a file you have already read" in RESURRECTION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# No-op repeat suppression
# ---------------------------------------------------------------------------


def test_a_repeated_inspection_is_not_re_executed(tmp_path: Path) -> None:
    # The second PyPore failure: the agent emitted the identical sed command
    # five times. Re-running an inspection cannot reveal anything new.
    class Counter:
        def __init__(self) -> None:
            self.execs: list[list[str]] = []
            self.previous_turns: list[object] = []

        def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
            self.execs.append(list(cmd))
            return ExecResult(0, "file body", "", 0.1)

        def snapshot(self, tag: str) -> str:
            return tag

        def last_successful_snapshot(self) -> str | None:
            return None

    sandbox = Counter()
    repeat = ["bash", "-lc", "sed -n '150,320p' parsers.py"]
    agent = ScriptedAgent(
        [
            RepairAction(kind="exec", cmd=repeat),
            RepairAction(kind="exec", cmd=repeat),
            RepairAction(kind="exec", cmd=repeat),
            RepairAction(kind="exec", cmd=["bash", "-lc", "run"], is_sanity_check=True),
        ]
    )
    outcome = RepairLoop(_spec(), sandbox, agent, contracts_root=tmp_path).run()

    assert outcome.verdict == "PASS"
    # Executed once, not three times; the sanity command still ran.
    assert sandbox.execs.count(repeat) == 1


def test_a_suppressed_repeat_tells_the_agent_why() -> None:
    result = RepairLoop._noop_repeat(7)
    assert result.returncode != 0
    assert "turn 7" in result.stderr
    assert "something DIFFERENT" in result.stderr


def test_a_repeated_build_command_is_still_re_executed(tmp_path: Path) -> None:
    # Re-running a build or test after a fix is the whole point — only pure
    # inspection is safe to suppress.
    class Counter:
        def __init__(self) -> None:
            self.execs: list[list[str]] = []
            self.previous_turns: list[object] = []

        def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
            self.execs.append(list(cmd))
            return ExecResult(0, "", "", 0.1)

        def snapshot(self, tag: str) -> str:
            return tag

        def last_successful_snapshot(self) -> str | None:
            return None

    sandbox = Counter()
    build = ["bash", "-lc", "python setup.py build_ext --inplace"]
    agent = ScriptedAgent(
        [
            RepairAction(kind="exec", cmd=build),
            RepairAction(kind="exec", cmd=build, is_sanity_check=True),
        ]
    )
    RepairLoop(_spec(), sandbox, agent, contracts_root=tmp_path).run()
    assert sandbox.execs.count(build) == 2


def test_is_read_only_classifies_commands() -> None:
    assert is_read_only(["bash", "-lc", "sed -n '1,10p' f.py"])
    assert is_read_only(["bash", "-lc", "grep -n foo f.py"])
    assert not is_read_only(["bash", "-lc", "pip install numpy"])
    assert not is_read_only(["bash", "-lc", "python setup.py build"])
    assert not is_read_only([])


def test_notes_are_recorded_in_provenance(tmp_path: Path) -> None:
    import json

    agent = ScriptedAgent(
        [RepairAction(kind="exec", cmd=["run"], is_sanity_check=True, notes="core.py is 261 lines")]
    )
    RepairLoop(_spec(), _Sandbox(), agent, contracts_root=tmp_path).run()
    record = json.loads((tmp_path / "pypore" / "PROVENANCE.json").read_text())
    assert record["agent_notes"] == ["core.py is 261 lines"]
