"""Tests for sanjeevini.repair.agent (the single-step planner)."""

from __future__ import annotations

import pytest

from sanjeevini.repair.agent import (
    DEFAULT_MODEL,
    LLMRepairAgent,
    parse_action,
    render_state,
)
from sanjeevini.repair.loop import LoopState, RepairAction, ResurrectionSpec, TurnOutcome


def _spec() -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="foo",
        goal="revive foo",
        sanity_check="output VCF has ≥ 1 variant record",
        base_image="python:3.10-slim",
        repo_url="https://github.com/acme/foo",
    )


def _state(**kw: object) -> LoopState:
    base: dict[str, object] = {
        "turn": 1,
        "max_turns": 60,
        "goal": "revive foo",
        "sanity_check": "≥ 1 variant",
        "base_image": "python:3.10-slim",
        "last_returncode": None,
        "last_stdout": "",
        "last_stderr": "",
        "patch_history": [],
        "history": [],
    }
    base.update(kw)
    return LoopState(**base)  # type: ignore[arg-type]


# ---- parse_action ---------------------------------------------------------


def test_parse_bare_exec_action() -> None:
    action = parse_action('{"action": "exec", "cmd": ["ls", "-la"]}')
    assert action.kind == "exec"
    assert action.cmd == ["ls", "-la"]
    assert action.is_sanity_check is False


def test_parse_action_in_prose_and_fence() -> None:
    reply = 'Sure — here is the next step:\n```json\n{"action":"exec","cmd":"python run.py"}\n```\n'
    action = parse_action(reply)
    assert action.kind == "exec"
    assert action.cmd == ["python", "run.py"]  # shell string coerced to argv


def test_parse_sanity_check_and_patch() -> None:
    action = parse_action(
        '{"action":"exec","cmd":["pytest"],"is_sanity_check":true,'
        '"patch":"--- a\\n+++ b","bug_class":"dep_conflict","bug_description":"pin numpy"}'
    )
    assert action.is_sanity_check is True
    assert action.patch == "--- a\n+++ b"
    assert action.bug_class == "dep_conflict"


def test_parse_give_up() -> None:
    action = parse_action('{"action":"give_up","reason":"host lacks AVX"}')
    assert action.kind == "give_up"
    assert "AVX" in action.reason


def test_parse_reason_only_is_give_up() -> None:
    assert parse_action('{"reason": "cannot proceed"}').kind == "give_up"


def test_parse_non_json_is_give_up() -> None:
    assert parse_action("I think we should try harder!").kind == "give_up"


def test_parse_exec_without_command_is_give_up() -> None:
    assert parse_action('{"action": "exec"}').kind == "give_up"


# ---- render_state ---------------------------------------------------------


def test_render_state_first_turn_orients() -> None:
    text = render_state(_spec(), _state())
    assert "first turn" in text
    assert "github.com/acme/foo" in text
    assert "output VCF has ≥ 1 variant record" in text


def test_render_state_includes_traceback_and_history() -> None:
    prev = TurnOutcome(
        action=RepairAction(kind="exec", cmd=["pip", "install", "."]),
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'numpy'",
        duration_s=0.1,
    )
    text = render_state(
        _spec(),
        _state(
            turn=2,
            last_returncode=1,
            last_stderr="ModuleNotFoundError: No module named 'numpy'",
            patch_history=["--- a/setup.py\n+++ b/setup.py"],
            history=[prev],
        ),
    )
    assert "exit code: 1" in text
    assert "ModuleNotFoundError" in text
    assert "Patches applied" in text
    assert "pip install ." in text


# ---- LLMRepairAgent -------------------------------------------------------


def test_agent_next_action_uses_injected_completion() -> None:
    seen: dict[str, str] = {}

    def fake_complete(system: str, user: str) -> tuple[str, float]:
        seen["system"] = system
        seen["user"] = user
        return '{"action":"exec","cmd":["python","predict.py"],"is_sanity_check":true}', 0.02

    agent = LLMRepairAgent(_spec(), complete=fake_complete, model="test-model")
    action = agent.next_action(_state())
    assert action.kind == "exec"
    assert action.is_sanity_check is True
    assert action.cost_usd == 0.02
    assert "Jeeva" in seen["system"]
    assert "github.com/acme/foo" in seen["user"]


def test_agent_retries_transient_bad_reply_then_succeeds() -> None:
    replies = iter(
        [
            ("not json at all", 0.0),  # transient garbage
            ('{"action":"exec","cmd":["ls"]}', 0.01),  # good on retry
        ]
    )

    def flaky_complete(system: str, user: str) -> tuple[str, float]:
        return next(replies)

    action = LLMRepairAgent(_spec(), complete=flaky_complete).next_action(_state())
    assert action.kind == "exec"
    assert action.cmd == ["ls"]


def test_agent_honours_real_give_up_without_retrying() -> None:
    calls = {"n": 0}

    def complete(system: str, user: str) -> tuple[str, float]:
        calls["n"] += 1
        return '{"action":"give_up","reason":"host lacks AVX"}', 0.0

    action = LLMRepairAgent(_spec(), complete=complete).next_action(_state())
    assert action.kind == "give_up"
    assert calls["n"] == 1  # a deliberate give_up is not retried


def test_agent_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEEVA_MODEL", raising=False)
    agent = LLMRepairAgent(_spec(), complete=lambda _s, _u: ("{}", 0.0))
    assert agent.model == DEFAULT_MODEL


def test_agent_honours_jeeva_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEEVA_MODEL", "claude-opus-4-8")
    agent = LLMRepairAgent(_spec(), complete=lambda _s, _u: ("{}", 0.0))
    assert agent.model == "claude-opus-4-8"
