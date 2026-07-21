"""Tests for bounded self-escalation onto an alternate base image."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanjeevini.repair.escalation import (
    AttemptRecord,
    EscalatingResurrection,
    propose_escalation,
)
from sanjeevini.repair.knowledge import error_signature
from sanjeevini.repair.loop import (
    RepairAction,
    RepairLoop,
    RepairOutcome,
    ResurrectionSpec,
    ScriptedAgent,
)
from sanjeevini.sandbox.checkpoint import TurnRecord
from sanjeevini.sandbox.docker_sandbox import ExecResult


def _outcome(
    verdict: str = "TIMEOUT", *, blockers: list[str] | None = None, **kw: object
) -> RepairOutcome:
    return RepairOutcome(
        verdict=verdict,  # type: ignore[arg-type]
        turns=kw.pop("turns", 12),  # type: ignore[arg-type]
        cost_usd=0.0,
        reason=str(kw.pop("reason", "")),
        blockers=blockers or [],
    )


# ---------------------------------------------------------------------------
# propose_escalation — when a retry is justified at all
# ---------------------------------------------------------------------------


def test_pass_never_escalates() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="PASS",
        reason="",
        blockers=["SyntaxError: Missing parentheses in call to 'print'"],
    )
    assert step is None


def test_no_blockers_no_escalation() -> None:
    """Without evidence there is no justified alternative. Failing is honest."""
    step = propose_escalation(
        base_image="python:3.11-slim", verdict="TIMEOUT", reason="turn limit", blockers=[]
    )
    assert step is None


def test_unmatched_blockers_do_not_escalate() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="FAILED",
        reason="",
        blockers=["AssertionError: expected 12 events, got 3", "ValueError: bad threshold"],
    )
    assert step is None


def test_agent_api_failure_does_not_escalate() -> None:
    """A dead API taught us nothing about the image; retrying only burns credit."""
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="FAILED",
        reason="agent call failed: BadRequestError: credit balance is too low",
        blockers=["gcc: command not found"],
    )
    assert step is None


# ---------------------------------------------------------------------------
# The Python 2 rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blocker",
    [
        "SyntaxError: Missing parentheses in call to 'print'",
        "ImportError: No module named urllib2",
        "ModuleNotFoundError: No module named 'cPickle'",
        "  except IOError, e:",
    ],
)
def test_python2_evidence_retargets_at_27(blocker: str) -> None:
    step = propose_escalation(
        base_image="python:3.11-slim", verdict="TIMEOUT", reason="", blockers=[blocker]
    )
    assert step is not None
    assert step.base_image == "python:2.7-slim"
    assert step.rule == "python2_sources"
    assert step.signal == blocker


def test_python2_retarget_preserves_fullness() -> None:
    step = propose_escalation(
        base_image="python:3.11",
        verdict="TIMEOUT",
        reason="",
        blockers=["No module named cStringIO"],
    )
    assert step is not None
    assert step.base_image == "python:2.7"


def test_python2_retarget_drops_codename_suffix() -> None:
    """2.7 predates codename tags — `python:2.7-slim-bookworm` does not exist."""
    step = propose_escalation(
        base_image="python:3.11-slim-bookworm",
        verdict="TIMEOUT",
        reason="",
        blockers=["Missing parentheses in call to 'print'"],
    )
    assert step is not None
    assert step.base_image == "python:2.7-slim"


def test_python2_rule_declines_on_a_non_python_image() -> None:
    """Swapping the tag on a conda/vendor image would discard its environment."""
    step = propose_escalation(
        base_image="continuumio/miniconda3:latest",
        verdict="TIMEOUT",
        reason="",
        blockers=["Missing parentheses in call to 'print'"],
    )
    assert step is None


def test_python2_rule_declines_when_already_on_python2() -> None:
    step = propose_escalation(
        base_image="python:2.7-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=["Missing parentheses in call to 'print'"],
    )
    assert step is None


# ---------------------------------------------------------------------------
# The toolchain rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blocker",
    [
        "gcc: command not found",
        "/bin/sh: 1: cc: not found",
        "error: command 'x86_64-linux-gnu-gcc' failed with exit status 1",
        "unable to execute 'gcc': No such file or directory",
        "fatal error: Python.h: No such file or directory",
    ],
)
def test_toolchain_evidence_widens_the_image(blocker: str) -> None:
    step = propose_escalation(
        base_image="python:3.11-slim", verdict="TIMEOUT", reason="", blockers=[blocker]
    )
    assert step is not None
    assert step.base_image == "python:3.11"
    assert step.rule == "missing_toolchain"


def test_toolchain_rule_keeps_the_codename() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim-bookworm",
        verdict="TIMEOUT",
        reason="",
        blockers=["gcc: command not found"],
    )
    assert step is not None
    assert step.base_image == "python:3.11-bookworm"


def test_toolchain_rule_declines_on_an_already_full_image() -> None:
    """A full image already ships gcc — the fix is apt, not another image."""
    step = propose_escalation(
        base_image="python:3.11", verdict="TIMEOUT", reason="", blockers=["gcc: command not found"]
    )
    assert step is None


def test_gcc_inside_a_word_is_not_a_toolchain_signal() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=["ERROR: file gcc-notes.txt not found"],
    )
    assert step is None


# ---------------------------------------------------------------------------
# The musl rule
# ---------------------------------------------------------------------------


def test_musl_evidence_leaves_alpine() -> None:
    step = propose_escalation(
        base_image="python:3.11-alpine",
        verdict="TIMEOUT",
        reason="",
        blockers=["ERROR: Failed building wheel for numpy"],
    )
    assert step is not None
    assert step.base_image == "python:3.11"
    assert step.rule == "musl_incompatible"


def test_musl_rule_only_fires_on_alpine() -> None:
    """The same wheel failure on Debian means something else; don't misdiagnose."""
    step = propose_escalation(
        base_image="python:3.11",
        verdict="TIMEOUT",
        reason="",
        blockers=["ERROR: Failed building wheel for numpy"],
    )
    assert step is None


# ---------------------------------------------------------------------------
# Priority and repetition
# ---------------------------------------------------------------------------


def test_python2_outranks_toolchain() -> None:
    """A setup.py that won't parse also reports build errors — fix the cause."""
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=[
            "error: command 'gcc' failed with exit status 1",
            "SyntaxError: Missing parentheses in call to 'print'",
        ],
    )
    assert step is not None
    assert step.rule == "python2_sources"


def test_most_recent_matching_blocker_is_the_signal() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=["gcc: command not found", "unable to execute 'gcc': No such file or directory"],
    )
    assert step is not None
    assert step.signal == "unable to execute 'gcc': No such file or directory"


def test_an_already_tried_image_is_never_proposed_again() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=["gcc: command not found"],
        tried=["python:3.11-slim", "python:3.11"],
    )
    assert step is None


def test_step_to_dict_is_serialisable() -> None:
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="",
        blockers=["gcc: command not found"],
    )
    assert step is not None
    assert set(step.to_dict()) == {"base_image", "rule", "rationale", "signal"}


# ---------------------------------------------------------------------------
# EscalatingResurrection — the bounded driver
# ---------------------------------------------------------------------------


def test_a_passing_first_attempt_runs_once() -> None:
    images: list[str] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        images.append(image)
        return _outcome("PASS")

    runner = EscalatingResurrection(base_image="python:3.11-slim", run_attempt=attempt)
    outcome = runner.run()

    assert outcome.verdict == "PASS"
    assert images == ["python:3.11-slim"]
    assert len(runner.attempts) == 1


def test_failure_with_evidence_retries_on_the_new_image() -> None:
    images: list[str] = []
    announced: list[str] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        images.append(image)
        if image == "python:2.7-slim":
            return _outcome("PASS", turns=22)
        return _outcome("TIMEOUT", blockers=["Missing parentheses in call to 'print'"])

    runner = EscalatingResurrection(
        base_image="python:3.11-slim", run_attempt=attempt, announce=announced.append
    )
    outcome = runner.run()

    assert outcome.verdict == "PASS"
    assert images == ["python:3.11-slim", "python:2.7-slim"]
    assert len(announced) == 1
    assert "python:2.7-slim" in announced[0]


def test_escalation_zero_disables_retries() -> None:
    images: list[str] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        images.append(image)
        return _outcome("TIMEOUT", blockers=["Missing parentheses in call to 'print'"])

    runner = EscalatingResurrection(
        base_image="python:3.11-slim", run_attempt=attempt, max_extra_attempts=0
    )
    outcome = runner.run()

    assert outcome.verdict == "TIMEOUT"
    assert images == ["python:3.11-slim"]


def test_the_extra_attempt_budget_is_a_hard_ceiling() -> None:
    """Each attempt yields fresh evidence; the cap, not the evidence, must stop it."""
    images: list[str] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        images.append(image)
        blocker = (
            "Missing parentheses in call to 'print'" if "3." in image else "gcc: command not found"
        )
        return _outcome("TIMEOUT", blockers=[blocker])

    runner = EscalatingResurrection(
        base_image="python:3.11-slim", run_attempt=attempt, max_extra_attempts=1
    )
    runner.run()

    assert images == ["python:3.11-slim", "python:2.7-slim"]


def test_the_trail_explains_why_the_run_left_each_image() -> None:
    """The justification lives on the abandoned attempt, not the one it led to.

    A contract emitted by the escalated attempt only carries the *prior* records,
    so a rationale stored on the current attempt would never reach the file.
    """

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        if image == "python:2.7-slim":
            return _outcome("PASS", turns=22)
        return _outcome("TIMEOUT", turns=40, blockers=["No module named cPickle"])

    runner = EscalatingResurrection(base_image="python:3.11-slim", run_attempt=attempt)
    runner.run()

    first, second = runner.attempts
    assert (first.base_image, first.verdict, first.turns) == ("python:3.11-slim", "TIMEOUT", 40)
    assert (first.escalated_to, first.rule) == ("python:2.7-slim", "python2_sources")
    assert "Python 2" in first.rationale
    assert first.signal == "No module named cPickle"
    # The attempt that passed was never escalated away from.
    assert (second.base_image, second.verdict, second.escalated_to) == (
        "python:2.7-slim",
        "PASS",
        "",
    )


def test_each_attempt_receives_the_trail_so_far() -> None:
    seen: list[int] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        seen.append(len(prior))
        return _outcome("TIMEOUT", blockers=["gcc: command not found"])

    EscalatingResurrection(base_image="python:3.11-slim", run_attempt=attempt).run()

    assert seen == [0, 1]


def test_attempt_record_to_dict_is_serialisable() -> None:
    record = AttemptRecord(base_image="python:3.11-slim", verdict="TIMEOUT", turns=40)
    assert set(record.to_dict()) == {
        "base_image",
        "verdict",
        "turns",
        "reason",
        "escalated_to",
        "rule",
        "rationale",
        "signal",
    }


# ---------------------------------------------------------------------------
# The loop side — collecting the evidence escalation runs on
# ---------------------------------------------------------------------------


class _Sandbox:
    """Replays a scripted list of (returncode, stdout, stderr) triples."""

    def __init__(self, results: list[tuple[int, str, str]]) -> None:
        self._results = list(results)
        self.execs: list[list[str]] = []

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        self.execs.append(list(cmd))
        rc, out, err = self._results.pop(0) if self._results else (0, "", "")
        return ExecResult(returncode=rc, stdout=out, stderr=err, duration_s=0.01)

    def snapshot(self, tag: str) -> str:
        return tag

    def last_successful_snapshot(self) -> str | None:
        return None

    @property
    def previous_turns(self) -> list[TurnRecord]:
        return []


def _spec() -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="foo",
        goal="revive foo",
        sanity_check="the run exits 0",
        base_image="python:3.11-slim",
        repo_url="https://github.com/acme/foo",
    )


def _run(
    results: list[tuple[int, str, str]],
    actions: list[RepairAction],
    tmp_path: Path,
    **kw: object,
) -> RepairOutcome:
    return RepairLoop(
        _spec(),
        _Sandbox(results),  # type: ignore[arg-type]
        ScriptedAgent(actions),
        contracts_root=tmp_path,
        **kw,  # type: ignore[arg-type]
    ).run()


def test_failing_turns_are_recorded_as_blockers(tmp_path: Path) -> None:
    outcome = _run(
        [(1, "", "gcc: command not found"), (0, "ok", "")],
        [
            RepairAction(kind="exec", cmd=["make"]),
            RepairAction(kind="exec", cmd=["pytest"], is_sanity_check=True),
        ],
        tmp_path,
    )
    assert outcome.blockers == ["gcc: command not found"]


def test_successful_turns_contribute_no_blockers(tmp_path: Path) -> None:
    outcome = _run(
        [(0, "built fine", "")],
        [RepairAction(kind="exec", cmd=["make"], is_sanity_check=True)],
        tmp_path,
    )
    assert outcome.blockers == []


def test_blockers_fall_back_to_stdout(tmp_path: Path) -> None:
    """Agents pipe `2>&1 | tail` constantly, which leaves stderr empty."""
    outcome = _run(
        [(1, "fatal error: Python.h: No such file or directory", ""), (0, "", "")],
        [
            RepairAction(kind="exec", cmd=["make"]),
            RepairAction(kind="exec", cmd=["pytest"], is_sanity_check=True),
        ],
        tmp_path,
    )
    assert outcome.blockers == ["fatal error: Python.h: No such file or directory"]


def test_repeated_failures_are_recorded_once(tmp_path: Path) -> None:
    outcome = _run(
        [(1, "", "gcc: command not found")] * 3 + [(0, "", "")],
        [RepairAction(kind="exec", cmd=["make", str(i)]) for i in range(3)]
        + [RepairAction(kind="exec", cmd=["pytest"], is_sanity_check=True)],
        tmp_path,
    )
    assert outcome.blockers == ["gcc: command not found"]


def test_the_loops_own_synthetic_failures_are_not_blockers(tmp_path: Path) -> None:
    """A suppressed no-op prints our prose, not the tool's — it is not evidence."""
    outcome = _run(
        [(1, "", "ls: cannot access '/workspace': No such file or directory")],
        [
            RepairAction(kind="exec", cmd=["bash", "-lc", "ls /workspace"]),
            # Identical, so the loop suppresses it with a synthetic rc 126.
            RepairAction(kind="exec", cmd=["bash", "-lc", "ls /workspace"]),
            RepairAction(kind="give_up", reason="stuck"),
        ],
        tmp_path,
    )
    assert outcome.blockers == ["ls: cannot access '/workspace': No such file or directory"]


def test_provenance_carries_blockers_and_the_escalation_trail(tmp_path: Path) -> None:
    prior = [
        AttemptRecord(
            base_image="python:3.11-slim",
            verdict="TIMEOUT",
            turns=40,
            escalated_to="python:2.7-slim",
            rule="python2_sources",
            rationale="the sources are Python 2",
            signal="SyntaxError: Missing parentheses in call to 'print'",
        )
    ]
    outcome = _run(
        [(1, "", "No module named cPickle"), (0, "", "")],
        [
            RepairAction(kind="exec", cmd=["make"]),
            RepairAction(kind="exec", cmd=["pytest"], is_sanity_check=True),
        ],
        tmp_path,
        prior_attempts=prior,
    )
    assert outcome.contract_dir is not None
    record = json.loads((outcome.contract_dir / "PROVENANCE.json").read_text())
    assert record["blockers"] == ["No module named cPickle"]
    assert record["escalation"] == [
        {
            "base_image": "python:3.11-slim",
            "verdict": "TIMEOUT",
            "turns": 40,
            "reason": "",
            "escalated_to": "python:2.7-slim",
            "rule": "python2_sources",
            "rationale": "the sources are Python 2",
            "signal": "SyntaxError: Missing parentheses in call to 'print'",
        }
    ]


def test_real_pypore_output_escalates_to_the_image_that_actually_worked() -> None:
    """The end-to-end chain, on text a real python:3.11 container really prints.

    ``jmschrei/PyPore`` has two modules that do not parse under Python 3. The
    resurrection that passed ran on ``python:2.7-slim``; escalation has to reach
    that image from nothing but the traceback, or it is not doing its job.
    """
    raw = (
        "running build\n"
        "Traceback (most recent call last):\n"
        '  File "setup.py", line 3, in <module>\n'
        "    from PyPore.DataTypes import *\n"
        '  File "/src/PyPore/DataTypes.py", line 970\n'
        '    print "Analysis time is {} seconds".format(t)\n'
        "          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        "SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?"
    )
    step = propose_escalation(
        base_image="python:3.11-slim",
        verdict="TIMEOUT",
        reason="turn limit (60) reached before the sanity check passed",
        blockers=[error_signature(raw)],
    )
    assert step is not None
    assert step.base_image == "python:2.7-slim"
    assert step.rule == "python2_sources"


def test_provenance_escalation_is_empty_for_a_single_attempt(tmp_path: Path) -> None:
    outcome = _run(
        [(0, "", "")],
        [RepairAction(kind="exec", cmd=["pytest"], is_sanity_check=True)],
        tmp_path,
    )
    assert outcome.contract_dir is not None
    record = json.loads((outcome.contract_dir / "PROVENANCE.json").read_text())
    assert record["escalation"] == []
