"""Tests for the empirical half of the sanity-check quality gate — verifying a
check's structural claim against the files a run actually produced."""

from __future__ import annotations

import json
from pathlib import Path

from sanjeevini.contracts.output_type import GENERIC_CHECK
from sanjeevini.repair.loop import (
    RepairAction,
    RepairLoop,
    ResurrectionSpec,
    ScriptedAgent,
)
from sanjeevini.sandbox.docker_sandbox import DockerError, ExecResult

BAM_CHECK = "the BAM output passes `samtools quickcheck` and contains ≥ 1 alignment"


class ProbeSandbox:
    """A sandbox whose final exec (the evidence probe) returns scripted output."""

    def __init__(
        self,
        probe_stdout: str = "",
        probe_rc: int = 0,
        probe_raises: Exception | None = None,
    ) -> None:
        self._probe_stdout = probe_stdout
        self._probe_rc = probe_rc
        self._probe_raises = probe_raises
        self.execs: list[list[str]] = []
        self.previous_turns: list[object] = []

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        self.execs.append(list(cmd))
        if cmd[0] == "find":
            if self._probe_raises is not None:
                raise self._probe_raises
            return ExecResult(self._probe_rc, self._probe_stdout, "", 0.1)
        return ExecResult(0, "ok", "", 0.1)

    def snapshot(self, tag: str) -> str:
        return tag

    def last_successful_snapshot(self) -> str | None:
        return None


def _run(spec_check: str, sandbox: ProbeSandbox, tmp_path: Path):
    spec = ResurrectionSpec(
        tool_slug="foo",
        goal="revive foo",
        sanity_check=spec_check,
        base_image="python:3.10",
        repo_url="https://github.com/acme/foo",
    )
    agent = ScriptedAgent([RepairAction(kind="exec", cmd=["run"], is_sanity_check=True)])
    return RepairLoop(spec, sandbox, agent, contracts_root=tmp_path).run()


def test_a_claim_backed_by_real_files_is_supported(tmp_path: Path) -> None:
    sandbox = ProbeSandbox(probe_stdout="./out/sample.bam\n")
    outcome = _run(BAM_CHECK, sandbox, tmp_path)

    assert outcome.verdict == "PASS"
    assert outcome.evidence.status == "supported"
    assert outcome.evidence.found == ["./out/sample.bam"]
    assert not outcome.evidence.contradicts_claim


def test_a_claim_with_no_matching_file_is_unsupported(tmp_path: Path) -> None:
    # The NanoFilt failure mode: the check names BAM, the run emits none.
    sandbox = ProbeSandbox(probe_stdout="")
    outcome = _run(BAM_CHECK, sandbox, tmp_path)

    assert outcome.evidence.status == "unsupported"
    assert outcome.evidence.contradicts_claim


def test_an_unsupported_claim_does_not_overturn_the_verdict(tmp_path: Path) -> None:
    # A real exit code outranks a filesystem heuristic — this is a qualifier.
    outcome = _run(BAM_CHECK, ProbeSandbox(probe_stdout=""), tmp_path)
    assert outcome.verdict == "PASS"


def test_an_untyped_check_has_nothing_to_verify(tmp_path: Path) -> None:
    sandbox = ProbeSandbox()
    outcome = _run(GENERIC_CHECK, sandbox, tmp_path)

    assert outcome.evidence.status == "untyped"
    assert not outcome.evidence.contradicts_claim
    # No probe should have run at all.
    assert all(cmd[0] != "find" for cmd in sandbox.execs)


def test_a_probe_that_errors_is_unknown_not_unsupported(tmp_path: Path) -> None:
    outcome = _run(BAM_CHECK, ProbeSandbox(probe_raises=DockerError("gone")), tmp_path)
    assert outcome.evidence.status == "unknown"
    assert not outcome.evidence.contradicts_claim


def test_a_probe_that_times_out_is_unknown(tmp_path: Path) -> None:
    outcome = _run(BAM_CHECK, ProbeSandbox(probe_raises=TimeoutError("slow")), tmp_path)
    assert outcome.evidence.status == "unknown"


def test_a_nonzero_probe_is_unknown(tmp_path: Path) -> None:
    outcome = _run(BAM_CHECK, ProbeSandbox(probe_rc=1), tmp_path)
    assert outcome.evidence.status == "unknown"


def test_a_failed_run_is_not_probed(tmp_path: Path) -> None:
    spec = ResurrectionSpec(
        tool_slug="foo", goal="g", sanity_check=BAM_CHECK, base_image="python:3.10"
    )
    sandbox = ProbeSandbox()
    agent = ScriptedAgent([RepairAction(kind="give_up", reason="unresolvable")])
    outcome = RepairLoop(spec, sandbox, agent, contracts_root=tmp_path).run()

    assert outcome.verdict == "FAILED"
    assert outcome.evidence.status == "untyped"
    assert sandbox.execs == []


def test_evidence_lands_in_the_provenance_record(tmp_path: Path) -> None:
    _run(BAM_CHECK, ProbeSandbox(probe_stdout="./x.bam\n"), tmp_path)
    record = json.loads((tmp_path / "foo" / "PROVENANCE.json").read_text())

    assert record["sanity_check_evidence"]["status"] == "supported"
    assert record["sanity_check_evidence"]["found"] == ["./x.bam"]


def test_reproduce_md_flags_an_unsupported_claim(tmp_path: Path) -> None:
    _run(BAM_CHECK, ProbeSandbox(probe_stdout=""), tmp_path)
    text = (tmp_path / "foo" / "REPRODUCE.md").read_text()

    assert "no such file was found" in text
    assert "more scepticism" in text


def test_reproduce_md_lists_supporting_artefacts(tmp_path: Path) -> None:
    _run(BAM_CHECK, ProbeSandbox(probe_stdout="./out/a.bam\n"), tmp_path)
    text = (tmp_path / "foo" / "REPRODUCE.md").read_text()

    assert "## Evidence" in text
    assert "./out/a.bam" in text


def test_reproduce_md_omits_the_section_for_an_untyped_check(tmp_path: Path) -> None:
    _run(GENERIC_CHECK, ProbeSandbox(), tmp_path)
    assert "## Evidence" not in (tmp_path / "foo" / "REPRODUCE.md").read_text()


def test_the_probe_searches_for_every_claimed_extension(tmp_path: Path) -> None:
    sandbox = ProbeSandbox(probe_stdout="./x.bam\n")
    _run(BAM_CHECK, sandbox, tmp_path)
    probe = next(cmd for cmd in sandbox.execs if cmd[0] == "find")

    assert "*.bam" in probe and "*.cram" in probe
    # Non-empty files only — a zero-byte BAM proves nothing.
    assert "+0c" in probe


def test_found_files_are_capped(tmp_path: Path) -> None:
    sandbox = ProbeSandbox(probe_stdout="\n".join(f"./f{i}.bam" for i in range(50)))
    outcome = _run(BAM_CHECK, sandbox, tmp_path)
    assert len(outcome.evidence.found) == 20
