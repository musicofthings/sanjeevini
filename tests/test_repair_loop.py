"""Tests for sanjeevini.repair.loop (target: 70% branch coverage)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from sanjeevini.contracts.schema import ContractSchema
from sanjeevini.repair.loop import (
    DecayCheckCommand,
    DecayVerdict,
    LoopState,
    RepairAction,
    RepairLoop,
    RepairOutcome,
    ResurrectCommand,
    ResurrectionSpec,
    ScriptedAgent,
    classify_decay,
    run_decay_check,
    select_plan,
    snapshot_from_dir,
    spec_from_plan,
    tool_slug,
)
from sanjeevini.sandbox.checkpoint import TurnRecord
from sanjeevini.sandbox.docker_sandbox import ExecResult
from sanjeevini.scouts.python_scout import PythonResurrectionPlan
from sanjeevini.scouts.r_scout import RResurrectionPlan
from sanjeevini.scouts.workflow_scout import WorkflowResurrectionPlan

_BIOC_DESCRIPTION = """\
Package: MyBioPkg
Title: Differential Expression
Version: 1.2.0
Description: Tools for DE.
Depends: R (>= 4.1), methods
biocViews: RNASeq
"""


# ---- fakes ----------------------------------------------------------------


class FakeSandbox:
    """A minimal SandboxProtocol stand-in that records exec/snapshot calls."""

    def __init__(
        self,
        returncodes: list[int] | None = None,
        prior: list[TurnRecord] | None = None,
    ) -> None:
        self._returncodes = list(returncodes or [])
        self._prior = list(prior or [])
        self.execs: list[list[str]] = []
        self.snapshots: list[str] = []

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        self.execs.append(list(cmd))
        rc = self._returncodes.pop(0) if self._returncodes else 0
        return ExecResult(returncode=rc, stdout="out", stderr="err", duration_s=0.01)

    def snapshot(self, tag: str) -> str:
        self.snapshots.append(tag)
        return tag

    def last_successful_snapshot(self) -> str | None:
        return None

    @property
    def previous_turns(self) -> list[TurnRecord]:
        return self._prior


class InfiniteAgent:
    """Always returns a non-sanity exec, so the loop hits its turn limit."""

    def next_action(self, state: LoopState) -> RepairAction:
        return RepairAction(kind="exec", cmd=["true"])


def _spec(**kw: object) -> ResurrectionSpec:
    base = {
        "tool_slug": "foo",
        "goal": "revive foo",
        "sanity_check": "output VCF contains ≥ 1 variant record",
        "base_image": "python:3.10-slim",
        "repo_url": "https://github.com/acme/foo",
    }
    base.update(kw)
    return ResurrectionSpec(**base)  # type: ignore[arg-type]


# ---- slugs / snapshots ----------------------------------------------------


def test_tool_slug_from_url_and_name() -> None:
    assert tool_slug("https://github.com/acme/My_Tool.git") == "my-tool"
    assert tool_slug("Sniffles2") == "sniffles2"


def test_snapshot_from_dir_reads_present_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "requirements.txt").write_text("numpy")
    snap = snapshot_from_dir(tmp_path, "https://github.com/acme/foo")
    assert snap.get("README.md") == "hello"
    assert snap.has("requirements.txt")
    assert not snap.has("DESCRIPTION")


# ---- plan dispatch --------------------------------------------------------


async def test_select_plan_dispatches_to_workflow_for_main_nf(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=2\nworkflow { }")
    plan = await select_plan(tmp_path, "https://github.com/acme/pipe", confirm=False)
    assert isinstance(plan, WorkflowResurrectionPlan)
    assert plan.language == "nextflow"


async def test_select_plan_dispatches_to_r_for_description(tmp_path: Path) -> None:
    (tmp_path / "DESCRIPTION").write_text(_BIOC_DESCRIPTION)
    plan = await select_plan(tmp_path, "https://github.com/acme/rpkg", confirm=False)
    assert isinstance(plan, RResurrectionPlan)
    assert plan.bioc_release == "3.14"


async def test_select_plan_defaults_to_python(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("A tool that emits sequences.")
    (tmp_path / "requirements.txt").write_text("torch\n")
    plan = await select_plan(tmp_path, "https://github.com/acme/pytool", confirm=False)
    assert isinstance(plan, PythonResurrectionPlan)
    assert plan.framework == "pytorch"


# ---- spec normalisation ---------------------------------------------------


def test_spec_from_workflow_plan_sets_language_and_base_image() -> None:
    plan = WorkflowResurrectionPlan(
        language="nextflow",
        entry_point="nextflow run main.nf -profile docker",
        container_strategy="-profile docker",
        runner_version_pin=None,
        sanity_check="pipeline exits 0 and output dir has ≥ 1 file",
    )
    spec = spec_from_plan(plan, url="https://github.com/acme/pipe")
    assert spec.workflow_type == "nextflow"
    assert spec.base_image.startswith("nextflow/nextflow")
    assert spec.entry_command == "nextflow run main.nf -profile docker"


def test_spec_from_python_plan_detects_gpu() -> None:
    plan = PythonResurrectionPlan(
        capability="x",
        base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel",
        goal="g",
        sanity_check="≥ 1 variant",
        test_input="t",
    )
    spec = spec_from_plan(plan, url="https://github.com/acme/foo")
    assert spec.workflow_type == "python"
    assert spec.gpu_required is True


def test_spec_from_r_plan() -> None:
    plan = RResurrectionPlan(
        capability="x",
        bioc_release="3.14",
        r_version="4.1",
        base_image="rocker/bioconductor:3.14",
        goal="g",
        sanity_check="0 errors",
        package_name="MyBioPkg",
    )
    spec = spec_from_plan(plan, url="https://github.com/acme/rpkg")
    assert spec.workflow_type == "r"
    assert spec.base_image == "rocker/bioconductor:3.14"


# ---- repair loop ----------------------------------------------------------


def test_loop_pass_emits_full_contract(tmp_path: Path) -> None:
    actions = [
        RepairAction(
            kind="exec",
            cmd=["pip", "install", "."],
            patch="--- a/install.sh\n+++ b/install.sh",
            bug_class="dead_endpoint",
            bug_description="release URL moved",
        ),
        RepairAction(
            kind="exec",
            cmd=["python", "predict.py"],
            is_sanity_check=True,
            cost_usd=0.5,
        ),
    ]
    sandbox = FakeSandbox(returncodes=[0, 0])
    loop = RepairLoop(
        _spec(),
        sandbox,
        ScriptedAgent(actions),
        max_turns=10,
        contracts_root=tmp_path,
        today="2026-07-19",
    )
    outcome = loop.run()

    assert outcome.verdict == "PASS"
    assert outcome.turns == 2
    assert outcome.cost_usd == 0.5
    assert outcome.sanity_cmd == ["python", "predict.py"]
    assert outcome.bugs_fixed[0]["class"] == "dead_endpoint"
    assert sandbox.snapshots[-1] == "sanjeevini/foo:resurrected"

    d = tmp_path / "foo"
    for name in ("contract.yaml", "predict.py", "Dockerfile", "smoke_test.sh",
                 "REPRODUCE.md", "PROVENANCE.json"):
        assert (d / name).is_file(), name

    # contract.yaml embeds a valid ContractSchema
    payload = yaml.safe_load((d / "contract.yaml").read_text())
    ContractSchema.model_validate(payload["schema"])

    # smoke_test.sh runs the sanity command; predict.py records it
    assert "python predict.py" in (d / "smoke_test.sh").read_text()
    assert '"python"' in (d / "predict.py").read_text()


def test_provenance_written_on_pass_has_required_keys(tmp_path: Path) -> None:
    sandbox = FakeSandbox(returncodes=[0])
    loop = RepairLoop(
        _spec(),
        sandbox,
        ScriptedAgent([RepairAction(kind="exec", cmd=["t"], is_sanity_check=True)]),
        contracts_root=tmp_path,
        today="2026-07-19",
    )
    outcome = loop.run()
    prov = json.loads((tmp_path / "foo" / "PROVENANCE.json").read_text())
    assert prov["schema_version"] == "1.0"
    assert prov["sanity_check_verdict"] == "PASS"
    assert prov["turn_count"] == outcome.turns
    assert prov["resurrection_date"] == "2026-07-19"
    assert prov["final_image"] == "sanjeevini/foo:resurrected"


def test_loop_give_up_is_failed_and_writes_provenance_only(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    loop = RepairLoop(_spec(), sandbox, ScriptedAgent([]), contracts_root=tmp_path)
    outcome = loop.run()
    assert outcome.verdict == "FAILED"
    assert outcome.turns == 0
    d = tmp_path / "foo"
    assert (d / "PROVENANCE.json").is_file()
    assert not (d / "contract.yaml").exists()
    prov = json.loads((d / "PROVENANCE.json").read_text())
    assert prov["sanity_check_verdict"] == "FAILED"


def test_loop_enforces_turn_limit(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    loop = RepairLoop(
        _spec(), sandbox, InfiniteAgent(), max_turns=3, contracts_root=tmp_path
    )
    outcome = loop.run()
    assert outcome.verdict == "TIMEOUT"
    assert outcome.turns == 3
    assert len(sandbox.execs) == 3


def test_loop_resumes_from_prior_turns(tmp_path: Path) -> None:
    prior = [
        TurnRecord(turn=1, cmd=["a"], returncode=0, stdout="", stderr="", duration_s=0.0),
        TurnRecord(turn=2, cmd=["b"], returncode=0, stdout="", stderr="", duration_s=0.0),
    ]
    sandbox = FakeSandbox(returncodes=[0], prior=prior)
    loop = RepairLoop(
        _spec(),
        sandbox,
        ScriptedAgent([RepairAction(kind="exec", cmd=["t"], is_sanity_check=True)]),
        max_turns=10,
        contracts_root=tmp_path,
    )
    outcome = loop.run()
    assert outcome.verdict == "PASS"
    assert outcome.turns == 3  # 2 resumed + 1 new


def test_loop_turn_limit_already_reached_on_resume(tmp_path: Path) -> None:
    prior = [
        TurnRecord(turn=1, cmd=["a"], returncode=0, stdout="", stderr="", duration_s=0.0),
        TurnRecord(turn=2, cmd=["b"], returncode=0, stdout="", stderr="", duration_s=0.0),
    ]
    sandbox = FakeSandbox(prior=prior)
    loop = RepairLoop(
        _spec(), sandbox, InfiniteAgent(), max_turns=2, contracts_root=tmp_path
    )
    outcome = loop.run()
    assert outcome.verdict == "TIMEOUT"
    assert outcome.turns == 2
    assert sandbox.execs == []


# ---- decay check ----------------------------------------------------------


@pytest.mark.parametrize(
    ("naive_runs", "stage", "expected"),
    [
        (True, "example", "naive_runs"),
        (False, "install", "install_fails"),
        (False, "clone", "install_fails"),
        (False, "example", "run_fails"),
        (None, "unknown", "unknown"),
        (False, "timeout", "unknown"),
    ],
)
def test_classify_decay(naive_runs: bool | None, stage: str, expected: str) -> None:
    assert classify_decay(naive_runs, stage) == expected


def test_run_decay_check_with_injected_probe() -> None:
    def probe(url: str, mode: str, host: str | None) -> tuple[bool | None, str, str]:
        return False, "install", "pip_requirements_failed"

    result = run_decay_check("https://github.com/acme/foo", probe=probe)
    assert result.verdict == "install_fails"
    assert result.decayed is True
    assert result.to_dict() == {
        "verdict": "install_fails",
        "stage": "install",
        "reason": "pip_requirements_failed",
        "url": "https://github.com/acme/foo",
    }


def test_decay_verdict_not_decayed_when_runs() -> None:
    v = DecayVerdict(url="u", verdict="naive_runs", stage="example", reason="ran_ok")
    assert v.decayed is False


def test_decay_check_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    def probe(url: str, mode: str, host: str | None) -> tuple[bool | None, str, str]:
        return True, "example", "ran_ok"

    args = argparse.Namespace(
        url="https://github.com/acme/foo", sandbox="docker", json=True, fail_on_decay=False
    )
    DecayCheckCommand(args, probe=probe).run()
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"verdict", "stage", "reason", "url"}
    assert out["verdict"] == "naive_runs"


def test_decay_check_command_fail_on_decay_exits(capsys: pytest.CaptureFixture[str]) -> None:
    def probe(url: str, mode: str, host: str | None) -> tuple[bool | None, str, str]:
        return False, "example", "example_failed"

    args = argparse.Namespace(
        url="https://github.com/acme/foo", sandbox="docker", json=False, fail_on_decay=True
    )
    with pytest.raises(SystemExit) as exc:
        DecayCheckCommand(args, probe=probe).run()
    assert exc.value.code == 1
    assert "run_fails" in capsys.readouterr().out


# ---- resurrect command ----------------------------------------------------


def test_resurrect_build_agent_returns_llm_agent_or_needs_anthropic() -> None:
    from sanjeevini.repair.agent import LLMRepairAgent

    args = argparse.Namespace(url="https://github.com/acme/foo")
    try:
        agent = ResurrectCommand(args)._build_agent(_spec())
    except RuntimeError as exc:
        assert "anthropic" in str(exc)  # package not installed in this env
    else:
        assert isinstance(agent, LLMRepairAgent)


def test_report_prints_contract(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(url="https://github.com/acme/foo")
    outcome = RepairOutcome(
        verdict="PASS", turns=3, cost_usd=0.1, contract_dir=Path("/tmp/c/foo"),
        sanity_cmd=["t"],
    )
    ResurrectCommand(args)._report(outcome)
    out = capsys.readouterr().out
    assert "PASS" in out and "smoke test" in out
