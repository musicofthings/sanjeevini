"""Integration tests for escalation's container lifecycle, against real Docker.

The decision logic in :mod:`sanjeevini.repair.escalation` is covered offline by
``test_escalation.py``. What those tests cannot reach is the part that only
exists at runtime: that an escalated attempt gets a genuinely *new* container on
the new image, seeded afresh with the repo, and never inherits the ruled-out
image's state. That is what these exercise.

No LLM is involved — a :class:`ScriptedAgent` drives real containers — so the
whole escalation path is verifiable without an API key.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from sanjeevini.repair.escalation import AttemptRecord, EscalatingResurrection
from sanjeevini.repair.loop import (
    RepairAction,
    RepairOutcome,
    ResurrectCommand,
    ResurrectionSpec,
    ScriptedAgent,
)

integration = pytest.mark.integration
requires_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker binary not found"
)

# A file that parses under Python 2 and raises SyntaxError under Python 3 — the
# decay escalation exists to escape, in one line.
_PY2_SOURCE = 'print "resurrected"\n'


def _args(tmp_path: Path, **kw: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "workdir": "/workspace",
        "docker_host": None,
        "gpus": None,
        "keep": False,
        "turns": 4,
        "escalate": 1,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _spec() -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="py2demo",
        goal="run the module",
        sanity_check="the module prints and exits 0",
        base_image="python:3.11-slim",
        repo_url="https://github.com/acme/py2demo",
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(_PY2_SOURCE)
    return repo


def _actions() -> list[RepairAction]:
    """Run the py2 module, twice — the sanity check is the second run."""
    cmd = ["bash", "-lc", "cd /workspace/repo && python main.py"]
    return [
        RepairAction(kind="exec", cmd=cmd),
        RepairAction(kind="exec", cmd=cmd, is_sanity_check=True),
    ]


@integration
@requires_docker
def test_python3_attempt_fails_and_records_a_python2_blocker(tmp_path: Path) -> None:
    """The evidence escalation needs has to survive a real container run."""
    command = ResurrectCommand(_args(tmp_path))
    outcome = command.attempt(
        "python:3.11-slim",
        [],
        spec=_spec(),
        repo_dir=_repo(tmp_path),
        agent=ScriptedAgent(_actions()),
    )

    assert outcome.verdict != "PASS"
    assert any("Missing parentheses in call to 'print'" in b for b in outcome.blockers)


@integration
@requires_docker
def test_python2_attempt_passes_on_the_escalated_image(tmp_path: Path) -> None:
    """The same unmodified source must pass once the interpreter is right."""
    command = ResurrectCommand(_args(tmp_path))
    prior = [AttemptRecord(base_image="python:3.11-slim", verdict="TIMEOUT", turns=4)]
    outcome = command.attempt(
        "python:2.7-slim",
        prior,
        spec=_spec(),
        repo_dir=_repo(tmp_path),
        agent=ScriptedAgent(_actions()),
        contracts_root=tmp_path / "contracts",
    )

    assert outcome.verdict == "PASS"
    assert outcome.blockers == []


@integration
@requires_docker
def test_the_escalated_attempt_reseeds_the_repo(tmp_path: Path) -> None:
    """A fresh container starts empty; without re-seeding the retry has no repo."""
    command = ResurrectCommand(_args(tmp_path))
    outcome = command.attempt(
        "python:2.7-slim",
        [AttemptRecord(base_image="python:3.11-slim", verdict="TIMEOUT", turns=4)],
        spec=_spec(),
        repo_dir=_repo(tmp_path),
        agent=ScriptedAgent(
            [
                RepairAction(
                    kind="exec",
                    cmd=["bash", "-lc", "test -f /workspace/repo/main.py"],
                    is_sanity_check=True,
                )
            ]
        ),
        contracts_root=tmp_path / "contracts",
    )
    assert outcome.verdict == "PASS"


@integration
@requires_docker
def test_end_to_end_escalation_switches_images_and_reaches_pass(tmp_path: Path) -> None:
    """The whole path: py3 fails, evidence is read, py2 is chosen, py2 passes."""
    command = ResurrectCommand(_args(tmp_path))
    spec, repo = _spec(), _repo(tmp_path)
    announced: list[str] = []

    def attempt(image: str, prior: list[AttemptRecord]) -> RepairOutcome:
        return command.attempt(
            image,
            prior,
            spec=spec,
            repo_dir=repo,
            agent=ScriptedAgent(_actions()),
            contracts_root=tmp_path / "contracts",
        )

    runner = EscalatingResurrection(
        base_image=spec.base_image,
        run_attempt=attempt,
        max_extra_attempts=1,
        announce=announced.append,
    )
    outcome = runner.run()

    assert outcome.verdict == "PASS"
    assert [a.base_image for a in runner.attempts] == ["python:3.11-slim", "python:2.7-slim"]
    assert runner.attempts[0].rule == "python2_sources"
    assert announced and "python:2.7-slim" in announced[0]
    # The spec the contract is emitted from must name the image that actually won.
    assert spec.base_image == "python:2.7-slim"
