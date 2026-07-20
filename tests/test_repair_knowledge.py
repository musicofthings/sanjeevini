"""Tests for cross-run learning — the knowledge store, the loop's recording of
lessons, and the agent's retrieval of them into the prompt."""

from __future__ import annotations

from pathlib import Path

from sanjeevini.repair.agent import LLMRepairAgent, render_state
from sanjeevini.repair.knowledge import (
    KnowledgeStore,
    Lesson,
    error_signature,
    lessons_from_bugs,
)
from sanjeevini.repair.loop import (
    LoopState,
    RepairAction,
    RepairLoop,
    ResurrectionSpec,
    ScriptedAgent,
)
from sanjeevini.sandbox.docker_sandbox import ExecResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSandbox:
    """A sandbox that returns a scripted exit code per call."""

    def __init__(self, returncodes: list[int], stderr: str = "") -> None:
        self._returncodes = list(returncodes)
        self._stderr = stderr
        self.previous_turns: list[object] = []

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        rc = self._returncodes.pop(0) if self._returncodes else 0
        return ExecResult(rc, "", self._stderr if rc else "", 0.1)

    def snapshot(self, tag: str) -> str:
        return tag

    def last_successful_snapshot(self) -> str | None:
        return None


def _spec(framework: str = "tensorflow-1.x") -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="demo",
        goal="resurrect demo",
        sanity_check="the command exits 0",
        base_image="python:3.10",
        framework=framework,
    )


def _state(last_stderr: str = "") -> LoopState:
    return LoopState(
        turn=2,
        max_turns=10,
        goal="resurrect demo",
        sanity_check="exits 0",
        base_image="python:3.10",
        last_returncode=1,
        last_stdout="",
        last_stderr=last_stderr,
        patch_history=[],
        history=[],
    )


# ---------------------------------------------------------------------------
# error_signature
# ---------------------------------------------------------------------------


def test_error_signature_prefers_the_error_looking_line() -> None:
    text = "building wheels\nrunning setup.py\nImportError: No module named foo\n"
    assert error_signature(text) == "ImportError: No module named foo"


def test_error_signature_falls_back_to_the_last_line() -> None:
    assert error_signature("step one\nstep two\n") == "step two"


def test_error_signature_of_blank_text_is_empty() -> None:
    assert error_signature("   \n\n") == ""


def test_error_signature_is_bounded() -> None:
    assert len(error_signature("Error: " + "x" * 500, limit=50)) == 50


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


def test_store_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.json"
    store = KnowledgeStore(path)
    assert store.add(Lesson("dep_conflict", "numpy is too new", "pin numpy<1.24"))

    reopened = KnowledgeStore(path)
    assert [lesson.fix for lesson in reopened.all()] == ["pin numpy<1.24"]


def test_store_deduplicates_identical_lessons(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    lesson = Lesson("dep_conflict", "numpy too new", "pin numpy")
    assert store.add(lesson) is True
    assert store.add(Lesson("dep_conflict", "numpy too new", "pin numpy")) is False
    assert len(store.all()) == 1


def test_store_rejects_an_empty_lesson(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    assert store.add(Lesson("unknown", "", "")) is False


def test_store_survives_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text("{not json", encoding="utf-8")
    assert KnowledgeStore(path).all() == []


def test_store_ignores_a_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text('{"lessons": []}', encoding="utf-8")
    assert KnowledgeStore(path).all() == []


def test_extend_reports_how_many_were_new(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    lessons = [Lesson("a", "sym one", "fix one"), Lesson("b", "sym two", "fix two")]
    assert store.extend(lessons) == 2
    assert store.extend(lessons) == 0


def test_relevance_ranks_symptom_overlap_above_unrelated(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    store.add(Lesson("dep", "ImportError: No module named tensorflow", "pip install tf"))
    store.add(Lesson("other", "disk quota exceeded", "free space"))

    hits = store.relevant(error_text="ImportError: No module named tensorflow")
    assert [lesson.fix for lesson in hits] == ["pip install tf"]


def test_relevance_scores_a_framework_match(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    store.add(Lesson("dep", "unrelated symptom", "fix", framework="tensorflow-1.x"))
    assert store.relevant(framework="tensorflow-1.x") != []
    assert store.relevant(framework="plain-python") == []


def test_relevance_honours_top_k(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    for i in range(6):
        store.add(Lesson("dep", f"import error number {i}", f"fix {i}"))
    assert len(store.relevant(error_text="import error", top_k=3)) == 3


def test_relevance_with_no_context_returns_nothing(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    store.add(Lesson("dep", "some symptom", "some fix"))
    assert store.relevant() == []


# ---------------------------------------------------------------------------
# lessons_from_bugs
# ---------------------------------------------------------------------------


def test_lessons_from_bugs_prefers_the_description() -> None:
    bugs = [{"class": "dep", "description": "pinned numpy", "patch": "--- a\n+++ b\n+x"}]
    lessons = lessons_from_bugs(bugs, framework="py", tool="demo")
    assert lessons[0].fix == "pinned numpy"
    assert lessons[0].framework == "py"
    assert lessons[0].tool == "demo"


def test_lessons_from_bugs_falls_back_to_the_patch_body() -> None:
    bugs = [{"class": "dep", "description": "", "patch": "--- a\n+++ b\n@@\n+numpy<1.24"}]
    assert lessons_from_bugs(bugs)[0].fix == "+numpy<1.24"


def test_lessons_from_bugs_skips_a_bug_with_no_usable_fix() -> None:
    assert lessons_from_bugs([{"class": "dep", "description": "", "patch": ""}]) == []


def test_lesson_hint_includes_class_symptom_and_fix() -> None:
    hint = Lesson("dep_conflict", "numpy too new", "pin it", framework="py").as_hint()
    assert "dep_conflict" in hint and "numpy too new" in hint and "pin it" in hint


# ---------------------------------------------------------------------------
# Loop records, agent retrieves
# ---------------------------------------------------------------------------


def test_loop_records_a_lesson_with_the_symptom_it_responded_to(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    agent = ScriptedAgent(
        [
            RepairAction(kind="exec", cmd=["run"]),
            RepairAction(
                kind="exec",
                cmd=["fix"],
                patch="--- a\n+++ b\n+pin",
                bug_class="dep_conflict",
                bug_description="pinned numpy to the commit era",
                is_sanity_check=True,
            ),
        ]
    )
    # Turn 1 fails with the traceback; turn 2 patches it and passes.
    sandbox = FakeSandbox([1, 0], stderr="ImportError: No module named numpy")
    loop = RepairLoop(
        _spec(), sandbox, agent, contracts_root=tmp_path / "contracts", knowledge=store
    )
    outcome = loop.run()

    assert outcome.verdict == "PASS"
    assert outcome.bugs_fixed[0]["symptom"] == "ImportError: No module named numpy"

    stored = KnowledgeStore(tmp_path / "k.json").all()
    assert len(stored) == 1
    assert stored[0].fix == "pinned numpy to the commit era"
    assert stored[0].framework == "tensorflow-1.x"


def test_loop_without_a_store_still_runs(tmp_path: Path) -> None:
    agent = ScriptedAgent([RepairAction(kind="exec", cmd=["run"], is_sanity_check=True)])
    loop = RepairLoop(_spec(), FakeSandbox([0]), agent, contracts_root=tmp_path / "contracts")
    assert loop.run().verdict == "PASS"


def test_agent_injects_relevant_prior_lessons_into_the_prompt(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "k.json")
    store.add(Lesson("dep", "ImportError: No module named numpy", "pin numpy<1.24"))

    prompts: list[str] = []

    def fake_complete(system: str, user: str) -> tuple[str, float]:
        prompts.append(user)
        return '{"action": "exec", "cmd": ["ls"]}', 0.0

    agent = LLMRepairAgent(_spec(), complete=fake_complete, knowledge=store)
    agent.next_action(_state("ImportError: No module named numpy"))

    assert "Prior experience from earlier resurrections" in prompts[0]
    assert "pin numpy<1.24" in prompts[0]


def test_agent_without_a_store_omits_the_lessons_section() -> None:
    prompts: list[str] = []

    def fake_complete(system: str, user: str) -> tuple[str, float]:
        prompts.append(user)
        return '{"action": "exec", "cmd": ["ls"]}', 0.0

    LLMRepairAgent(_spec(), complete=fake_complete).next_action(_state("boom"))
    assert "Prior experience" not in prompts[0]


def test_render_state_omits_the_section_for_an_empty_lesson_list() -> None:
    assert "Prior experience" not in render_state(_spec(), _state("boom"), [])
