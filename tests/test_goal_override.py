"""Tests for ``--goal-file`` / ``--no-scout``, which let a caller supply the goal
and pass criterion instead of the Scout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sanjeevini.repair.loop import (
    GoalOverride,
    ResurrectCommand,
    ResurrectionSpec,
    _apply_override,
    parse_goal_file,
)

FALSIFIABLE = "the JSON output parses and contains ≥ 10 detected events"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "goal.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _args(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = {"goal_file": None, "image": None, "no_scout": False}
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# parse_goal_file
# ---------------------------------------------------------------------------


def test_plain_text_becomes_the_goal(tmp_path: Path) -> None:
    override = parse_goal_file(_write(tmp_path, "Revive PyPore on Python 2.7."))
    assert override.goal == "Revive PyPore on Python 2.7."
    assert override.sanity_check == ""


def test_yaml_sets_both_fields(tmp_path: Path) -> None:
    override = parse_goal_file(
        _write(tmp_path, f"goal: Revive PyPore\nsanity_check: {FALSIFIABLE}\n")
    )
    assert override.goal == "Revive PyPore"
    assert override.sanity_check == FALSIFIABLE


def test_yaml_may_set_only_the_sanity_check(tmp_path: Path) -> None:
    override = parse_goal_file(_write(tmp_path, f"sanity_check: {FALSIFIABLE}\n"))
    assert override.goal == ""
    assert override.sanity_check == FALSIFIABLE


def test_an_unfalsifiable_override_is_rejected(tmp_path: Path) -> None:
    # Overriding the criterion must not be a way to smuggle in an unfalsifiable
    # claim — that would defeat the whole guarantee.
    path = _write(tmp_path, "sanity_check: the tool runs without error\n")
    with pytest.raises(ValueError, match="falsifiable"):
        parse_goal_file(path)


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_goal_file(_write(tmp_path, "   \n"))


def test_a_mapping_with_neither_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither"):
        parse_goal_file(_write(tmp_path, "unrelated: value\n"))


def test_a_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        parse_goal_file(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# _apply_override
# ---------------------------------------------------------------------------


def _spec() -> ResurrectionSpec:
    return ResurrectionSpec(
        tool_slug="foo",
        goal="scout goal",
        sanity_check="scout check with ≥ 1 record",
        base_image="python:3.10",
    )


def test_override_replaces_only_what_it_supplies() -> None:
    spec = _spec()
    _apply_override(spec, GoalOverride(sanity_check=FALSIFIABLE))
    assert spec.goal == "scout goal"
    assert spec.sanity_check == FALSIFIABLE


def test_an_empty_override_changes_nothing() -> None:
    spec = _spec()
    _apply_override(spec, GoalOverride())
    assert spec.goal == "scout goal"
    assert spec.sanity_check == "scout check with ≥ 1 record"


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_no_goal_file_yields_an_empty_override() -> None:
    assert ResurrectCommand(_args())._goal_override() == GoalOverride()


def test_a_bad_goal_file_exits_cleanly(tmp_path: Path) -> None:
    path = _write(tmp_path, "sanity_check: it works\n")
    with pytest.raises(SystemExit) as exc:
        ResurrectCommand(_args(goal_file=str(path)))._goal_override()
    assert exc.value.code == 1


def test_no_scout_builds_a_spec_from_the_cli(tmp_path: Path) -> None:
    cmd = ResurrectCommand(_args(no_scout=True, image="python:2.7-slim"))
    spec = cmd._spec_without_scout(
        "https://github.com/jmschrei/PyPore",
        "abc123",
        GoalOverride(goal="Revive PyPore", sanity_check=FALSIFIABLE),
    )
    assert spec.tool_slug == "pypore"
    assert spec.base_image == "python:2.7-slim"
    assert spec.sanity_check == FALSIFIABLE
    assert spec.repo_commit == "abc123"


@pytest.mark.parametrize(
    "image,override",
    [
        (None, GoalOverride(goal="g", sanity_check=FALSIFIABLE)),
        ("python:2.7-slim", GoalOverride(goal="g")),
        ("python:2.7-slim", GoalOverride(sanity_check=FALSIFIABLE)),
    ],
)
def test_no_scout_requires_image_goal_and_check(image: str | None, override: GoalOverride) -> None:
    cmd = ResurrectCommand(_args(no_scout=True, image=image))
    with pytest.raises(SystemExit) as exc:
        cmd._spec_without_scout("https://github.com/acme/foo", "", override)
    assert exc.value.code == 1
