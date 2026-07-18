"""Tests for sanjeevini.sandbox.checkpoint (target: 90% branch coverage)."""

from __future__ import annotations

import json
from pathlib import Path

from sanjeevini.sandbox.checkpoint import (
    STDERR_LIMIT,
    STDOUT_LIMIT,
    CheckpointStore,
    TurnRecord,
)


def _record(turn: int, *, returncode: int = 0, tag: str | None = None, **kw) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        cmd=["echo", str(turn)],
        returncode=returncode,
        stdout=kw.get("stdout", "out"),
        stderr=kw.get("stderr", "err"),
        duration_s=kw.get("duration_s", 0.1),
        snapshot_tag=tag,
        timestamp="2026-07-18T00:00:00+00:00",
        cost_usd=kw.get("cost_usd"),
    )


def test_read_all_empty(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    assert store.read_all() == []
    assert store.latest() is None
    assert store.last_successful_snapshot() is None
    assert store.next_turn() == 1
    assert store.cost_usd() == 0.0


def test_write_and_read_all_sorted(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    # write out of order
    for t in (3, 1, 2):
        store.write(_record(t))
    turns = [r.turn for r in store.read_all()]
    assert turns == [1, 2, 3]


def test_latest_and_next_turn(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1))
    store.write(_record(2))
    assert store.latest().turn == 2
    assert store.next_turn() == 3


def test_last_successful_snapshot(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1, returncode=0, tag="img:turn-1"))
    store.write(_record(2, returncode=0, tag="img:turn-2"))
    store.write(_record(3, returncode=1, tag=None))  # failed turn, most recent
    # most recent *successful* snapshot wins
    assert store.last_successful_snapshot() == "img:turn-2"


def test_last_successful_snapshot_ignores_success_without_tag(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1, returncode=0, tag="img:turn-1"))
    store.write(_record(2, returncode=0, tag=None))  # succeeded but not snapshotted
    assert store.last_successful_snapshot() == "img:turn-1"


def test_atomic_write_leaves_no_tmp(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1))
    assert list(tmp_checkpoint_dir.glob("*.tmp")) == []


def test_manifest_written_and_lists_turns(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1))
    store.write(_record(2))
    manifest = json.loads((tmp_checkpoint_dir / "manifest.json").read_text())
    assert manifest["turns"] == [1, 2]


def test_manifest_fallback_to_glob(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1))
    store.write(_record(2))
    (tmp_checkpoint_dir / "manifest.json").unlink()  # corrupt/missing manifest
    # read_all still recovers via glob
    assert [r.turn for r in store.read_all()] == [1, 2]


def test_stdout_stderr_truncated_on_write(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    big = "x" * (STDOUT_LIMIT * 2)
    big_err = "y" * (STDERR_LIMIT * 2)
    store.write(_record(1, stdout=big, stderr=big_err))
    rec = store.read_all()[0]
    assert len(rec.stdout.encode("utf-8")) <= STDOUT_LIMIT + 32
    assert len(rec.stderr.encode("utf-8")) <= STDERR_LIMIT + 32
    assert rec.stdout.endswith("[truncated]")


def test_cost_usd_sums_present_values(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1, cost_usd=0.5))
    store.write(_record(2, cost_usd=None))  # missing cost contributes 0
    store.write(_record(3, cost_usd=0.25))
    assert store.cost_usd() == 0.75


def test_overwrite_same_turn(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1, returncode=1))
    store.write(_record(1, returncode=0, tag="img:x"))
    records = store.read_all()
    assert len(records) == 1
    assert records[0].ok
    assert json.loads((tmp_checkpoint_dir / "manifest.json").read_text())["turns"] == [1]


def test_from_dict_ignores_unknown_keys() -> None:
    data = _record(1).to_dict()
    data["future_field"] = "ignored"
    rec = TurnRecord.from_dict(data)
    assert rec.turn == 1
    assert not hasattr(rec, "future_field")


def test_read_all_skips_missing_file(tmp_checkpoint_dir: Path) -> None:
    store = CheckpointStore(tmp_checkpoint_dir)
    store.write(_record(1))
    store.write(_record(2))
    (tmp_checkpoint_dir / "turn_0001.json").unlink()  # manifest still lists turn 1
    assert [r.turn for r in store.read_all()] == [2]
