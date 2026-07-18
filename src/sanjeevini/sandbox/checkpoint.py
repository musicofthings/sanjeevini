"""Turn-level checkpoint persistence for the repair loop.

A :class:`CheckpointStore` is the durable memory of a resurrection run. After
each turn the loop (or the :class:`~sanjeevini.sandbox.docker_sandbox.DockerSandbox`)
appends a :class:`TurnRecord` describing what was executed, what came back, and
which Docker snapshot banks the result. On resume, the store replays those
records so the loop can restart from the last known-good state instead of from
turn zero.

The store is deliberately decoupled from Docker: the repair loop also
checkpoints its own reasoning state (applied patches, traceback history) which
has nothing to do with container exec results. Records are stored one JSON file
per turn (``turn_{turn:04d}.json``) plus a ``manifest.json`` index for fast
enumeration, and every write is atomic (``.tmp`` then :func:`os.replace`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STDOUT_LIMIT = 8 * 1024
"""Maximum stored stdout size, in bytes (8 KB)."""

STDERR_LIMIT = 4 * 1024
"""Maximum stored stderr size, in bytes (4 KB)."""

_MANIFEST = "manifest.json"


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` clipped to ``limit`` bytes (UTF-8 safe).

    Args:
        text: The string to bound.
        limit: Maximum size in bytes.

    Returns:
        ``text`` unchanged if it already fits, otherwise the longest prefix
        that fits within ``limit`` bytes with a truncation marker appended.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    clipped = encoded[:limit].decode("utf-8", "ignore")
    return clipped + "\n…[truncated]"


@dataclass
class TurnRecord:
    """A single turn of a resurrection run.

    Attributes:
        turn: 1-based turn number; determines file ordering.
        cmd: The command executed inside the sandbox, as an argv list.
        returncode: Process exit code (0 == success).
        stdout: Captured stdout, truncated to :data:`STDOUT_LIMIT` bytes.
        stderr: Captured stderr, truncated to :data:`STDERR_LIMIT` bytes.
        duration_s: Wall-clock duration of the command, in seconds.
        snapshot_tag: Docker image tag banking this turn, or ``None`` if the
            turn was not snapshotted (typically because it failed).
        patches_applied: Unified-diff strings applied by the loop this turn.
        timestamp: ISO 8601 timestamp of when the record was created.
        cost_usd: Optional agent cost accrued this turn, in USD.
    """

    turn: int
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    snapshot_tag: str | None = None
    patches_applied: list[str] = field(default_factory=list)
    timestamp: str = ""
    cost_usd: float | None = None

    @property
    def ok(self) -> bool:
        """Whether the command in this turn succeeded (exit code 0)."""
        return self.returncode == 0

    def truncated(self) -> TurnRecord:
        """Return a copy with stdout/stderr clipped to the byte limits.

        Returns:
            A new :class:`TurnRecord` whose ``stdout`` and ``stderr`` fit
            within :data:`STDOUT_LIMIT` and :data:`STDERR_LIMIT`.
        """
        return TurnRecord(
            turn=self.turn,
            cmd=list(self.cmd),
            returncode=self.returncode,
            stdout=_truncate(self.stdout, STDOUT_LIMIT),
            stderr=_truncate(self.stderr, STDERR_LIMIT),
            duration_s=self.duration_s,
            snapshot_tag=self.snapshot_tag,
            patches_applied=list(self.patches_applied),
            timestamp=self.timestamp,
            cost_usd=self.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        """Reconstruct a :class:`TurnRecord` from a decoded JSON dict.

        Args:
            data: A mapping produced by :meth:`to_dict`. Unknown keys are
                ignored so records written by newer versions still load.

        Returns:
            The reconstructed record.
        """
        fields = {
            "turn",
            "cmd",
            "returncode",
            "stdout",
            "stderr",
            "duration_s",
            "snapshot_tag",
            "patches_applied",
            "timestamp",
            "cost_usd",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})


class CheckpointStore:
    """Append-and-replay store of :class:`TurnRecord` files in a directory.

    Each record is written to ``directory/turn_{turn:04d}.json`` atomically, and
    a ``manifest.json`` index tracks the set of turn numbers for fast
    enumeration without globbing.
    """

    def __init__(self, directory: Path) -> None:
        """Open (or create) a checkpoint store rooted at ``directory``.

        Args:
            directory: Filesystem directory holding the turn records. Created
                if it does not exist.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def _manifest_path(self) -> Path:
        return self.directory / _MANIFEST

    def _turn_path(self, turn: int) -> Path:
        return self.directory / f"turn_{turn:04d}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        """Write ``payload`` to ``path`` via a temp file and :func:`os.replace`.

        Args:
            path: Destination file.
            payload: Text to write.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    def _read_manifest(self) -> list[int]:
        """Return the sorted turn numbers recorded in the manifest.

        Falls back to globbing ``turn_*.json`` if the manifest is missing or
        corrupt, so the store still works after a partial write.
        """
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            turns = data.get("turns", [])
            return sorted(int(t) for t in turns)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
            turns = []
            for p in self.directory.glob("turn_*.json"):
                try:
                    turns.append(int(p.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            return sorted(turns)

    def _write_manifest(self, turns: list[int]) -> None:
        payload = json.dumps({"turns": sorted(set(turns))}, indent=2)
        self._atomic_write(self._manifest_path, payload)

    def write(self, record: TurnRecord) -> None:
        """Persist ``record`` and update the manifest.

        stdout/stderr are clipped to the byte limits before writing so no single
        record file grows unbounded.

        Args:
            record: The turn record to store. Its ``turn`` field determines the
                filename; writing the same turn twice overwrites the earlier
                file.
        """
        clipped = record.truncated()
        payload = json.dumps(clipped.to_dict(), indent=2)
        self._atomic_write(self._turn_path(record.turn), payload)
        turns = self._read_manifest()
        if record.turn not in turns:
            turns.append(record.turn)
        self._write_manifest(turns)

    def read_all(self) -> list[TurnRecord]:
        """Return every stored record, ordered by turn number.

        Returns:
            The records sorted ascending by ``turn``. Missing files listed in
            the manifest are skipped rather than raising.
        """
        records: list[TurnRecord] = []
        for turn in self._read_manifest():
            path = self._turn_path(turn)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            records.append(TurnRecord.from_dict(data))
        records.sort(key=lambda r: r.turn)
        return records

    def latest(self) -> TurnRecord | None:
        """Return the highest-numbered record, or ``None`` if the store is empty."""
        records = self.read_all()
        return records[-1] if records else None

    def last_successful_snapshot(self) -> str | None:
        """Return the snapshot tag of the most recent successful turn.

        Returns:
            The ``snapshot_tag`` of the highest-numbered record that both
            succeeded and was snapshotted, or ``None`` if no such turn exists.
        """
        for record in reversed(self.read_all()):
            if record.ok and record.snapshot_tag:
                return record.snapshot_tag
        return None

    def next_turn(self) -> int:
        """Return the turn number a new record should use.

        Returns:
            One greater than the highest stored turn, or 1 for an empty store.
        """
        turns = self._read_manifest()
        return (turns[-1] + 1) if turns else 1

    def cost_usd(self) -> float:
        """Return the total agent cost across all recorded turns, in USD.

        Records without a ``cost_usd`` value contribute zero.

        Returns:
            The summed cost.
        """
        return sum(r.cost_usd for r in self.read_all() if r.cost_usd is not None)
