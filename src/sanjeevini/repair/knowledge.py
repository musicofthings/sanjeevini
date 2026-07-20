"""Cross-run learning — a durable store of ``symptom → fix`` lessons.

Each resurrection that repairs something teaches a lesson: the error the agent
saw (the *symptom*) and the change that resolved it (the *fix*). The repair loop
records those lessons after a run; the agent retrieves the relevant ones on the
next run and injects them into its prompt. Over time the store makes each
resurrection smarter than the last — the "recursive learning" in the loop.

The store is a plain JSON file under the Sanjeevini cache. Relevance is a light,
dependency-free score: a framework match plus keyword overlap between the current
traceback and a lesson's symptom. That is enough to surface "last time a
TensorFlow 1.x tool hit a missing-AVX abort, a non-AVX build fixed it" exactly
when the agent is staring at that traceback.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from sanjeevini.pinners import cache_root

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Lines that look like the actual error, preferred when signing a traceback.
_ERROR_LINE_RE = re.compile(
    r"(error|exception|traceback|not found|no such|failed|illegal|abort|"
    r"cannot|undefined|missing|unresolved|incompatible)",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    """Return the set of lowercase alphanumeric tokens in ``text``."""
    return set(_TOKEN_RE.findall(text.lower()))


def error_signature(text: str, limit: int = 200) -> str:
    """Return a compact one-line signature of a traceback/error blob.

    Prefers the last line that looks like an actual error; falls back to the last
    non-empty line. Bounded to ``limit`` characters.

    Args:
        text: The stderr/traceback text.
        limit: Maximum length of the returned signature.

    Returns:
        A short signature string (possibly empty if ``text`` has no content).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if _ERROR_LINE_RE.search(line):
            return line[:limit]
    return lines[-1][:limit]


@dataclass
class Lesson:
    """One learned ``symptom → fix`` pair.

    Attributes:
        bug_class: Classification of the bug (e.g. ``dep_conflict``).
        symptom: Compact signature of the error that triggered the fix.
        fix: What resolved it (a description, or the first line of a patch).
        framework: Context the lesson came from (framework/language), for relevance.
        tool: Slug of the resurrection that produced the lesson.
        patch: The unified diff that fixed it, if any.
    """

    bug_class: str
    symptom: str
    fix: str
    framework: str = ""
    tool: str = ""
    patch: str = ""

    def key(self) -> tuple[str, str, str]:
        """Return the de-duplication key (class, symptom, fix)."""
        return (self.bug_class, self.symptom, self.fix)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict of this lesson."""
        return {
            "bug_class": self.bug_class,
            "symptom": self.symptom,
            "fix": self.fix,
            "framework": self.framework,
            "tool": self.tool,
            "patch": self.patch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Lesson:
        """Reconstruct a lesson from a decoded JSON dict (unknown keys ignored)."""
        fields = {"bug_class", "symptom", "fix", "framework", "tool", "patch"}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def as_hint(self) -> str:
        """Render the lesson as a one-line hint for the agent prompt."""
        where = f" [{self.framework}]" if self.framework else ""
        return f'{self.bug_class}{where}: on "{self.symptom[:120]}" → {self.fix[:160]}'


class KnowledgeStore:
    """A durable, append-only store of :class:`Lesson` records (one JSON file)."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the store at ``path`` and load existing lessons.

        Args:
            path: JSON file the lessons are persisted to.
        """
        self.path = Path(path)
        self._lessons: list[Lesson] = self._load()

    def _load(self) -> list[Lesson]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [Lesson.from_dict(d) for d in data if isinstance(d, dict)]

    def all(self) -> list[Lesson]:
        """Return every stored lesson."""
        return list(self._lessons)

    def add(self, lesson: Lesson) -> bool:
        """Add ``lesson`` (skipping duplicates) and persist. Returns whether added."""
        if not lesson.fix and not lesson.symptom:
            return False
        if any(existing.key() == lesson.key() for existing in self._lessons):
            return False
        self._lessons.append(lesson)
        self.save()
        return True

    def extend(self, lessons: list[Lesson]) -> int:
        """Add several lessons; return how many were newly stored."""
        added = sum(1 for lesson in lessons if self.add(lesson))
        return added

    def relevant(
        self, *, framework: str = "", error_text: str = "", top_k: int = 5
    ) -> list[Lesson]:
        """Return the lessons most relevant to the current context.

        Scores each lesson by a framework match plus keyword overlap between
        ``error_text`` and the lesson's symptom. Lessons with a zero score are
        dropped.

        Args:
            framework: The current tool's framework/language.
            error_text: The traceback the agent is currently reacting to.
            top_k: Maximum number of lessons to return.

        Returns:
            The highest-scoring lessons, best first.
        """
        query = _tokens(error_text)
        scored: list[tuple[float, Lesson]] = []
        for lesson in self._lessons:
            score = 0.0
            if framework and lesson.framework == framework:
                score += 1.0
            if query and lesson.symptom:
                score += len(query & _tokens(lesson.symptom))
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

    def save(self) -> None:
        """Persist all lessons atomically to :attr:`path`."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([lesson.to_dict() for lesson in self._lessons], indent=2)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)


def default_store() -> KnowledgeStore:
    """Return the shared knowledge store under the Sanjeevini cache root."""
    return KnowledgeStore(cache_root() / "knowledge.json")


def lessons_from_bugs(
    bugs_fixed: list[dict[str, str]], *, framework: str = "", tool: str = ""
) -> list[Lesson]:
    """Extract lessons from a run's ``bugs_fixed`` records.

    Args:
        bugs_fixed: Per-fix dicts (``class``/``description``/``patch``/``symptom``).
        framework: The resurrected tool's framework (context for the lesson).
        tool: The resurrected tool's slug.

    Returns:
        One :class:`Lesson` per fixed bug that carries a usable fix.
    """
    lessons: list[Lesson] = []
    for bug in bugs_fixed:
        fix = bug.get("description", "") or _first_patch_line(bug.get("patch", ""))
        if not fix:
            continue
        lessons.append(
            Lesson(
                bug_class=bug.get("class", "unknown"),
                symptom=bug.get("symptom", ""),
                fix=fix,
                framework=framework,
                tool=tool,
                patch=bug.get("patch", ""),
            )
        )
    return lessons


def _first_patch_line(patch: str) -> str:
    """Return the first meaningful line of a unified diff (for a fix summary)."""
    for line in patch.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("---", "+++", "@@", "diff ")):
            return stripped[:160]
    return ""
