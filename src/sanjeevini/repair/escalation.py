"""Bounded self-escalation — retrying a failed run on a base image that can work.

Some resurrections cannot be repaired from inside the container they were given.
If the Scout picked ``python:3.11-slim`` for a tool whose sources are Python 2,
no amount of patching inside that container reaches a PASS: the interpreter is
wrong. A human watching the run would say "try 2.7" and start again. This module
lets Jeeva say it to itself.

Escalation is deliberately **evidence-driven and bounded**:

* A retry is proposed only when a rule matches something the run actually
  printed. There is no blind "try the next image" fallback — a run that failed
  for a reason the image cannot fix is left failed, which is the honest verdict.
* Each rule names the transform it applies to the *current* image, so the choice
  is auditable after the fact rather than a model's guess.
* An image is never tried twice, and the caller caps how many extra attempts are
  allowed.

Deliberately *not* a rule: bumping an end-of-life Debian codename when apt
mirrors 404. That decay is real, but the agent repairs it in place by repointing
``sources.list`` at ``archive.debian.org`` — and bumping the distro under a
pinned old interpreter usually destroys a working environment to fix a
one-command problem.

This module depends on :mod:`sanjeevini.repair.loop` for types only; the runtime
dependency runs the other way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sanjeevini.repair.loop import RepairOutcome

# Any of these in a run's blockers means the sources are Python 2: either the
# parser said so outright, or the code imported a module that only ever existed
# in the Python 2 standard library.
_PY2_MARKERS = re.compile(
    r"missing parentheses in call to"
    r"|no module named ['\"]?(?:urllib2|configparser2|cpickle|cstringio|stringio"
    r"|tkinter\.|queue\b|commands|httplib|htmlparser|socketserver|copy_reg|"
    r"urlparse|cprofile2|thread\b)"
    r"|except\s+\w+\s*,\s*\w+\s*:",
    re.IGNORECASE,
)

# The image has no C toolchain (or no headers to compile against). Distinct from
# a musl problem: here the compiler is simply absent.
_TOOLCHAIN_MARKERS = re.compile(
    r"(?:^|[^\w-])(?:gcc|cc|g\+\+|clang|make|ld)(?::| :)?\s*(?:command )?not found"
    r"|unable to execute '[^']*(?:gcc|cc|g\+\+)"
    r"|command '[^']*(?:gcc|cc|g\+\+)[^']*' failed"
    r"|fatal error:\s*\S+\.h: no such file"
    r"|microsoft visual c\+\+ .{0,40}required",
    re.IGNORECASE,
)

# Alpine's musl libc rejects the manylinux wheels most scientific packages ship,
# so every dependency falls back to a source build that then fails.
_MUSL_MARKERS = re.compile(
    r"musl"
    r"|manylinux"
    r"|failed building wheel"
    r"|undefined symbol"
    r"|error: could not build wheels",
    re.IGNORECASE,
)

_PY3_IMAGE = re.compile(r"^(?P<repo>(?:[\w.\-/]+/)?python):3(?:\.\d+)?(?P<variant>[\w.\-]*)$")
_VARIANT_SUFFIX = re.compile(r"-(?:slim|alpine)(?=-|$)")

# A repair loop that died because the agent itself could not be reached (API
# outage, exhausted credits) learned nothing about the image. Retrying on a
# different one just spends more of whatever is left.
_UNESCALATABLE = ("agent call failed",)


@dataclass(frozen=True)
class EscalationStep:
    """One proposed retry on a different base image.

    Attributes:
        base_image: The image to try next.
        rule: Name of the rule that fired, for the provenance record.
        rationale: Human-readable justification, shown on the CLI.
        signal: The blocker line that triggered the rule — the evidence.
    """

    base_image: str
    rule: str
    rationale: str
    signal: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict for the provenance record."""
        return {
            "base_image": self.base_image,
            "rule": self.rule,
            "rationale": self.rationale,
            "signal": self.signal,
        }


def _to_python2(image: str) -> str | None:
    """Retarget a ``python:3.x`` image at Python 2.7, preserving slimness."""
    match = _PY3_IMAGE.match(image)
    if match is None:
        # Only a python:3* image can be safely retargeted. A conda, ubuntu, or
        # vendor image carries an environment that swapping the tag would
        # silently discard, and we would have no idea what we broke.
        return None
    # 2.7 predates the codename-suffixed tags, so `-slim-bookworm` has no 2.7
    # counterpart. Carry over only whether it was slim; anything else would name
    # a tag that does not exist and turn a repairable run into a pull failure.
    variant = "-slim" if "slim" in match["variant"] else ""
    return f"{match['repo']}:2.7{variant}"


def _drop_variant(image: str) -> str | None:
    """Trade a ``-slim``/``-alpine`` variant for the full image, which ships gcc."""
    widened = _VARIANT_SUFFIX.sub("", image, count=1)
    return widened if widened != image else None


@dataclass(frozen=True)
class _Rule:
    """A signal-to-image-transform rule. Order in :data:`_RULES` is priority."""

    name: str
    markers: re.Pattern[str]
    retarget: Callable[[str], str | None]
    rationale: str
    requires_alpine: bool = False


_RULES: tuple[_Rule, ...] = (
    # Highest priority: a wrong interpreter cannot be patched around, and it
    # also *causes* spurious toolchain errors when setup.py fails to parse.
    _Rule(
        name="python2_sources",
        markers=_PY2_MARKERS,
        retarget=_to_python2,
        rationale="the sources are Python 2; no repair inside a Python 3 image can reach a PASS",
    ),
    _Rule(
        name="musl_incompatible",
        markers=_MUSL_MARKERS,
        retarget=_drop_variant,
        rationale="Alpine's musl libc rejects manylinux wheels, forcing every dependency "
        "to build from source",
        requires_alpine=True,
    ),
    _Rule(
        name="missing_toolchain",
        markers=_TOOLCHAIN_MARKERS,
        retarget=_drop_variant,
        rationale="the image has no C toolchain and the build needs one",
    ),
)


def propose_escalation(
    *,
    base_image: str,
    verdict: str,
    reason: str,
    blockers: Sequence[str],
    tried: Iterable[str] = (),
) -> EscalationStep | None:
    """Propose the next base image to try, or ``None`` to accept the failure.

    Args:
        base_image: The image the failed attempt ran on.
        verdict: That attempt's verdict; ``PASS`` never escalates.
        reason: The attempt's failure reason, used to skip infrastructure faults.
        blockers: Error signatures the attempt collected, newest last.
        tried: Images already attempted, so none is repeated.

    Returns:
        The highest-priority :class:`EscalationStep` whose rule matched and whose
        target has not been tried, or ``None`` if nothing justifies a retry.
    """
    # A PASS has nothing to escalate; a BUDGET stop means the cost cap is spent,
    # so trying another image would only breach the cap the caller asked us to honour.
    if verdict in ("PASS", "BUDGET"):
        return None
    if any(marker in reason for marker in _UNESCALATABLE):
        return None

    seen = {base_image, *tried}
    for rule in _RULES:
        if rule.requires_alpine and "alpine" not in base_image:
            continue
        signal = next((b for b in reversed(blockers) if rule.markers.search(b)), None)
        if signal is None:
            continue
        target = rule.retarget(base_image)
        if target is None or target in seen:
            continue
        return EscalationStep(
            base_image=target,
            rule=rule.name,
            rationale=rule.rationale,
            signal=signal,
        )
    return None


@dataclass
class AttemptRecord:
    """What one attempt did and why the run moved on from it.

    The escalation fields describe the move *away* from this image rather than
    the move towards it. Written the other way round, the reason a run left an
    image would live on the record of the image it left for — which is the one
    record a contract emitted mid-escalation does not yet have, so the
    justification would never reach the provenance file.

    Attributes:
        base_image: The image this attempt ran on.
        verdict: Its terminal verdict.
        turns: Turns it consumed.
        reason: Its failure reason, if any.
        escalated_to: The image tried next, if this attempt was escalated away from.
        rule: The rule that justified that move.
        rationale: Why that rule fired.
        signal: The blocker line the rule matched — the evidence.
        cost_usd: Agent cost this attempt accrued, so a caller enforcing a global
            budget can subtract it from the remaining allowance for later attempts.
    """

    base_image: str
    verdict: str
    turns: int
    reason: str = ""
    escalated_to: str = ""
    rule: str = ""
    rationale: str = ""
    signal: str = ""
    cost_usd: float = 0.0

    def record_escalation(self, step: EscalationStep) -> None:
        """Note that the run left this image for ``step``'s image, and why."""
        self.escalated_to = step.base_image
        self.rule = step.rule
        self.rationale = step.rationale
        self.signal = step.signal

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict for the provenance record."""
        return {
            "base_image": self.base_image,
            "verdict": self.verdict,
            "turns": self.turns,
            "reason": self.reason,
            "escalated_to": self.escalated_to,
            "rule": self.rule,
            "rationale": self.rationale,
            "signal": self.signal,
            "cost_usd": self.cost_usd,
        }


@dataclass
class EscalatingResurrection:
    """Runs attempts on successive base images until one passes or the budget ends.

    The caller supplies ``run_attempt``, which owns everything image-specific —
    starting a sandbox, seeding the repo, driving the loop. This class owns only
    the decision of whether to try again and on what. That split is what makes
    escalation testable without Docker.

    Attributes:
        base_image: The image the first attempt uses.
        run_attempt: Callable taking the image and the trail so far, returning
            that attempt's :class:`~sanjeevini.repair.loop.RepairOutcome`.
        max_extra_attempts: How many retries are permitted beyond the first.
        announce: Called with each escalation decision, for CLI output.
    """

    base_image: str
    run_attempt: Callable[[str, list[AttemptRecord]], RepairOutcome]
    max_extra_attempts: int = 1
    announce: Callable[[str], None] = lambda _msg: None
    attempts: list[AttemptRecord] = field(default_factory=list)

    def run(self) -> RepairOutcome:
        """Run attempts until PASS, no justified escalation, or budget exhausted.

        Returns:
            The final attempt's outcome — the PASS if one was reached, otherwise
            the last failure. Every attempt is recorded in :attr:`attempts`.
        """
        image = self.base_image
        tried: list[str] = []

        while True:
            outcome = self.run_attempt(image, list(self.attempts))
            record = AttemptRecord(
                base_image=image,
                verdict=outcome.verdict,
                turns=outcome.turns,
                reason=outcome.reason,
                cost_usd=outcome.cost_usd,
            )
            self.attempts.append(record)
            tried.append(image)

            if len(tried) > self.max_extra_attempts:
                return outcome
            step = propose_escalation(
                base_image=image,
                verdict=outcome.verdict,
                reason=outcome.reason,
                blockers=outcome.blockers,
                tried=tried,
            )
            if step is None:
                return outcome
            record.record_escalation(step)
            self.announce(
                f"escalating to {step.base_image}: {step.rationale} (saw: {step.signal[:120]})"
            )
            image = step.base_image
