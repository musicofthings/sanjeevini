"""Evidence-weighted inference of the genomic file type a tool *emits*.

A resurrection is only as meaningful as its sanity check, and a structural check
is only correct if it names the type the tool actually produces. Checking a
FASTQ-emitting read filter with ``samtools quickcheck`` is worse than useless:
it can pass for the wrong reason or fail for no reason, and either way PASS stops
meaning anything.

Naive keyword matching gets this wrong in a specific, common way. A tool's README
mentions many formats — the ones it consumes, the ones sibling tools produce, the
ones in a comparison table — and the first mention in some fixed scan order wins.
NanoFilt (which reads FASTQ and writes FASTQ) has "BAM" in its README, so a
first-match scan hands it a BAM check.

This module scores instead of scanning. Every mention of a format is weighted by
the words around it: a mention next to "writes"/"output"/"produces" is evidence
the tool emits that type; one next to "input"/"requires"/"takes" is evidence
against. The winner must also clear the runner-up by a margin — when the evidence
is genuinely ambiguous the caller gets ``None`` and should fall back to a
type-agnostic check. A weaker-but-true check beats a specific-but-wrong one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sanjeevini.contracts.schema import GenomicFileType

# Weights applied to one mention based on the words surrounding it.
_W_OUTPUT = 3.0  # "writes a BAM", "output.bam", "produces"
_W_NEUTRAL = 1.0  # a bare mention with no directional context
_W_INPUT = -2.0  # "takes a BAM", "input.bam" — evidence it is consumed, not emitted

# The winner must beat the runner-up by this much to be trusted. Below it the
# evidence is ambiguous and the caller should use a type-agnostic check.
_MARGIN = 2.0

# Direction is attributed per mention by *proximity* within its own sentence, not
# by presence anywhere nearby. "takes a fastq as input and writes a filtered
# fastq" must score its two mentions in opposite directions, which a
# presence-based test cannot do.
_OUTPUT_CONTEXT = re.compile(
    r"(\b(output|outputs|outputting|writes?|writing|written|produces?|produced|"
    r"producing|emits?|emitted|generates?|generated|generating|creates?|created|"
    r"returns?|saved?|saves|results?|reports?)\b|--out\b|(?<![\w-])-o\b|>)",
    re.IGNORECASE,
)
_INPUT_CONTEXT = re.compile(
    r"(\b(input|inputs|reads? in|reading|requires?|required|takes?|taking|accepts?|"
    r"accepted|given|supply|supplied|provide[ds]?|consumes?)\b|--in\b|(?<![\w-])-i\b)",
    re.IGNORECASE,
)

# Sentences are the scope for directional context; a marker in a neighbouring
# sentence says nothing about this mention.
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]")


@dataclass(frozen=True)
class OutputProfile:
    """One candidate output type, with how to name it and how to prove it.

    Two keyword sets, deliberately: ``aliases`` is permissive because loose words
    ("assembly", "report") are real evidence when *scoring* a README, but
    ``format_names`` is strict because identifying the type named by a sanity
    check must not fire on prose. A Nextflow check reading "…and the report
    contains ≥ 1 process" makes no claim about a ``.tsv`` file, and treating it
    as one would manufacture false "unsupported" warnings.

    Attributes:
        file_type: The genomic file type this profile describes.
        aliases: Words that suggest this type in prose, for scoring a corpus.
        format_names: Tokens that unambiguously name this format, for reading a
            sanity check's structural claim back out.
        extensions: Filename suffixes that prove the type on disk.
        check: A falsifiable structural sanity check for this type.
    """

    file_type: GenomicFileType
    aliases: tuple[str, ...]
    format_names: tuple[str, ...]
    extensions: tuple[str, ...]
    check: str


# Ordered only for stable tie-breaking; selection is by score, never position.
OUTPUT_PROFILES: tuple[OutputProfile, ...] = (
    OutputProfile(
        GenomicFileType.VCF,
        ("vcf", "variant call format", "gvcf"),
        ("vcf", "gvcf"),
        (".vcf", ".vcf.gz", ".bcf", ".gvcf"),
        "the VCF output parses with `bcftools stats` and contains ≥ 1 variant record",
    ),
    OutputProfile(
        GenomicFileType.BAM,
        ("bam", "cram", "alignment file"),
        ("bam", "cram"),
        (".bam", ".cram"),
        "the BAM output passes `samtools quickcheck` and contains ≥ 1 alignment",
    ),
    OutputProfile(
        GenomicFileType.FASTQ,
        ("fastq", "fastq.gz", "fq"),
        ("fastq", "fq"),
        (".fastq", ".fq", ".fastq.gz", ".fq.gz"),
        "the FASTQ output is valid 4-line-per-record FASTQ and contains ≥ 1 read",
    ),
    OutputProfile(
        GenomicFileType.FASTA,
        ("fasta", "fa", "assembly", "contigs"),
        ("fasta",),
        (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz"),
        "the FASTA output is valid and contains ≥ 1 sequence",
    ),
    OutputProfile(
        GenomicFileType.BED,
        ("bed", "bedmethyl", "intervals"),
        ("bed", "bedmethyl"),
        (".bed", ".bed.gz", ".bedmethyl"),
        "the BED output is valid 3+-column BED and contains ≥ 1 interval",
    ),
    OutputProfile(
        GenomicFileType.GTF,
        ("gtf", "gff", "gff3", "annotation file"),
        ("gtf", "gff", "gff3"),
        (".gtf", ".gff", ".gff3"),
        "the GTF/GFF output is valid 9-column format and contains ≥ 1 feature",
    ),
    OutputProfile(
        GenomicFileType.PAF,
        ("paf", "pairwise alignment format"),
        ("paf",),
        (".paf",),
        "the PAF output is valid 12+-column PAF and contains ≥ 1 alignment record",
    ),
    OutputProfile(
        GenomicFileType.GFA,
        ("gfa", "assembly graph"),
        ("gfa",),
        (".gfa",),
        "the GFA output is valid and contains ≥ 1 segment record",
    ),
    OutputProfile(
        GenomicFileType.H5AD,
        ("h5ad", "anndata"),
        ("h5ad", "anndata"),
        (".h5ad",),
        "the AnnData output opens with `anndata.read_h5ad` and has ≥ 1 observation",
    ),
    OutputProfile(
        GenomicFileType.JSON,
        ("json", "json report", "json summary"),
        ("json",),
        (".json",),
        "the JSON output parses and is a non-empty object",
    ),
    OutputProfile(
        GenomicFileType.TSV,
        ("tsv", "csv", "table", "summary table", "html report", "report"),
        ("tsv", "csv"),
        (".tsv", ".csv", ".txt", ".html"),
        "the TSV/CSV output is non-empty and contains ≥ 1 data row beyond the header",
    ),
)

# The check used when no type wins clearly — true of any tool, still falsifiable.
GENERIC_CHECK = (
    "The primary output file is produced, is non-empty and at least 10 KB, "
    "and parses without error."
)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Compile a word-boundary matcher for one alias (dots escaped, not wildcards)."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)


def _sentence_bounds(corpus: str, start: int, end: int) -> tuple[int, int]:
    """Return the offsets of the sentence containing ``corpus[start:end]``."""
    left = 0
    right = len(corpus)
    for match in _SENTENCE_SPLIT.finditer(corpus):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return left, right


def _nearest(pattern: re.Pattern[str], sentence: str, offset: int) -> int | None:
    """Return the character distance from ``offset`` to the nearest match, if any."""
    distances = [
        min(abs(match.start() - offset), abs(match.end() - offset))
        for match in pattern.finditer(sentence)
    ]
    return min(distances) if distances else None


def _mention_weight(corpus: str, start: int, end: int) -> float:
    """Weight one mention by the directional marker nearest it in its sentence.

    Proximity, not presence: in "takes a fastq as input and writes a filtered
    fastq", the first mention is nearest "takes"/"input" and the second nearest
    "writes", so they score in opposite directions — which is exactly right.

    Args:
        corpus: The full text being scored.
        start: Start offset of the mention.
        end: End offset of the mention.

    Returns:
        :data:`_W_OUTPUT`, :data:`_W_INPUT`, or :data:`_W_NEUTRAL`.
    """
    left, right = _sentence_bounds(corpus, start, end)
    sentence = corpus[left:right]
    offset = start - left
    out_dist = _nearest(_OUTPUT_CONTEXT, sentence, offset)
    in_dist = _nearest(_INPUT_CONTEXT, sentence, offset)
    if out_dist is None and in_dist is None:
        return _W_NEUTRAL
    if in_dist is None:
        return _W_OUTPUT
    if out_dist is None:
        return _W_INPUT
    if out_dist == in_dist:
        return _W_NEUTRAL
    return _W_OUTPUT if out_dist < in_dist else _W_INPUT


def _mentions(corpus: str, aliases: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return non-overlapping mention spans for ``aliases``, longest alias first.

    Overlaps are dropped so ``fastq.gz`` is not also counted as ``fastq``.
    """
    spans: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        for match in _alias_pattern(alias).finditer(corpus):
            if any(match.start() < end and start < match.end() for start, end in spans):
                continue
            spans.append((match.start(), match.end()))
    return spans


def score_output_types(corpus: str) -> dict[GenomicFileType, float]:
    """Score every candidate output type by weighted mentions in ``corpus``.

    Args:
        corpus: README, docs, and paper-abstract text describing the tool.

    Returns:
        A mapping of file type to score, including only types scoring above zero.
    """
    scores: dict[GenomicFileType, float] = {}
    for profile in OUTPUT_PROFILES:
        total = sum(
            _mention_weight(corpus, start, end) for start, end in _mentions(corpus, profile.aliases)
        )
        if total > 0:
            scores[profile.file_type] = total
    return scores


def infer_output_type(corpus: str) -> OutputProfile | None:
    """Return the type the tool most likely emits, or ``None`` if ambiguous.

    ``None`` is a deliberate, useful answer: it tells the caller to fall back to
    a type-agnostic check rather than assert a structural claim the evidence does
    not support.

    Args:
        corpus: README, docs, and paper-abstract text describing the tool.

    Returns:
        The winning :class:`OutputProfile`, or ``None`` when no type clears the
        runner-up by :data:`_MARGIN`.
    """
    scores = score_output_types(corpus)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < _MARGIN:
        return None
    return profile_for(ranked[0][0])


def profile_for(file_type: GenomicFileType) -> OutputProfile | None:
    """Return the profile describing ``file_type``, or ``None`` if untracked."""
    for profile in OUTPUT_PROFILES:
        if profile.file_type is file_type:
            return profile
    return None


def extensions_for_check(sanity_check: str) -> tuple[str, ...]:
    """Return the file extensions a sanity check's claimed type would produce.

    Lets the repair loop test a check's structural claim against the files a run
    actually produced. Matches on :attr:`OutputProfile.format_names` only — a
    check must *name a format* to be verified against one. A check that names no
    tracked type returns ``()``, which the caller should read as "nothing to
    verify", not "verification failed".

    Args:
        sanity_check: The sanity-check text from the plan.

    Returns:
        The matching profile's extensions, or ``()`` when no format is named.
    """
    for profile in OUTPUT_PROFILES:
        for name in profile.format_names:
            if _alias_pattern(name).search(sanity_check):
                return profile.extensions
    return ()
