"""Tests for evidence-weighted output-type inference — the sanity-check quality
gate's static half."""

from __future__ import annotations

from sanjeevini.contracts.output_type import (
    GENERIC_CHECK,
    extensions_for_check,
    infer_output_type,
    output_type_for_check,
    profile_for,
    score_output_types,
)
from sanjeevini.contracts.schema import GenomicFileType
from sanjeevini.scouts.python_scout import generate_sanity_check
from sanjeevini.scouts.repo import RepoSnapshot


def _snapshot(readme: str) -> RepoSnapshot:
    snap = RepoSnapshot(url="https://github.com/acme/tool", owner="acme", name="tool")
    snap.files["README.md"] = readme
    return snap


# ---------------------------------------------------------------------------
# The regression this gate exists for
# ---------------------------------------------------------------------------

NANOFILT_README = """
# NanoFilt

Filtering and trimming of Oxford Nanopore sequencing data.

NanoFilt takes a fastq file as input and writes a filtered fastq file to stdout.
Reads are filtered on quality and length.

Usage:
    NanoFilt -q 10 -l 500 input.fastq > filtered.fastq
    gunzip -c reads.fastq.gz | NanoFilt -q 12 > trimmed.fastq

For alignment-based filtering of a bam file, see the companion tool instead.
"""


def test_nanofilt_gets_a_fastq_check_not_a_bam_one() -> None:
    # The bug this module was written for: a single incidental "bam" mention
    # used to win over the fastq the tool actually emits.
    profile = infer_output_type(NANOFILT_README)
    assert profile is not None
    assert profile.file_type is GenomicFileType.FASTQ

    check = generate_sanity_check(_snapshot(NANOFILT_README))
    assert "FASTQ" in check
    assert "BAM" not in check and "samtools" not in check


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_output_context_outweighs_a_bare_mention() -> None:
    corpus = "The tool writes a VCF file. Also mentions bam somewhere unrelated."
    scores = score_output_types(corpus)
    assert scores[GenomicFileType.VCF] > scores[GenomicFileType.BAM]


def test_input_context_scores_negatively() -> None:
    corpus = "The tool requires a bam file as input."
    assert GenomicFileType.BAM not in score_output_types(corpus)


def test_a_type_that_is_both_input_and_output_is_neutral_not_negative() -> None:
    corpus = "Takes a bam as input and writes a bam as output."
    assert score_output_types(corpus).get(GenomicFileType.BAM, 0) > 0


def test_aliases_match_on_word_boundaries_only() -> None:
    # "bambusa" and "sample" must not count as bam/sam mentions.
    assert score_output_types("bambusa sampled the fasta output") == {GenomicFileType.FASTA: 3.0}


def test_repeated_output_mentions_accumulate() -> None:
    once = score_output_types("writes a vcf")
    twice = score_output_types("writes a vcf; the output vcf is sorted")
    assert twice[GenomicFileType.VCF] > once[GenomicFileType.VCF]


# ---------------------------------------------------------------------------
# Ambiguity → decline to guess
# ---------------------------------------------------------------------------


def test_ambiguous_evidence_returns_none() -> None:
    corpus = "Writes a vcf output. Writes a bam output."
    assert infer_output_type(corpus) is None


def test_ambiguity_falls_back_to_the_generic_check() -> None:
    check = generate_sanity_check(_snapshot("Writes a vcf output. Writes a bam output."))
    assert check == GENERIC_CHECK


def test_no_format_mentions_returns_none() -> None:
    assert infer_output_type("A tool that does something unspecified.") is None


def test_a_clear_winner_beats_the_margin() -> None:
    corpus = "Writes a vcf output. The output vcf is indexed. Reads a bam as input."
    profile = infer_output_type(corpus)
    assert profile is not None and profile.file_type is GenomicFileType.VCF


# ---------------------------------------------------------------------------
# Precedence inside the scout
# ---------------------------------------------------------------------------


def test_a_benchmark_threshold_still_outranks_type_inference() -> None:
    readme = "Achieves an F1 score of 0.92. Writes a vcf output file."
    check = generate_sanity_check(_snapshot(readme))
    assert "F1" in check and "0.87" in check


def test_structure_checks_survive_for_non_genomic_tools() -> None:
    check = generate_sanity_check(_snapshot("Predicts a protein structure from sequence."))
    assert "structure" in check.lower()


# ---------------------------------------------------------------------------
# extensions_for_check (the bridge to empirical verification)
# ---------------------------------------------------------------------------


def test_extensions_for_a_typed_check() -> None:
    assert ".bam" in extensions_for_check("the BAM output passes samtools quickcheck")


def test_extensions_for_an_untyped_check_are_empty() -> None:
    assert extensions_for_check(GENERIC_CHECK) == ()


def test_output_type_for_a_typed_check() -> None:
    assert output_type_for_check("the VCF output contains ≥ 1 variant") is GenomicFileType.VCF
    assert output_type_for_check("the BAM passes samtools quickcheck") is GenomicFileType.BAM


def test_output_type_for_an_untyped_check_is_none() -> None:
    assert output_type_for_check(GENERIC_CHECK) is None


def test_profile_for_a_tracked_type() -> None:
    profile = profile_for(GenomicFileType.FASTQ)
    assert profile is not None and ".fastq" in profile.extensions


def test_profile_for_an_untracked_type_is_none() -> None:
    assert profile_for(GenomicFileType.PICKLE) is None


def test_every_generated_check_identifies_its_own_type() -> None:
    # The invariant tying the static and empirical halves together: a check this
    # module generates must be readable back as the type it was generated for,
    # or the loop would probe for the wrong extensions (or none at all).
    from sanjeevini.contracts.output_type import OUTPUT_PROFILES

    for profile in OUTPUT_PROFILES:
        assert extensions_for_check(profile.check) == profile.extensions, profile.file_type


def test_prose_words_do_not_identify_a_format() -> None:
    # "report" scores as TSV evidence but must not make a Nextflow check look
    # like a claim about a .tsv file — that would manufacture false warnings.
    assert extensions_for_check("nextflow run exits 0 and the report has ≥ 1 process") == ()
    assert extensions_for_check("the assembly completes with ≥ 1 contig") == ()


def test_workflow_and_r_checks_claim_no_file_type() -> None:
    for check in (
        "snakemake --dryrun exits 0 and resolves ≥ 1 job",
        "cwltool exits 0 on the bundled test inputs.",
        "The package installs, loads with library(), and its example() runs with 0 errors.",
    ):
        assert extensions_for_check(check) == ()


def test_every_profile_check_is_falsifiable() -> None:
    from sanjeevini.contracts.output_type import OUTPUT_PROFILES
    from sanjeevini.scouts.python_scout import is_falsifiable

    for profile in OUTPUT_PROFILES:
        assert is_falsifiable(profile.check), profile.file_type
    assert is_falsifiable(GENERIC_CHECK)
