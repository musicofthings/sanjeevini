"""Typed I/O schema for Sanjeevini integration contracts.

Every revived tool emits a contract with inputs/outputs drawn from this
vocabulary.  Compose uses ContractSchema.compatible_with() to validate
port compatibility at *pipeline-build time* rather than at runtime — the
same principle as type-checking before execution.

Long-read-specific types (POD5, FAST5, BLOW5, BAM_HIFI, MODBAM, etc.) are
first-class citizens here, not afterthoughts.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Vocabulary enums
# ---------------------------------------------------------------------------

class GenomicFileType(str, Enum):
    # ── Reads / alignments ──────────────────────────────────────────────────
    FASTQ        = "fastq"
    FASTQ_GZ     = "fastq.gz"
    BAM          = "bam"          # generic aligned/unaligned BAM
    CRAM         = "cram"
    SAM          = "sam"
    # ── Long-read raw signal ─────────────────────────────────────────────────
    POD5         = "pod5"         # ONT current-generation signal format
    FAST5        = "fast5"        # ONT legacy signal format
    BLOW5        = "blow5"        # SLOW5/BLOW5 (ONT archival / HPC-friendly)
    BAM_HIFI     = "bam.hifi"     # PacBio HiFi / CCS — unaligned BAM from instrument
    SUBREADS_BAM = "subreads.bam" # PacBio CLR subread BAM
    SUBREADS_H5  = "subreads.h5"  # PacBio legacy HDF5 (RS II era)
    # ── Variants ─────────────────────────────────────────────────────────────
    VCF          = "vcf"
    VCF_GZ       = "vcf.gz"
    BCF          = "bcf"
    GVCF         = "gvcf"
    MAF          = "maf"          # Mutation Annotation Format (somatic)
    # ── Intervals / annotation ───────────────────────────────────────────────
    BED          = "bed"
    BED_GZ       = "bed.gz"
    GTF          = "gtf"
    GFF3         = "gff3"
    # ── Reference / sequence ─────────────────────────────────────────────────
    FASTA        = "fasta"
    FASTA_GZ     = "fasta.gz"
    FASTA_IDX    = "fasta.fai"    # samtools fai index
    # ── Modified bases / methylation ─────────────────────────────────────────
    MODBAM       = "modbam"       # BAM with MM/ML tags (ONT modkit / modkit2)
    BEDMETHYL    = "bedmethyl"    # CpG methylation BED (modkit, bismark)
    # ── Alignment formats ────────────────────────────────────────────────────
    PAF          = "paf"          # Pairwise Alignment Format (minimap2)
    GFA          = "gfa"          # Graphical Fragment Assembly (hifiasm, etc.)
    # ── Count matrices / single-cell ────────────────────────────────────────
    H5AD         = "h5ad"         # AnnData
    HDF5         = "hdf5"
    MTX          = "mtx"          # 10x MEX sparse matrix
    # ── Generic ──────────────────────────────────────────────────────────────
    CSV          = "csv"
    TSV          = "tsv"
    JSON         = "json"
    PICKLE       = "pickle"
    DIRECTORY    = "directory"    # tool emits / consumes a whole directory
    ANY          = "any"          # wildcard — use sparingly


class ReferenceGenome(str, Enum):
    HG19    = "hg19"        # GRCh37
    HG38    = "hg38"        # GRCh38
    CHM13   = "chm13"       # T2T-CHM13v2.0
    MM10    = "mm10"        # GRCm38
    MM39    = "mm39"        # GRCm39
    RNOR6   = "rnor6"       # Rat Rnor_6.0
    DANRER11= "danrerio11"  # Zebrafish GRCz11
    UNKNOWN = "unknown"


class SequencingPlatform(str, Enum):
    ILLUMINA      = "illumina"
    ONT           = "ont"           # Oxford Nanopore (any chemistry)
    PACBIO_HIFI   = "pacbio_hifi"   # PacBio HiFi / CCS (Sequel IIe, Revio)
    PACBIO_CLR    = "pacbio_clr"    # PacBio CLR (Sequel / RS II) — legacy
    ELEMENT       = "element"       # Element Biosciences AVITI
    ULTIMA        = "ultima"        # Ultima Genomics UG 100
    ANY           = "any"


# ---------------------------------------------------------------------------
# Model-bundle reference (Dorado, medaka, Clair3, DeepVariant, pb-CpG-tools)
# ---------------------------------------------------------------------------

class ModelBundleRef(BaseModel):
    """Pointer to an external model bundle required by the tool.

    Model bundles are NOT shipped in Docker images for these tools — they must
    be downloaded separately.  The contract emitter records the canonical
    source_url so Jeeva can fetch and cache the right one.
    """
    tool: str           # e.g. "dorado", "medaka", "clair3", "deepvariant"
    version: str        # tool version the model was released for, e.g. "0.7.2"
    model_name: str     # canonical name, e.g. "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"
    source_url: str     # direct download URL (not a landing page)
    sha256: str | None = None     # for verification after download
    size_gb: float | None = None
    gpu_required: bool = False
    chemistry: str | None = None  # e.g. "r10.4.1_e8.2", "r941_min"


# ---------------------------------------------------------------------------
# I/O ports
# ---------------------------------------------------------------------------

class IOPort(BaseModel):
    """A single typed input or output port of a revived tool."""
    name: str
    type: GenomicFileType
    description: str = ""
    optional: bool = False
    multiple: bool = False              # port accepts/emits a list of files
    reference: ReferenceGenome | None = None
    platform: SequencingPlatform = SequencingPlatform.ANY
    # For tools that require a model bundle on this port
    model_bundle: ModelBundleRef | None = None


# ---------------------------------------------------------------------------
# Contract schema
# ---------------------------------------------------------------------------

class ContractSchema(BaseModel):
    """Full typed I/O schema embedded in every Sanjeevini integration contract.

    A contract schema is emitted by the Contract Emitter organ and consumed
    by Compose to validate port compatibility before any containers start.
    """
    schema_version: Literal["1.0"] = "1.0"
    inputs:  list[IOPort] = Field(default_factory=list)
    outputs: list[IOPort] = Field(default_factory=list)

    # Hardware / environment requirements
    platform:     SequencingPlatform = SequencingPlatform.ANY
    reference:    ReferenceGenome | None = None
    gpu_required: bool = False
    min_ram_gb:   float | None = None
    min_cores:    int | None = None

    # Workflow type — set by the Workflow Scout, not the Python Scout
    workflow_type: Literal["python", "r", "nextflow", "snakemake", "wdl", "cwl", "binary"] = "python"

    @model_validator(mode="after")
    def _no_duplicate_port_names(self) -> "ContractSchema":
        in_names  = [p.name for p in self.inputs]
        out_names = [p.name for p in self.outputs]
        if len(in_names) != len(set(in_names)):
            raise ValueError("Duplicate input port names")
        if len(out_names) != len(set(out_names)):
            raise ValueError("Duplicate output port names")
        return self

    # ------------------------------------------------------------------
    # Compatibility check (used by Compose at pipeline-build time)
    # ------------------------------------------------------------------

    def compatible_with(
        self,
        downstream: "ContractSchema",
        port_map: dict[str, str],
    ) -> list[str]:
        """Check that each (my_output_port → their_input_port) mapping is type-safe.

        Parameters
        ----------
        downstream:
            The ContractSchema of the next step in the pipeline.
        port_map:
            Mapping from this contract's output port name to the downstream
            contract's input port name.  E.g. {"bam_out": "bam_in"}.

        Returns
        -------
        list[str]
            Empty list if all ports are compatible; otherwise a list of
            human-readable error messages.
        """
        my_outputs   = {p.name: p for p in self.outputs}
        their_inputs = {p.name: p for p in downstream.inputs}
        errors: list[str] = []

        for my_name, their_name in port_map.items():
            src = my_outputs.get(my_name)
            dst = their_inputs.get(their_name)

            if src is None:
                errors.append(f"Output port '{my_name}' not found in source contract")
                continue
            if dst is None:
                errors.append(f"Input port '{their_name}' not found in target contract")
                continue

            # Type check — ANY is a wildcard on the destination side
            if dst.type not in (src.type, GenomicFileType.ANY):
                errors.append(
                    f"Type mismatch on {my_name}→{their_name}: "
                    f"{src.type} cannot flow into {dst.type}"
                )

            # Reference genome check
            if (src.reference and dst.reference and src.reference != dst.reference):
                errors.append(
                    f"Reference mismatch on {my_name}→{their_name}: "
                    f"{src.reference} → {dst.reference}"
                )

            # Platform check
            if (
                dst.platform != SequencingPlatform.ANY
                and src.platform != SequencingPlatform.ANY
                and src.platform != dst.platform
            ):
                errors.append(
                    f"Platform mismatch on {my_name}→{their_name}: "
                    f"{src.platform} → {dst.platform}"
                )

        return errors

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ContractSchema":
        return cls.model_validate_json(text)

    @classmethod
    def from_file(cls, path: Path) -> "ContractSchema":
        return cls.from_json(path.read_text())
