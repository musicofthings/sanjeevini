"""Workflow-aware Scout for Sanjeevini.

When the Scout reads a target repo and detects a workflow language entry
point (Nextflow main.nf, Snakefile, *.wdl, *.cwl), it switches from the
default Python resurrection strategy to a *workflow-aware* one.

The key differences from Python resurrection
--------------------------------------------
- The "brick" emitted by the Contract Emitter is a wrapper around the
  workflow runner (``nextflow run``, ``snakemake``, ``miniwdl run``), not
  an importable Python module.
- The repair loop focuses on container profile resolution and process-level
  dependency pinning rather than Python import errors.
- For Nextflow: DSL version matters — DSL1 is deprecated and requires a
  patched runner or a DSL2 migration.
- For nf-core pipelines: the Scout queries the nf-core API to find the
  last working Nextflow version for that pipeline revision.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

WorkflowLanguage = Literal["nextflow", "snakemake", "wdl", "cwl", "unknown"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Files whose presence strongly indicate a workflow language
_SIGNATURES: list[tuple[str, WorkflowLanguage]] = [
    ("main.nf", "nextflow"),
    ("nextflow.config", "nextflow"),
    ("Snakefile", "snakemake"),
    ("workflow/Snakefile", "snakemake"),
    ("workflow.cwl", "cwl"),
    ("main.cwl", "cwl"),
]
# For WDL we scan for *.wdl rather than a fixed filename
_WDL_GLOB = "*.wdl"
_NF_GLOB = "*.nf"


def detect_workflow_language(repo_root: Path) -> WorkflowLanguage:
    """Return the dominant workflow language in *repo_root*, or 'unknown'."""
    for filename, lang in _SIGNATURES:
        if (repo_root / filename).exists():
            return lang
    if list(repo_root.rglob(_WDL_GLOB)):
        return "wdl"
    if list(repo_root.rglob(_NF_GLOB)):
        return "nextflow"
    return "unknown"


# ---------------------------------------------------------------------------
# Nextflow-specific analysis
# ---------------------------------------------------------------------------


@dataclass
class NextflowProfile:
    name: str
    container_engine: Literal["docker", "singularity", "conda", "none"] = "docker"
    # per-process containers harvested from the profile block
    process_containers: dict[str, str] = field(default_factory=dict)


@dataclass
class NextflowAnalysis:
    dsl_version: Literal[1, 2, "unknown"] = "unknown"
    min_nextflow_version: str | None = None  # e.g. "23.04.0"
    main_nf: Path | None = None
    config_files: list[Path] = field(default_factory=list)
    profiles: list[NextflowProfile] = field(default_factory=list)
    nfcore_pipeline: bool = False
    nfcore_name: str | None = None  # e.g. "sarek"
    nfcore_revision: str | None = None  # e.g. "3.3.2"
    # Containers referenced at the process level (outside profiles)
    process_containers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


_DSL2_RE = re.compile(r"nextflow\s*\.\s*enable\s*\.\s*dsl\s*=\s*2", re.IGNORECASE)
_DSL1_RE = re.compile(r"nextflow\s*\.\s*enable\s*\.\s*dsl\s*=\s*1", re.IGNORECASE)
_NFVER_RE = re.compile(r"nextflowVersion\s*[=!><]+\s*['\"]?([0-9][^\s'\"]+)['\"]?", re.IGNORECASE)
_CONTAINER_RE = re.compile(r"""container\s+['"]([^'"]+)['"]""")
_PROFILE_RE = re.compile(r"profiles\s*\{", re.IGNORECASE)
_NFCORE_META = re.compile(r"nf-core/([a-z0-9_-]+)", re.IGNORECASE)


def analyse_nextflow(repo_root: Path) -> NextflowAnalysis:
    """Parse a Nextflow repo and return an analysis that shapes resurrection."""
    ana = NextflowAnalysis()

    # Locate main.nf
    main_candidates = list(repo_root.glob("main.nf")) + list(repo_root.glob("workflow/main.nf"))
    if main_candidates:
        ana.main_nf = main_candidates[0]

    # Locate all config files
    ana.config_files = sorted(repo_root.rglob("*.config"))

    # Determine DSL version from main.nf or any .nf file
    nf_sources = list(repo_root.rglob("*.nf"))[:20]  # sample first 20
    for nf in nf_sources:
        try:
            text = nf.read_text(errors="replace")
        except OSError:
            continue
        if _DSL2_RE.search(text):
            ana.dsl_version = 2
            break
        if _DSL1_RE.search(text):
            ana.dsl_version = 1
            ana.issues.append(
                "DSL1 detected — Nextflow ≥ 22.12 dropped DSL1; "
                "resurrection must pin nextflow ≤ 22.10.8 or migrate to DSL2."
            )
            break

    if ana.dsl_version == "unknown" and nf_sources:
        # Most repos after 2022 are DSL2 without the explicit enable statement
        ana.dsl_version = 2

    # Parse nextflow.config for version requirement and containers
    for cfg in ana.config_files:
        try:
            cfg_text = cfg.read_text(errors="replace")
        except OSError:
            continue
        m = _NFVER_RE.search(cfg_text)
        if m:
            ana.min_nextflow_version = m.group(1).strip("'\" ")
        # Harvest container directives outside profiles
        for hit in _CONTAINER_RE.finditer(cfg_text):
            img = hit.group(1)
            # Attribute to process if we can find the surrounding withName block
            # (simplified: just collect all unique container images)
            if img not in ana.process_containers.values():
                key = f"process_{len(ana.process_containers)}"
                ana.process_containers[key] = img

    # Detect nf-core
    all_text = ""
    for p in [ana.main_nf, *ana.config_files[:3]]:
        if p and p.exists():
            with contextlib.suppress(OSError):
                all_text += p.read_text(errors="replace")
    m = _NFCORE_META.search(all_text)
    if m:
        ana.nfcore_pipeline = True
        ana.nfcore_name = m.group(1).lower()

    return ana


# ---------------------------------------------------------------------------
# Snakemake analysis
# ---------------------------------------------------------------------------


@dataclass
class SnakemakeAnalysis:
    snakefile: Path | None = None
    min_snakemake_version: str | None = None
    conda_envs: list[Path] = field(default_factory=list)
    singularity_images: list[str] = field(default_factory=list)
    docker_images: list[str] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


_SM_VERSION_RE = re.compile(r'min_version\s*\(\s*["\']([0-9][^"\']+)["\']', re.IGNORECASE)
_SINGULARITY_RE = re.compile(r'singularity:\s*["\']([^"\']+)["\']')
_DOCKER_RE = re.compile(r'container:\s*["\']docker://([^"\']+)["\']')


def analyse_snakemake(repo_root: Path) -> SnakemakeAnalysis:
    ana = SnakemakeAnalysis()

    # Locate Snakefile
    candidates = [repo_root / "Snakefile", repo_root / "workflow" / "Snakefile"]
    for c in candidates:
        if c.exists():
            ana.snakefile = c
            break

    # Locate config files
    ana.config_files = sorted(repo_root.glob("config/*.yaml")) + sorted(
        repo_root.glob("config/*.yml")
    )

    if ana.snakefile:
        try:
            text = ana.snakefile.read_text(errors="replace")
        except OSError:
            return ana
        m = _SM_VERSION_RE.search(text)
        if m:
            ana.min_snakemake_version = m.group(1)
        ana.singularity_images = _SINGULARITY_RE.findall(text)
        ana.docker_images = _DOCKER_RE.findall(text)

    # Conda env files
    ana.conda_envs = sorted(repo_root.rglob("envs/*.yaml")) + sorted(repo_root.rglob("envs/*.yml"))

    return ana


# ---------------------------------------------------------------------------
# WDL / CWL (lightweight — these are less common)
# ---------------------------------------------------------------------------


@dataclass
class WDLAnalysis:
    main_wdl: Path | None = None
    runtime_images: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


_WDL_DOCKER_RE = re.compile(r'docker\s*:\s*["\']([^"\']+)["\']')


def analyse_wdl(repo_root: Path) -> WDLAnalysis:
    ana = WDLAnalysis()
    wdl_files = sorted(repo_root.rglob("*.wdl"))
    if not wdl_files:
        return ana
    ana.main_wdl = wdl_files[0]
    for wf in wdl_files[:10]:
        try:
            text = wf.read_text(errors="replace")
        except OSError:
            continue
        ana.runtime_images.extend(_WDL_DOCKER_RE.findall(text))
    ana.runtime_images = list(dict.fromkeys(ana.runtime_images))  # dedup, order-preserving
    return ana


# ---------------------------------------------------------------------------
# Unified WorkflowPlan
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResurrectionPlan:
    """Structured resurrection plan produced by the Workflow Scout.

    This is consumed by the repair loop, which runs a workflow-appropriate
    sandbox strategy (container profile resolution, runner version pinning)
    rather than the default Python import + traceback loop.
    """

    language: WorkflowLanguage
    entry_point: str  # e.g. "nextflow run main.nf"
    container_strategy: str  # e.g. "-profile docker"
    runner_version_pin: str | None  # e.g. "nextflow 23.10.1"
    sanity_check: str  # falsifiable pass criterion
    known_issues: list[str] = field(default_factory=list)
    # Raw analysis objects for downstream consumption
    nextflow: NextflowAnalysis | None = None
    snakemake: SnakemakeAnalysis | None = None
    wdl: WDLAnalysis | None = None


def build_resurrection_plan(repo_root: Path) -> WorkflowResurrectionPlan:
    """Top-level entry: detect language, analyse, return a WorkflowResurrectionPlan."""
    lang = detect_workflow_language(repo_root)

    if lang == "nextflow":
        nf = analyse_nextflow(repo_root)
        pin = None
        if nf.min_nextflow_version:
            pin = f"nextflow {nf.min_nextflow_version}"
        elif nf.dsl_version == 1:
            pin = "nextflow 22.10.8"  # last version with DSL1 support
        entry = f"nextflow run {nf.main_nf or 'main.nf'} -profile docker"
        plan = WorkflowResurrectionPlan(
            language="nextflow",
            entry_point=entry,
            container_strategy="-profile docker",
            runner_version_pin=pin,
            sanity_check=(
                "Pipeline exits 0 on the bundled test input "
                "(--profile test,docker) and the output directory is non-empty "
                "with at least one results file."
            ),
            known_issues=nf.issues,
            nextflow=nf,
        )
        if nf.nfcore_pipeline:
            plan.known_issues.append(
                f"nf-core/{nf.nfcore_name} detected — "
                "check https://nf-co.re for the last tested Nextflow version."
            )
        return plan

    if lang == "snakemake":
        sm = analyse_snakemake(repo_root)
        return WorkflowResurrectionPlan(
            language="snakemake",
            entry_point="snakemake --cores 4 --use-conda",
            container_strategy="--use-conda",
            runner_version_pin=(
                f"snakemake=={sm.min_snakemake_version}" if sm.min_snakemake_version else None
            ),
            sanity_check=(
                "snakemake --dryrun exits 0 on the bundled config; all target rules are reachable."
            ),
            known_issues=sm.issues,
            snakemake=sm,
        )

    if lang == "wdl":
        wdl = analyse_wdl(repo_root)
        return WorkflowResurrectionPlan(
            language="wdl",
            entry_point=f"miniwdl run {wdl.main_wdl or 'workflow.wdl'}",
            container_strategy="miniwdl (Docker backend)",
            runner_version_pin=None,
            sanity_check=(
                "miniwdl run exits 0 on the bundled test inputs; "
                "output JSON is valid and all declared outputs are present."
            ),
            known_issues=wdl.issues,
            wdl=wdl,
        )

    # CWL or unknown — generic fallback
    return WorkflowResurrectionPlan(
        language=lang,
        entry_point="cwltool workflow.cwl inputs.yaml",
        container_strategy="--podman",
        runner_version_pin=None,
        sanity_check="cwltool exits 0 on the bundled test inputs.",
    )
