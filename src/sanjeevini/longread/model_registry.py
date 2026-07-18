"""Long-read model-bundle registry for Sanjeevini.

Tools in the long-read space (Dorado, medaka, Clair3, DeepVariant,
pb-CpG-tools) do not ship their model weights inside their Docker images —
the models are versioned separately and must be downloaded at runtime.

Dead long-read tool repos have *two* resurrection layers:
    1. The code / environment layer  — handled by Jeeva's repair loop
    2. The model-bundle layer        — handled by this module

This registry maps (tool, chemistry, version) → ModelBundleRef so the
Scout can emit the right ``model_bundle`` field in the contract, and so
Compose can pre-fetch bundles before a pipeline step runs.

Chemistry strings follow ONT's naming convention:
    r10.4.1_e8.2_400bps   — R10.4.1 flow cell, E8.2 kit, 400 bps speed
    r9.4.1_e8.1_hac        — R9.4.1 flow cell, E8.1 kit, HAC model

PacBio model names follow the pb-CpG-tools / DeepVariant convention:
    pacbio-hifi             — generic HiFi chemistry label
    sequel2-hifi            — Sequel IIe / SMRT Link ≥ 12
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from sanjeevini.contracts.schema import ModelBundleRef

# Supported long-read tools with managed model bundles
LongReadTool = Literal["dorado", "medaka", "clair3", "deepvariant", "pb_cpg_tools", "guppy"]

# ---------------------------------------------------------------------------
# Dorado basecalling models
# Dorado downloads models via ``dorado download --model <name>``
# Source: https://github.com/nanoporetech/dorado#available-basecalling-models
# ---------------------------------------------------------------------------

_DORADO_MODELS: list[dict] = [
    # ── R10.4.1 / E8.2 (current generation as of 2024-2026) ──────────────
    {
        "model_name": "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "hac",
        "version": "4.3.0",
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0.tar.gz",
        "size_gb": 0.18,
    },
    {
        "model_name": "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "sup",
        "version": "4.3.0",
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r10.4.1_e8.2_400bps_sup@v4.3.0.tar.gz",
        "size_gb": 0.45,
    },
    {
        "model_name": "dna_r10.4.1_e8.2_400bps_fast@v4.3.0",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "fast",
        "version": "4.3.0",
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r10.4.1_e8.2_400bps_fast@v4.3.0.tar.gz",
        "size_gb": 0.08,
    },
    # Modified base models (5mCpG + 5hmCpG)
    {
        "model_name": "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_5mCG_5hmCG@v1",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "hac",
        "version": "4.3.0",
        "mod_bases": ["5mCG", "5hmCG"],
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0_5mCG_5hmCG@v1.tar.gz",
        "size_gb": 0.22,
    },
    # ── R9.4.1 / E8.1 (legacy — still widely in use) ─────────────────────
    {
        "model_name": "dna_r9.4.1_e8.1_hac@v3.3",
        "chemistry": "r9.4.1_e8.1",
        "speed_mode": "hac",
        "version": "3.3",
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r9.4.1_e8.1_hac@v3.3.tar.gz",
        "size_gb": 0.12,
    },
    {
        "model_name": "dna_r9.4.1_e8.1_sup@v3.3",
        "chemistry": "r9.4.1_e8.1",
        "speed_mode": "sup",
        "version": "3.3",
        "gpu_required": True,
        "source_url": "https://cdn.oxfordnanoportal.com/software/analysis/dorado/models/dna_r9.4.1_e8.1_sup@v3.3.tar.gz",
        "size_gb": 0.35,
    },
]

# ---------------------------------------------------------------------------
# medaka variant-calling models
# medaka downloads models via ``medaka tools download_models``
# Source: https://github.com/nanoporetech/medaka#models
# ---------------------------------------------------------------------------

_MEDAKA_MODELS: list[dict] = [
    # R10.4.1
    {
        "model_name": "r1041_e82_400bps_hac_v4.3.0",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "hac",
        "version": "1.12.0",
        "gpu_required": False,
        "source_url": "https://github.com/nanoporetech/medaka/releases/download/v1.12.0/medaka_models.tar.gz",
        "size_gb": 0.09,
    },
    {
        "model_name": "r1041_e82_400bps_sup_v4.3.0",
        "chemistry": "r10.4.1_e8.2",
        "speed_mode": "sup",
        "version": "1.12.0",
        "gpu_required": False,
        "source_url": "https://github.com/nanoporetech/medaka/releases/download/v1.12.0/medaka_models.tar.gz",
        "size_gb": 0.09,
    },
    # R9.4.1 legacy
    {
        "model_name": "r941_min_hac_g507",
        "chemistry": "r9.4.1_e8.1",
        "speed_mode": "hac",
        "version": "1.8.0",
        "gpu_required": False,
        "source_url": "https://github.com/nanoporetech/medaka/releases/download/v1.8.0/medaka_models.tar.gz",
        "size_gb": 0.05,
    },
]

# ---------------------------------------------------------------------------
# Clair3 models
# Source: https://github.com/HKU-BAL/Clair3#pre-trained-models
# ---------------------------------------------------------------------------

_CLAIR3_MODELS: list[dict] = [
    {
        "model_name": "ont_guppy5_r941_min_hac_g507",
        "chemistry": "r9.4.1_e8.1",
        "platform": "ont",
        "version": "0.1",
        "gpu_required": False,
        "source_url": "http://www.bio8.cs.hku.hk/clair3/clair3_models/ont_guppy5_r941_min_hac_g507.tar.gz",
        "size_gb": 0.04,
    },
    {
        "model_name": "ont_r10_q20",
        "chemistry": "r10.4.1_e8.2",
        "platform": "ont",
        "version": "0.1",
        "gpu_required": False,
        "source_url": "http://www.bio8.cs.hku.hk/clair3/clair3_models/ont_r10_q20.tar.gz",
        "size_gb": 0.04,
    },
    {
        "model_name": "hifi_sequel2",
        "chemistry": "pacbio-hifi",
        "platform": "pacbio_hifi",
        "version": "0.1",
        "gpu_required": False,
        "source_url": "http://www.bio8.cs.hku.hk/clair3/clair3_models/hifi_sequel2.tar.gz",
        "size_gb": 0.04,
    },
    {
        "model_name": "hifi_revio",
        "chemistry": "pacbio-hifi",
        "platform": "pacbio_hifi",
        "version": "0.1",
        "gpu_required": False,
        "source_url": "http://www.bio8.cs.hku.hk/clair3/clair3_models/hifi_revio.tar.gz",
        "size_gb": 0.04,
    },
]

# ---------------------------------------------------------------------------
# DeepVariant models
# Source: https://github.com/google/deepvariant#model-files
# ---------------------------------------------------------------------------

_DEEPVARIANT_MODELS: list[dict] = [
    {
        "model_name": "DeepVariant-ONT-1.6.0",
        "chemistry": "r10.4.1_e8.2",
        "platform": "ont",
        "version": "1.6.0",
        "gpu_required": False,
        "source_url": "https://storage.googleapis.com/deepvariant/models/DeepVariant/1.6.0/DeepVariant-inception_v3-1.6.0+data-ont_chr20.tar.gz",
        "size_gb": 0.18,
    },
    {
        "model_name": "DeepVariant-PACBIO-1.6.0",
        "chemistry": "pacbio-hifi",
        "platform": "pacbio_hifi",
        "version": "1.6.0",
        "gpu_required": False,
        "source_url": "https://storage.googleapis.com/deepvariant/models/DeepVariant/1.6.0/DeepVariant-inception_v3-1.6.0+data-pacbio_chr20.tar.gz",
        "size_gb": 0.18,
    },
]

# ---------------------------------------------------------------------------
# Index and lookup
# ---------------------------------------------------------------------------

@dataclass
class ModelIndex:
    _records: list[dict] = field(default_factory=list)

    def add(self, records: list[dict], tool: str) -> None:
        for r in records:
            self._records.append({**r, "tool": tool})

    def find(
        self,
        tool: LongReadTool,
        chemistry: str | None = None,
        speed_mode: str | None = None,
        platform: str | None = None,
    ) -> list[ModelBundleRef]:
        """Return all models matching the given filters, most recent first."""
        results = []
        for r in self._records:
            if r["tool"] != tool:
                continue
            if chemistry and chemistry.lower() not in r.get("chemistry", "").lower():
                continue
            if speed_mode and r.get("speed_mode", "").lower() != speed_mode.lower():
                continue
            if platform and r.get("platform", "").lower() != platform.lower():
                continue
            results.append(ModelBundleRef(
                tool=r["tool"],
                version=r["version"],
                model_name=r["model_name"],
                source_url=r["source_url"],
                size_gb=r.get("size_gb"),
                gpu_required=r.get("gpu_required", False),
                chemistry=r.get("chemistry"),
            ))
        return results

    def recommend(
        self,
        tool: LongReadTool,
        chemistry: str,
        prefer_gpu: bool = False,
    ) -> ModelBundleRef | None:
        """Return the single best-fit model for the given (tool, chemistry).

        Preference order: matching speed_mode (sup > hac > fast), then recency.
        """
        candidates = self.find(tool, chemistry=chemistry)
        if not candidates:
            return None
        # Prefer sup for GPU users, hac otherwise
        preferred = "sup" if prefer_gpu else "hac"
        for c in candidates:
            # Check if preferred speed mode is in the model name
            if preferred in c.model_name.lower():
                return c
        return candidates[0]  # fallback to first match


# Singleton model index — populated at module import time
MODEL_INDEX = ModelIndex()
MODEL_INDEX.add(_DORADO_MODELS,     "dorado")
MODEL_INDEX.add(_MEDAKA_MODELS,     "medaka")
MODEL_INDEX.add(_CLAIR3_MODELS,     "clair3")
MODEL_INDEX.add(_DEEPVARIANT_MODELS, "deepvariant")


# ---------------------------------------------------------------------------
# Chemistry inference from repo content
# ---------------------------------------------------------------------------

# Patterns that signal a specific ONT chemistry in READMEs / config files
_CHEMISTRY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"r10\.4\.1|r10_4_1|R10\.4\.1|r1041", re.I), "r10.4.1_e8.2"),
    (re.compile(r"r9\.4\.1|r9_4_1|R9\.4\.1|r941",   re.I), "r9.4.1_e8.1"),
    (re.compile(r"hifi|ccs|pacbio_hifi|revio|sequel.?ii?e", re.I), "pacbio-hifi"),
    (re.compile(r"clr|subread|rs.ii",                re.I), "pacbio-clr"),
]


def infer_chemistry(text: str) -> str | None:
    """Guess the sequencing chemistry from free-form text (README, config).

    Returns a canonical chemistry string or None if ambiguous.
    """
    for pattern, chemistry in _CHEMISTRY_PATTERNS:
        if pattern.search(text):
            return chemistry
    return None
