"""Sanjeevini Compose — wire resurrected bricks into typed pipelines (Phase 6).

A Compose pipeline is a YAML file whose steps reference revived tools by slug.
Compose validates every port mapping against the tools' typed
:class:`~sanjeevini.contracts.schema.ContractSchema` *before* any container
starts, so a BAM→VCF mismatch is caught at build time rather than three hours
into a run. See :mod:`sanjeevini.compose.pipeline`.
"""

from __future__ import annotations
