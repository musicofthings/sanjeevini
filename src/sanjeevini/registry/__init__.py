"""Sanjeevini registry — the living archive of resurrected tools (Phase 5).

Every resurrection emits a ``contract.yaml`` and a ``PROVENANCE.json``. The
registry is the discoverable index over those: one :class:`RegistryEntry` per
revived tool. :mod:`~sanjeevini.registry.catalog` loads and pulls entries;
:mod:`~sanjeevini.registry.search` ranks them by a natural-language query.
"""

from __future__ import annotations
