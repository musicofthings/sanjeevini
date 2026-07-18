"""Sanjeevini pinners: resolve packages to their commit-era versions.

All three pinners share one output contract — given package names and a target
date, return what was live on that date:

* :mod:`~sanjeevini.pinners.pypi` — PyPI JSON API (Python packages).
* :mod:`~sanjeevini.pinners.conda` — conda-forge + bioconda repodata.
* :mod:`~sanjeevini.pinners.bioc` — Bioconductor release calendar + VIEWS.
"""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """Return the Sanjeevini cache root, honouring ``SANJEEVINI_CACHE_DIR``.

    Returns:
        ``$SANJEEVINI_CACHE_DIR`` if set, otherwise ``~/.cache/sanjeevini``.
    """
    override = os.environ.get("SANJEEVINI_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "sanjeevini"
