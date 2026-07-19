"""Semantic (and lexical-fallback) search over the registry catalog.

Each entry's ``name + capability + domain`` text is ranked against a
natural-language query. When ``sentence-transformers`` is installed (the
``[search]`` extra), the engine embeds with ``all-MiniLM-L6-v2`` and ranks by
cosine similarity — accelerated by FAISS when available. Otherwise it falls back
to a dependency-free lexical scorer so ``jeeva registry search`` still works out
of the box; the CLI tells the user how to enable embeddings.

The public surface (:class:`RegistrySearchEngine` with :meth:`build_index` and
:meth:`search`) is identical across backends, so callers never branch on it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from sanjeevini.pinners import cache_root
from sanjeevini.registry.catalog import RegistryEntry

_MODEL_NAME = "all-MiniLM-L6-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from ``text``."""
    return _TOKEN_RE.findall(text.lower())


def index_path() -> Path:
    """Return the on-disk FAISS index location under the Sanjeevini cache."""
    return cache_root() / "registry" / "index.faiss"


class RegistrySearchEngine:
    """Ranks catalog entries against a query via embeddings or a lexical fallback.

    Attributes:
        backend: ``"semantic"`` once an embedding model is loaded, else
            ``"lexical"``. Set by :meth:`build_index`.
    """

    def __init__(self, catalog: list[RegistryEntry]) -> None:
        """Store the catalog to search over.

        Args:
            catalog: The registry entries to rank.
        """
        self.catalog = list(catalog)
        self.backend = "lexical"
        self._texts = [e.search_text() for e in self.catalog]
        self._token_sets: list[set[str]] = []
        self._embeddings: list[list[float]] = []
        self._model: Any = None

    def build_index(self) -> None:
        """Build the search index, choosing the best available backend.

        Tries the semantic (embedding) backend first and falls back to the
        lexical backend if ``sentence-transformers`` is not installed.
        """
        if self._build_semantic():
            self.backend = "semantic"
        else:
            self.backend = "lexical"
            self._token_sets = [set(_tokenize(t)) for t in self._texts]

    def _build_semantic(self) -> bool:  # pragma: no cover - requires optional deps
        """Load the embedding model and embed the catalog; ``False`` if absent."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return False
        self._model = SentenceTransformer(_MODEL_NAME)
        if not self.catalog:
            return True
        vectors = self._model.encode(self._texts, normalize_embeddings=True)
        self._embeddings = [[float(x) for x in row] for row in vectors]
        self._persist_faiss()
        return True

    def _persist_faiss(self) -> None:  # pragma: no cover - requires optional deps
        """Write the embeddings to a FAISS index, if faiss is installed."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            return
        if not self._embeddings:
            return
        matrix = np.asarray(self._embeddings, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        path = index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: str | None = None,
        platform_filter: str | None = None,
    ) -> list[tuple[RegistryEntry, float]]:
        """Return the ``top_k`` best-matching entries as ``(entry, score)`` pairs.

        Args:
            query: The natural-language query.
            top_k: Maximum number of results to return.
            domain_filter: If set, only entries with this domain are considered.
            platform_filter: If set, only entries with this platform are considered.

        Returns:
            ``(entry, score)`` pairs, highest score first. Entries scoring zero
            are dropped.
        """
        candidates = [
            i
            for i, entry in enumerate(self.catalog)
            if (domain_filter is None or entry.domain == domain_filter)
            and (platform_filter is None or entry.platform == platform_filter)
        ]
        if self.backend == "semantic":
            scores = self._semantic_scores(query, candidates)
        else:
            scores = self._lexical_scores(query, candidates)

        ranked = sorted(
            ((self.catalog[i], score) for i, score in scores if score > 0.0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _lexical_scores(
        self, query: str, candidates: list[int]
    ) -> list[tuple[int, float]]:
        """Score candidates by set-cosine overlap of query and entry tokens."""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        results: list[tuple[int, float]] = []
        for i in candidates:
            tokens = self._token_sets[i]
            if not tokens:
                results.append((i, 0.0))
                continue
            overlap = len(q_tokens & tokens)
            score = overlap / math.sqrt(len(q_tokens) * len(tokens))
            results.append((i, score))
        return results

    def _semantic_scores(  # pragma: no cover - requires optional deps
        self, query: str, candidates: list[int]
    ) -> list[tuple[int, float]]:
        """Score candidates by cosine similarity of embeddings."""
        if self._model is None or not self._embeddings:
            return [(i, 0.0) for i in candidates]
        q_vec = [
            float(x)
            for x in self._model.encode([query], normalize_embeddings=True)[0]
        ]
        results: list[tuple[int, float]] = []
        for i in candidates:
            emb = self._embeddings[i]
            score = sum(a * b for a, b in zip(q_vec, emb, strict=False))
            results.append((i, score))
        return results
