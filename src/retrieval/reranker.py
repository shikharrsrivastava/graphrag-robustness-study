"""Reciprocal Rank Fusion (RRF) Reranker for hybrid GraphRAG evidence."""

from __future__ import annotations
import logging
from typing import List, Optional
from src.core.models import Evidence

logger = logging.getLogger(__name__)


class RRFReranker:
    """Reranks Evidence using Reciprocal Rank Fusion combining graph and vector ranks."""

    def __init__(self, rrf_k: int = 60) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer.")
        self.rrf_k = rrf_k

    def rerank(self, evidence_list: List[Evidence], top_k: Optional[int] = 8) -> List[Evidence]:
        """Compute RRF scores and sort evidence items with deterministic tie-breaking."""
        if not evidence_list:
            return []

        for ev in evidence_list:
            graph_score = (
                1.0 / (self.rrf_k + ev.graph_rank)
                if ev.graph_rank is not None and ev.graph_rank > 0
                else 0.0
            )
            vector_score = (
                1.0 / (self.rrf_k + ev.vector_rank)
                if ev.vector_rank is not None and ev.vector_rank > 0
                else 0.0
            )
            ev.rrf_score = graph_score + vector_score

        # Deterministic sorting:
        # 1. rrf_score DESC
        # 2. Number of supporting edge IDs DESC (more graph grounding)
        # 3. Has graph rank (0 if has graph rank else 1)
        # 4. chunk_id ASC (lexicographical tie-breaker)
        sorted_evidence = sorted(
            evidence_list,
            key=lambda ev: (
                -ev.rrf_score,
                -len(ev.supporting_edge_ids),
                0 if ev.graph_rank is not None else 1,
                ev.chunk_id,
            ),
        )

        if top_k is not None:
            return sorted_evidence[:top_k]
        return sorted_evidence
