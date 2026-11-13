"""Vector evidence retriever embedding queries and searching ChromaDB text_chunks."""

from __future__ import annotations
import logging
from typing import List, Optional
from src.core.interfaces import EmbeddingProvider, Retriever
from src.core.models import Evidence, ProvenanceRoute, QueryPlan
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class VectorEvidenceRetriever:
    """Retrieves evidence chunks via dense vector similarity over text_chunks."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int = 10,
    ) -> None:
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        plan: QueryPlan,
        kb_id: str,
        scope_id: Optional[str] = None,
    ) -> List[Evidence]:
        """Retrieve vector evidence for the user's question, satisfying Retriever protocol."""
        if not question or not question.strip():
            return []

        query_emb = self.embedding_provider.embed_text(question)
        hits = self.vector_store.query_chunks(
            query_embedding=query_emb,
            top_k=self.top_k,
            scope_id=scope_id,
        )

        evidence_list: List[Evidence] = []
        rank = 1
        for hit in hits:
            cid = hit["id"]
            chunk = self.doc_store.get_chunk(cid)
            if chunk is None:
                continue

            ev = Evidence(
                kb_id=kb_id,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_numbers=list(chunk.page_numbers),
                text=chunk.text,
                routes=[ProvenanceRoute.VECTOR],
                graph_rank=None,
                vector_rank=rank,
                vector_distance=hit.get("distance"),
                rrf_score=0.0,
                supporting_graph_paths=[],
                supporting_edge_ids=[],
                community_ids=[],
            )
            evidence_list.append(ev)
            rank += 1

        return evidence_list
