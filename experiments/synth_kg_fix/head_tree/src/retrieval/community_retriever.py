"""Community retriever and Hybrid Retriever Orchestrator combining graph, vector, and community streams."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from src.core.interfaces import EmbeddingProvider
from src.core.models import (
    CommunityReport,
    Evidence,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    RetrievalResult,
)
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.reranker import RRFReranker
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.retrieval.vector_retriever import VectorEvidenceRetriever
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class CommunityRetriever:
    """Retrieves high-level community summaries and their supporting chunks for global queries."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        community_top_k: int = 3,
    ) -> None:
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.community_top_k = community_top_k

    def retrieve_communities(
        self, question: str, kb_id: str, scope_id: Optional[str] = None
    ) -> Tuple[List[CommunityReport], List[Evidence]]:
        """Retrieve relevant community reports and convert their supporting chunks into Evidence."""
        if not question or not question.strip():
            return [], []

        query_emb = self.embedding_provider.embed_text(question)
        hits = self.vector_store.query_community_reports(
            query_embedding=query_emb,
            top_k=self.community_top_k,
        )

        all_reports = self.doc_store.load_community_reports()
        reports_by_id: Dict[str, CommunityReport] = {r.report_id: r for r in all_reports}
        reports_by_comm_id: Dict[int, CommunityReport] = {r.community_id: r for r in all_reports}

        matched_reports: List[CommunityReport] = []
        community_evidence: List[Evidence] = []
        seen_chunk_ids = set()

        for hit in hits:
            report_id = hit["id"]
            report = reports_by_id.get(report_id)
            if report is None:
                # Fallback to metadata community_id if present
                meta_cid = hit.get("metadata", {}).get("community_id")
                if meta_cid is not None:
                    report = reports_by_comm_id.get(int(meta_cid))

            if report is None:
                continue

            matched_reports.append(report)

            # Gather supporting chunks for this community
            for cid in report.supporting_chunk_ids:
                if cid in seen_chunk_ids:
                    continue
                chunk = self.doc_store.get_chunk(cid)
                if chunk is None:
                    continue
                if scope_id and chunk.scope_id != scope_id:
                    continue

                seen_chunk_ids.add(cid)
                ev = Evidence(
                    kb_id=kb_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_numbers=list(chunk.page_numbers),
                    text=chunk.text,
                    routes=[ProvenanceRoute.COMMUNITY],
                    graph_rank=None,
                    vector_rank=None,
                    vector_distance=hit.get("distance"),
                    rrf_score=0.0,
                    supporting_graph_paths=[],
                    supporting_edge_ids=[],
                    community_ids=[report.community_id],
                )
                community_evidence.append(ev)

        return matched_reports, community_evidence


class HybridRetrieverOrchestrator:
    """Orchestrates hybrid retrieval dispatching by QueryIntent (LOCAL, GLOBAL, MIXED)."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        seed_retriever: SeedEntityRetriever,
        graph_retriever: GraphEvidenceRetriever,
        vector_retriever: VectorEvidenceRetriever,
        community_retriever: CommunityRetriever,
        reranker: Optional[RRFReranker] = None,
        final_context_chunks: int = 8,
    ) -> None:
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.seed_retriever = seed_retriever
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.community_retriever = community_retriever
        self.reranker = reranker or RRFReranker(rrf_k=60)
        self.final_context_chunks = final_context_chunks

    def retrieve_hybrid(
        self,
        question: str,
        plan: QueryPlan,
        kb_id: str,
        scope_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute hybrid retrieval based on the query plan's intent."""
        # 1. Check if clarification is required up front
        if plan.requires_clarification:
            return RetrievalResult(
                kb_id=kb_id,
                question=question,
                query_plan=plan,
                resolved_seed_entity_ids=[],
                evidence_list=[],
                community_reports=[],
                abstained=True,
                abstention_reason=plan.clarification_reason or "Query requires clarification.",
            )

        resolved_seed_ids: List[str] = []
        graph_ev: List[Evidence] = []
        vector_ev: List[Evidence] = []
        comm_reports: List[CommunityReport] = []
        comm_ev: List[Evidence] = []
        route_errors: Dict[str, str] = {}

        # 2. Dispatch by intent
        if plan.intent == QueryIntent.LOCAL:
            seed_res = None
            if plan.seed_entities:
                seed_res = self.seed_retriever.resolve_seeds(plan.seed_entities)
                resolved_seed_ids = seed_res.resolved_entity_ids
            if resolved_seed_ids or getattr(self.graph_retriever, "document_title_discovery_enabled", False):
                try:
                    graph_ev = self.graph_retriever.retrieve(
                        question, plan, kb_id, scope_id=scope_id, seed_resolution_result=seed_res
                    )
                except Exception as exc:
                    route_errors["graph"] = str(exc)
            try:
                vector_ev = self.vector_retriever.retrieve(question, plan, kb_id, scope_id=scope_id)
            except Exception as exc:
                route_errors["vector"] = str(exc)
            merged = EvidenceMerger.merge([graph_ev, vector_ev])
            reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)

            abstained = len(reranked) == 0
            reason = "No relevant evidence found in the knowledge base." if abstained else None

            return RetrievalResult(
                kb_id=kb_id,
                question=question,
                query_plan=plan,
                resolved_seed_entity_ids=resolved_seed_ids,
                evidence_list=reranked,
                community_reports=[],
                abstained=abstained,
                abstention_reason=reason,
                route_diagnostics={"vector_candidates": len(vector_ev), "graph_candidates": len(graph_ev),
                                   "community_candidates": 0, "seed_entity_count": len(resolved_seed_ids),
                                   "invalid_relation_count": len(plan.invalid_relationship_types),
                                   "evidence_after_merge": len(merged), "evidence_after_reranking": len(reranked),
                                   "route_errors": route_errors,
                                   "graph_details": dict(getattr(self.graph_retriever, "last_diagnostics", {}))},
            )

        elif plan.intent == QueryIntent.GLOBAL:
            try: comm_reports, comm_ev = self.community_retriever.retrieve_communities(question, kb_id, scope_id)
            except Exception as exc: route_errors["community"] = str(exc)
            try: vector_ev = self.vector_retriever.retrieve(question, plan, kb_id, scope_id=scope_id)
            except Exception as exc: route_errors["vector"] = str(exc)
            merged = EvidenceMerger.merge([comm_ev, vector_ev])
            reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)

            abstained = len(reranked) == 0 and len(comm_reports) == 0
            reason = "No relevant community or text evidence found." if abstained else None

            return RetrievalResult(
                kb_id=kb_id,
                question=question,
                query_plan=plan,
                resolved_seed_entity_ids=[],
                evidence_list=reranked,
                community_reports=comm_reports,
                abstained=abstained,
                abstention_reason=reason,
                route_diagnostics={"vector_candidates": len(vector_ev), "graph_candidates": 0,
                                   "community_candidates": len(comm_ev), "seed_entity_count": 0,
                                   "invalid_relation_count": len(plan.invalid_relationship_types),
                                   "evidence_after_merge": len(merged), "evidence_after_reranking": len(reranked),
                                   "route_errors": route_errors,
                                   "graph_details": dict(getattr(self.graph_retriever, "last_diagnostics", {}))},
            )

        else:  # MIXED intent
            seed_res = None
            if plan.seed_entities:
                seed_res = self.seed_retriever.resolve_seeds(plan.seed_entities)
                resolved_seed_ids = seed_res.resolved_entity_ids

            if resolved_seed_ids or getattr(self.graph_retriever, "document_title_discovery_enabled", False):
                try:
                    graph_ev = self.graph_retriever.retrieve(
                        question, plan, kb_id, scope_id=scope_id, seed_resolution_result=seed_res
                    )
                except Exception as exc:
                    route_errors["graph"] = str(exc)
            try: comm_reports, comm_ev = self.community_retriever.retrieve_communities(question, kb_id, scope_id)
            except Exception as exc: route_errors["community"] = str(exc)
            try: vector_ev = self.vector_retriever.retrieve(question, plan, kb_id, scope_id=scope_id)
            except Exception as exc: route_errors["vector"] = str(exc)

            merged = EvidenceMerger.merge([graph_ev, comm_ev, vector_ev])
            reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)

            abstained = len(reranked) == 0 and len(comm_reports) == 0
            reason = "No relevant evidence found across graph, vector, or community streams." if abstained else None

            return RetrievalResult(
                kb_id=kb_id,
                question=question,
                query_plan=plan,
                resolved_seed_entity_ids=resolved_seed_ids,
                evidence_list=reranked,
                community_reports=comm_reports,
                abstained=abstained,
                abstention_reason=reason,
                route_diagnostics={"vector_candidates": len(vector_ev), "graph_candidates": len(graph_ev),
                                   "community_candidates": len(comm_ev), "seed_entity_count": len(resolved_seed_ids),
                                   "invalid_relation_count": len(plan.invalid_relationship_types),
                                   "evidence_after_merge": len(merged), "evidence_after_reranking": len(reranked),
                                   "route_errors": route_errors,
                                   "graph_details": dict(getattr(self.graph_retriever, "last_diagnostics", {}))},
            )
