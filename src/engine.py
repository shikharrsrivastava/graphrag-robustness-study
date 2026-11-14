"""Hybrid GraphRAG Engine orchestrating end-to-end indexing and retrieval pipelines."""

from __future__ import annotations
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import igraph
from src.communities.community_summarizer import CommunitySummarizer
from src.core.configuration import EngineConfig
from src.core.interfaces import EmbeddingProvider
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CommunityReport,
    Document,
    EntityMention,
    Evidence,
    Page,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    RetrievalResult,
)
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.entity_extractor import OllamaEntityExtractor
from src.extraction.relation_extractor import OllamaRelationshipExtractor
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import OllamaAnswerGenerator
from src.graph.clustering import LeidenCommunityDetector
from src.graph.graph_builder import IGraphBuilder
from src.graph.graph_validator import GraphValidator
from src.graph.traversal import GraphTraverser
from src.ingestion.chunker import DeterministicChunker
from src.ingestion.pdf_loader import PDFLoader
from src.resolution.entity_resolver import ConservativeEntityResolver
from src.retrieval.community_retriever import CommunityRetriever, HybridRetrieverOrchestrator
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.query_planner import OllamaQueryPlanner
from src.retrieval.reranker import RRFReranker
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.retrieval.vector_retriever import VectorEvidenceRetriever
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class HybridGraphRAGEngine:
    """Unified Orchestrator for the Hybrid GraphRAG system supporting 19-stage indexing and querying."""

    def __init__(
        self,
        config: EngineConfig,
        llm_client: Optional[OllamaLLMClient] = None,
        embedding_service: Optional[EmbeddingProvider] = None,
        base_dir: Union[str, Path] = ".",
    ) -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.kb_id = config.kb_id

        # Initialize isolated KB directory structure
        self.kb_paths = config.initialize_kb_directories(base_dir=self.base_dir)

        # Initialize clients (defaulting to live local Ollama models)
        self.llm_client = llm_client or OllamaLLMClient(config=config.ollama)
        self.embedding_service = embedding_service or OllamaEmbeddingService(config=config.ollama)

        # Initialize stores
        self.doc_store = DocumentStore(
            kb_id=self.kb_id,
            storage_root=config.storage_root,
            base_dir=self.base_dir,
        )
        self.vector_store = ChromaVectorStore(
            kb_id=self.kb_id,
            kb_vector_store_dir=self.kb_paths["vector_store"],
        )

        # Graph in-memory cache
        self._graph: Optional[igraph.Graph] = None

    def index_directory(self, input_dir: Union[str, Path], rebuild: bool = True) -> Dict[str, Any]:
        """Execute Stages 1-10 of the offline indexing pipeline."""
        in_path = Path(input_dir)
        if not in_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {in_path}")

        if rebuild:
            # The operation is explicit and deterministic: JSONL files are replaced
            # below and Chroma records from an older manifest are removed first.
            self.vector_store.clear()

        total_started = time.perf_counter()
        stage_seconds: Dict[str, float] = {}
        llm_stats_before = dict(self.llm_client.stats) if hasattr(self.llm_client, "stats") else {}
        stage_started = time.perf_counter()
        logger.info("Stage 1-2: Loading raw documents from %s for KB '%s'...", in_path, self.kb_id)
        loader = PDFLoader()
        documents, pages = loader.load_directory(in_path, self.kb_id)
        self.doc_store.save_documents(documents)
        self.doc_store.save_pages(pages)
        stage_seconds["document_loading"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logger.info("Stage 3: Chunking %d pages...", len(pages))
        chunker = DeterministicChunker(self.config.chunking)
        chunks = chunker.chunk_pages(pages, self.kb_id)
        self.doc_store.save_chunks(chunks)
        stage_seconds["chunking"] = time.perf_counter() - stage_started

        logger.info("Stage 4: Extracting entity mentions and relationships from %d chunks...", len(chunks))
        entity_extractor = OllamaEntityExtractor(ontology=self.config.ontology, llm_client=self.llm_client)
        rel_extractor = OllamaRelationshipExtractor(ontology=self.config.ontology, llm_client=self.llm_client)

        all_mentions: List[EntityMention] = []
        all_triples: List[Relationship] = []
        extraction_failures: List[Dict[str, str]] = []
        entity_extraction_seconds = 0.0
        relationship_extraction_seconds = 0.0

        for chunk in chunks:
            entity_started = time.perf_counter()
            try:
                mentions = entity_extractor.extract_entities(chunk)
                all_mentions.extend(mentions)
            except Exception as exc:
                entity_extraction_seconds += time.perf_counter() - entity_started
                extraction_failures.append({"chunk_id": chunk.chunk_id, "stage": "entity_extraction", "error": str(exc)})
                logger.exception("Entity extraction failed for chunk %s; continuing", chunk.chunk_id)
                continue
            entity_extraction_seconds += time.perf_counter() - entity_started
            relation_started = time.perf_counter()
            try:
                triples = rel_extractor.extract_relationships(chunk, mentions)
                all_triples.extend(triples)
            except Exception as exc:
                relationship_extraction_seconds += time.perf_counter() - relation_started
                extraction_failures.append({"chunk_id": chunk.chunk_id, "stage": "relationship_extraction", "error": str(exc)})
                logger.exception("Relationship extraction failed for chunk %s; retaining entity mentions", chunk.chunk_id)
            else:
                relationship_extraction_seconds += time.perf_counter() - relation_started
        stage_seconds["entity_extraction"] = entity_extraction_seconds
        stage_seconds["relationship_extraction"] = relationship_extraction_seconds

        self.doc_store.save_entity_mentions(all_mentions)
        self.doc_store.save_triples(all_triples)

        logger.info(
            "Stage 5: Resolving canonical entities from %d mentions & remapping %d relationships...",
            len(all_mentions),
            len(all_triples),
        )
        stage_started = time.perf_counter()
        resolver = ConservativeEntityResolver(
            config=self.config,
            embedding_provider=self.embedding_service,
            llm_client=self.llm_client,
        )
        canonical_entities, mention_to_entity, graph_relationships = resolver.resolve_and_remap(
            all_mentions, all_triples
        )

        self.doc_store.save_canonical_entities(canonical_entities)
        self.doc_store.save_mention_to_entity(mention_to_entity)
        self.doc_store.save_relationships(graph_relationships)
        stage_seconds["entity_resolution"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logger.info("Stage 6: Generating embeddings & indexing into ChromaDB...")
        if canonical_entities:
            entity_texts = [f"{e.name} ({e.entity_type}): {e.description}" for e in canonical_entities]
            entity_embeddings = self.embedding_service.embed_batch(entity_texts)
            self.vector_store.upsert_entities(canonical_entities, entity_embeddings)

        if chunks:
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = self.embedding_service.embed_batch(chunk_texts)
            self.vector_store.upsert_chunks(chunks, chunk_embeddings)
        stage_seconds["embedding_and_vector_indexing"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logger.info("Stage 7: Constructing igraph knowledge graph and validating...")
        graph = IGraphBuilder.build_graph(canonical_entities, graph_relationships)
        self._graph = graph

        valid_chunk_ids = {c.chunk_id for c in chunks}
        manifest_path = self.kb_paths["kb_root"] / "manifest.json"
        manifest = GraphValidator.validate(
            graph=graph,
            entities=canonical_entities,
            relationships=graph_relationships,
            valid_chunk_ids=valid_chunk_ids,
            ontology=self.config.ontology,
            manifest_path=manifest_path,
            kb_id=self.kb_id,
        )
        stage_seconds["graph_construction_and_validation"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logger.info(
            "Stage 8: Running Leiden community detection (res=%.2f, seed=%d)...",
            self.config.leiden.resolution_parameter,
            self.config.leiden.random_seed,
        )
        detector = LeidenCommunityDetector(config=self.config.leiden)
        communities = detector.detect_communities(graph, kb_id=self.kb_id)
        self.doc_store.save_communities(communities)
        stage_seconds["community_detection"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logger.info("Stage 9: Generating community summaries and indexing...")
        summarizer = CommunitySummarizer(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_service,
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            model_tag=self.config.ollama.llm_model,
        )
        reports = summarizer.summarize_communities(communities, graph, doc_store=self.doc_store)
        stage_seconds["community_summarization"] = time.perf_counter() - stage_started
        stage_seconds["total_indexing"] = time.perf_counter() - total_started

        rejection_counts = Counter(
            triple.validation_status.value for triple in all_triples
            if triple.validation_status.value.startswith("REJECTED")
        )
        degrees = graph.degree() if graph.vcount() else []
        llm_stats_after = dict(self.llm_client.stats) if hasattr(self.llm_client, "stats") else {}

        logger.info("Stage 10: Indexing complete for KB '%s'.", self.kb_id)
        return {
            "kb_id": self.kb_id,
            "documents": len(documents),
            "pages": len(pages),
            "chunks": len(chunks),
            "entities": len(canonical_entities),
            "relationships": len(graph_relationships),
            "extracted_relationships": len(all_triples),
            "validated_relationships": sum(1 for t in all_triples if t.validation_status.value == "VALIDATED"),
            "rejected_relationships": sum(rejection_counts.values()),
            "rejected_relationships_by_reason": dict(rejection_counts),
            "entity_mentions": len(all_mentions),
            "canonical_entities": len(canonical_entities),
            "communities": len(communities),
            "community_reports": len(reports),
            "graph_vertices": graph.vcount(),
            "graph_edges": graph.ecount(),
            "graph_connected_components": manifest["connected_components_count"],
            "graph_isolated_nodes": len(manifest["isolated_nodes"]),
            "graph_degree_min": min(degrees) if degrees else 0,
            "graph_degree_mean": (sum(degrees) / len(degrees)) if degrees else 0.0,
            "graph_degree_max": max(degrees) if degrees else 0,
            "relationship_type_distribution": manifest["relation_type_distribution"],
            "extraction_failures": extraction_failures,
            "stage_seconds": stage_seconds,
            "llm_requests": llm_stats_after.get("requests", 0) - llm_stats_before.get("requests", 0),
            "llm_failed_requests": llm_stats_after.get("failed_requests", 0) - llm_stats_before.get("failed_requests", 0),
            "llm_failed_generations": llm_stats_after.get("failed_generations", 0) - llm_stats_before.get("failed_generations", 0),
            "manifest_fingerprint": manifest["graph_fingerprint"],
        }

    def get_or_load_graph(self) -> igraph.Graph:
        """Return the active in-memory graph or reconstruct from DocumentStore."""
        if self._graph is None or self._graph.vcount() == 0:
            self._graph = IGraphBuilder.reconstruct_from_store(self.doc_store)
        return self._graph

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        scope_id: Optional[str] = None,
    ) -> Tuple[str, RetrievalResult]:
        """Execute Stages 11-19 of the online retrieval and answer generation pipeline.

        Modes:
          - 'hybrid': Full hybrid pipeline (local/global/mixed query planning + graph + vector + RRF)
          - 'vector_only': Vector retrieval over text_chunks only
          - 'graph_only': Seed resolution and graph traversal only
          - 'community_aware': Forces community-level summaries alongside text/graph evidence
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        clean_question = question.strip()
        graph = self.get_or_load_graph()

        seed_retriever = SeedEntityRetriever(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_service,
            similarity_threshold=self.config.resolution.similarity_candidate_threshold,
            max_candidates_per_seed=getattr(self.config.retrieval, "max_candidates_per_seed", 5),
        )
        graph_retriever = GraphEvidenceRetriever(
            doc_store=self.doc_store,
            seed_retriever=seed_retriever,
            graph=graph,
            retrieval_config=self.config.retrieval,
        )
        vector_retriever = VectorEvidenceRetriever(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_service,
            top_k=self.config.retrieval.vector_top_k,
        )
        community_retriever = CommunityRetriever(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_service,
            community_top_k=self.config.retrieval.community_top_k,
        )
        reranker = RRFReranker(rrf_k=self.config.retrieval.rrf_k)
        orchestrator = HybridRetrieverOrchestrator(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_service,
            seed_retriever=seed_retriever,
            graph_retriever=graph_retriever,
            vector_retriever=vector_retriever,
            community_retriever=community_retriever,
            reranker=reranker,
            final_context_chunks=self.config.retrieval.final_context_chunks,
        )
        query_planner = OllamaQueryPlanner(ontology=self.config.ontology, llm_client=self.llm_client)
        context_builder = ContextBuilder(max_chars=self.config.retrieval.max_context_chars)
        answer_generator = OllamaAnswerGenerator(llm_client=self.llm_client, context_builder=context_builder)

        mode_clean = mode.strip().lower()
        planner_seconds = 0.0

        if mode_clean == "vector_only":
            plan = QueryPlan(question=clean_question, intent=QueryIntent.LOCAL)
            vector_ev = vector_retriever.retrieve(clean_question, plan, kb_id=self.kb_id, scope_id=scope_id)
            retrieval_result = RetrievalResult(
                kb_id=self.kb_id,
                question=clean_question,
                query_plan=plan,
                evidence_list=vector_ev,
                community_reports=[],
                abstained=(len(vector_ev) == 0),
                abstention_reason="No relevant text chunks found." if len(vector_ev) == 0 else None,
                route_diagnostics={"vector_candidates": len(vector_ev), "graph_candidates": 0,
                                   "community_candidates": 0, "seed_entity_count": 0,
                                   "invalid_relation_count": 0, "evidence_after_merge": len(vector_ev),
                                   "evidence_after_reranking": len(vector_ev), "route_errors": {}},
            )

        elif mode_clean == "graph_only":
            planner_started = time.perf_counter()
            plan = query_planner.plan(clean_question)
            planner_seconds = time.perf_counter() - planner_started
            seed_res = seed_retriever.resolve_seeds(plan.seed_entities)
            graph_ev = graph_retriever.retrieve(clean_question, plan, kb_id=self.kb_id, scope_id=scope_id)
            retrieval_result = RetrievalResult(
                kb_id=self.kb_id,
                question=clean_question,
                query_plan=plan,
                resolved_seed_entity_ids=seed_res.resolved_entity_ids,
                evidence_list=graph_ev,
                community_reports=[],
                abstained=(len(graph_ev) == 0),
                abstention_reason="No relevant graph traversal paths found." if len(graph_ev) == 0 else None,
                route_diagnostics={"vector_candidates": 0, "graph_candidates": len(graph_ev),
                                   "community_candidates": 0, "seed_entity_count": len(seed_res.resolved_entity_ids),
                                   "invalid_relation_count": len(plan.invalid_relationship_types),
                                   "evidence_after_merge": len(graph_ev), "evidence_after_reranking": len(graph_ev),
                                   "graph_details": dict(graph_retriever.last_diagnostics),
                                   "route_errors": {}},
            )

        elif mode_clean == "community_aware":
            planner_started = time.perf_counter()
            try:
                plan = query_planner.plan(clean_question)
            except Exception as exc:
                logger.warning("Query planner failed; using safe mixed plan: %s", exc)
                plan = QueryPlan(question=clean_question, intent=QueryIntent.MIXED)
            planner_seconds = time.perf_counter() - planner_started
            plan.intent = QueryIntent.MIXED  # Force mixed retrieval incorporating community reports
            retrieval_result = orchestrator.retrieve_hybrid(clean_question, plan, kb_id=self.kb_id, scope_id=scope_id)

        elif mode_clean == "hybrid":
            planner_started = time.perf_counter()
            try:
                plan = query_planner.plan(clean_question)
            except Exception as exc:
                logger.warning("Query planner failed; using safe mixed plan: %s", exc)
                plan = QueryPlan(question=clean_question, intent=QueryIntent.MIXED)
            planner_seconds = time.perf_counter() - planner_started
            retrieval_result = orchestrator.retrieve_hybrid(clean_question, plan, kb_id=self.kb_id, scope_id=scope_id)

        else:
            raise ValueError(
                f"Unknown retrieval mode: '{mode}'. Allowed modes: 'hybrid', 'vector_only', 'graph_only', 'community_aware'"
            )

        answer_started = time.perf_counter()
        structured_answer = answer_generator.generate_structured(retrieval_result)
        answer_seconds = time.perf_counter() - answer_started
        retrieval_result.route_diagnostics["query_planning_seconds"] = planner_seconds
        retrieval_result.route_diagnostics["answer_generation_seconds"] = answer_seconds
        answer = structured_answer.answer
        retrieval_result.generated_answer = answer
        retrieval_result.canonical_answer = structured_answer.canonical_answer
        retrieval_result.answer_abstained = structured_answer.abstained
        retrieval_result.abstained = structured_answer.abstained
        retrieval_result.abstention_reason = structured_answer.abstention_reason
        retrieval_result.cited_chunk_ids = structured_answer.cited_chunk_ids
        retrieval_result.citation_references = structured_answer.citation_references
        return answer, retrieval_result
