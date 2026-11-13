"""Live integration validation harness for Hybrid GraphRAG on an existing knowledge base.

Executes four retrieval routes (vector_only, graph_only, hybrid, community_aware)
under a shared unified QueryPlan without reindexing, rebuilding, or silent overwrites.
Answer generation is decoupled and executed only when explicitly requested.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.configuration import load_engine_config
from src.core.interfaces import EmbeddingProvider
from src.core.models import (
    Evidence,
    QueryIntent,
    QueryPlan,
    RetrievalResult,
    SeedEntitySpec,
)
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.engine import HybridGraphRAGEngine
from src.generation.answer_generator import OllamaAnswerGenerator
from src.retrieval.community_retriever import (
    CommunityRetriever,
    HybridRetrieverOrchestrator,
)
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.query_planner import OllamaQueryPlanner
from src.retrieval.reranker import RRFReranker
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.retrieval.vector_retriever import VectorEvidenceRetriever
from run_hotpot import (
    extract_canonical_candidate,
    normalize_answer,
    supporting_metrics,
    token_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_validation")


class InstrumentedEmbeddingService(EmbeddingProvider):
    """Transparent pass-through instrumentation layer tracking invocations and inferred request attempts."""

    def __init__(self, delegate: EmbeddingProvider) -> None:
        self.delegate = delegate
        self.embed_text_invocations = 0
        self.embed_batch_invocations = 0
        self.delegated_provider_calls = 0
        self.ollama_request_attempts_inferred = 0
        self.provider_internal_cache_hits = 0

    def _is_live_ollama_provider(self) -> bool:
        return (
            isinstance(self.delegate, OllamaEmbeddingService)
            and getattr(self.delegate, "custom_embedder", None) is None
        )

    def embed_text(self, text: str) -> List[float]:
        self.embed_text_invocations += 1
        self.delegated_provider_calls += 1
        if self._is_live_ollama_provider() and text:
            self.ollama_request_attempts_inferred += 1
        return self.delegate.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.embed_batch_invocations += 1
        self.delegated_provider_calls += 1
        if self._is_live_ollama_provider() and texts:
            self.ollama_request_attempts_inferred += 1
        return self.delegate.embed_batch(texts)

    def snapshot(self) -> Dict[str, int]:
        return {
            "embed_text_invocations": self.embed_text_invocations,
            "embed_batch_invocations": self.embed_batch_invocations,
            "total_wrapper_invocations": self.embed_text_invocations + self.embed_batch_invocations,
            "delegated_provider_calls": self.delegated_provider_calls,
            "ollama_request_attempts_inferred": self.ollama_request_attempts_inferred,
            "provider_internal_cache_hits": self.provider_internal_cache_hits,
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live integration validation on an existing knowledge base")
    p.add_argument(
        "--config",
        default="knowledge_bases/hotpotqa_eval/hotpot_full_5a8b57f25542995d1e6f1371/config_snapshot.yaml",
        help="Path to existing KB config snapshot (must exist)",
    )
    p.add_argument(
        "--sample-json",
        default="adapters/datasets/hotpotqa/hotpot_smoke_20.json",
        help="Path to HotpotQA sample dataset",
    )
    p.add_argument(
        "--qid",
        default="5a8b57f25542995d1e6f1371",
        help="Question ID to evaluate",
    )
    p.add_argument(
        "--output",
        default="experiments/hotpotqa/validation_post_repair.json",
        help="Destination path for new validation results (must NOT exist)",
    )
    p.add_argument(
        "--planner-ontology",
        default="configs/domains/academic.yaml",
        help="Path to domain YAML containing the authoritative ontology for query planning (must contain NATIONALITY)",
    )
    p.add_argument(
        "--generate-answers",
        action="store_true",
        help="Separately invoke LLM answer generation after retrieval validation completes",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Preflight Protection 1: Refuse Output File if Already Exists BEFORE Loading Anything ---
    out_path = Path(args.output).resolve()
    if out_path.exists():
        logger.error(
            "PREFLIGHT REFUSAL: Target output file '%s' already exists. "
            "Aborting immediately prior to loading stores or initiating inference.",
            out_path,
        )
        raise FileExistsError(f"Target output file '{out_path}' already exists.")

    # --- Preflight Protection 2: Verify Config File Exists ---
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logger.error("Configuration file '%s' not found.", config_path)
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    # --- Preflight Protection 3: Verify Planner Ontology Exists and Passes Strict Validation ---
    planner_ontology_path = Path(args.planner_ontology).resolve()
    if not planner_ontology_path.exists():
        logger.error("Planner ontology file '%s' not found.", planner_ontology_path)
        raise FileNotFoundError(f"Planner ontology file '{planner_ontology_path}' not found.")

    planner_cfg = load_engine_config(planner_ontology_path)
    planner_ontology = planner_cfg.ontology

    if "NATIONALITY" not in planner_ontology.relationship_types:
        logger.error("Planner ontology '%s' is missing required relationship type 'NATIONALITY'.", planner_ontology_path)
        raise ValueError(
            f"Planner ontology '{planner_ontology_path}' validation failed: missing required relationship type 'NATIONALITY'."
        )

    nat_constraint = planner_ontology.relationship_types["NATIONALITY"]
    if "PERSON" not in nat_constraint.allowed_source_types:
        logger.error("Planner ontology 'NATIONALITY' allowed_source_types does not include 'PERSON'.")
        raise ValueError(
            f"Planner ontology '{planner_ontology_path}' validation failed: 'NATIONALITY' allowed_source_types must include 'PERSON'."
        )

    if "COUNTRY" not in nat_constraint.allowed_target_types or "LOCATION" not in nat_constraint.allowed_target_types:
        logger.error("Planner ontology 'NATIONALITY' allowed_target_types does not include both 'COUNTRY' and 'LOCATION'.")
        raise ValueError(
            f"Planner ontology '{planner_ontology_path}' validation failed: 'NATIONALITY' allowed_target_types must include 'COUNTRY' and 'LOCATION'."
        )

    if "BORN_IN" not in planner_ontology.relationship_types:
        logger.error("Planner ontology '%s' is missing distinct relationship type 'BORN_IN'.", planner_ontology_path)
        raise ValueError(
            f"Planner ontology '{planner_ontology_path}' validation failed: missing distinct relationship type 'BORN_IN'."
        )

    logger.info("Planner ontology '%s' validated successfully (includes NATIONALITY and BORN_IN).", planner_ontology_path)

    # 1. Load engine configuration (Zero writes)
    logger.info("Loading engine configuration from %s...", config_path)
    cfg = load_engine_config(config_path)
    engine = HybridGraphRAGEngine(config=cfg)

    # 2. Verify existing KB stores are populated
    doc_count = len(engine.doc_store.load_documents())
    chunk_count = len(engine.doc_store.load_chunks())
    entity_count = len(engine.doc_store.load_canonical_entities())
    rel_count = len(engine.doc_store.load_relationships())
    graph = engine.get_or_load_graph()
    chroma_chunks = engine.vector_store.chunk_collection.count()
    chroma_reports = engine.vector_store.community_collection.count()

    logger.info(
        "Connected to existing KB '%s': docs=%d, chunks=%d, entities=%d, rels=%d, graph_v=%d, graph_e=%d, chroma_chunks=%d, chroma_reports=%d",
        cfg.kb_id,
        doc_count,
        chunk_count,
        entity_count,
        rel_count,
        graph.vcount(),
        graph.ecount(),
        chroma_chunks,
        chroma_reports,
    )

    if chunk_count == 0 or graph.vcount() == 0 or chroma_chunks == 0:
        logger.error("Knowledge base '%s' appears empty or unindexed. Aborting.", cfg.kb_id)
        sys.exit(1)

    # 3. Load sample question and gold benchmark data
    sample_file = Path(args.sample_json).resolve()
    samples = json.loads(sample_file.read_text(encoding="utf-8"))
    sample = next((s for s in samples if str(s.get("_id")) == args.qid), None)
    if not sample:
        logger.error("Question ID '%s' not found in %s.", args.qid, sample_file)
        sys.exit(1)

    question = sample["question"]
    gold_answer = str(sample.get("answer", ""))
    contexts = sample.get("context", [])

    logger.info("Question: %s", question)
    logger.info("Gold Answer: %s", gold_answer)

    # 4. Instrument embedding service to account for every embedding call
    instrumented_embedder = InstrumentedEmbeddingService(engine.embedding_service)

    # 5. Instantiate retrieval sub-components directly from engine state
    seed_retriever = SeedEntityRetriever(
        doc_store=engine.doc_store,
        vector_store=engine.vector_store,
        embedding_provider=instrumented_embedder,
        similarity_threshold=cfg.resolution.similarity_candidate_threshold,
    )

    # In-memory memoization cache for seed resolution to avoid duplicate resolution across routes
    _orig_resolve = seed_retriever.resolve_seeds
    _seed_cache: Dict[Tuple[Tuple[str, Optional[str]], ...], SeedResolutionResult] = {}

    def memoized_resolve(specs: List[SeedEntitySpec]) -> SeedResolutionResult:
        cache_key = tuple((s.name, s.expected_type) for s in specs)
        if cache_key not in _seed_cache:
            _seed_cache[cache_key] = _orig_resolve(specs)
        return _seed_cache[cache_key]

    seed_retriever.resolve_seeds = memoized_resolve

    graph_retriever = GraphEvidenceRetriever(
        doc_store=engine.doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        max_candidates=cfg.retrieval.max_graph_candidates,
    )
    vector_retriever = VectorEvidenceRetriever(
        doc_store=engine.doc_store,
        vector_store=engine.vector_store,
        embedding_provider=instrumented_embedder,
        top_k=cfg.retrieval.vector_top_k,
    )
    community_retriever = CommunityRetriever(
        doc_store=engine.doc_store,
        vector_store=engine.vector_store,
        embedding_provider=instrumented_embedder,
        community_top_k=cfg.retrieval.community_top_k,
    )
    reranker = RRFReranker(rrf_k=cfg.retrieval.rrf_k)
    orchestrator = HybridRetrieverOrchestrator(
        doc_store=engine.doc_store,
        vector_store=engine.vector_store,
        embedding_provider=instrumented_embedder,
        seed_retriever=seed_retriever,
        graph_retriever=graph_retriever,
        vector_retriever=vector_retriever,
        community_retriever=community_retriever,
        reranker=reranker,
        final_context_chunks=cfg.retrieval.final_context_chunks,
    )

    # --- Step 6: Exactly One Query Plan Call ---
    t_validation_start = time.perf_counter()
    logger.info("Generating unified QueryPlan (exactly once) using planner ontology from %s...", planner_ontology_path)
    t_plan_start = time.perf_counter()
    query_planner = OllamaQueryPlanner(ontology=planner_ontology, llm_client=engine.llm_client)
    shared_plan: QueryPlan = query_planner.plan(question)
    t_plan_seconds = time.perf_counter() - t_plan_start

    # Eagerly resolve and cache seed entities once, measuring seed resolution latency
    t_seed_start = time.perf_counter()
    seed_res = seed_retriever.resolve_seeds(shared_plan.seed_entities)
    t_seed_seconds = time.perf_counter() - t_seed_start
    resolved_seed_ids = list(seed_res.resolved_entity_ids)

    logger.info(
        "Unified QueryPlan generated in %.2fs: intent=%s, seeds=%s -> resolved_ids=%s, target=%s, rel_path=%s",
        t_plan_seconds,
        shared_plan.intent.value,
        [s.name for s in shared_plan.seed_entities],
        resolved_seed_ids,
        shared_plan.target_type,
        shared_plan.relationship_path,
    )

    # Build plan variants preserving exact seed entities
    hybrid_plan = shared_plan.model_copy(update={"intent": QueryIntent.LOCAL})
    comm_plan = shared_plan.model_copy(update={"intent": QueryIntent.MIXED})

    output_payload: Dict[str, Any] = {
        "metadata": {
            "kb_id": cfg.kb_id,
            "kb_config_path": str(config_path),
            "planner_ontology_path": str(planner_ontology_path),
            "planner_ontology_relationship_types": sorted(list(planner_ontology.relationship_types.keys())),
            "persisted_kb_ontology_relationship_types": sorted(list(cfg.ontology.relationship_types.keys())),
            "planner_ontology_differs_from_persisted_kb": (
                sorted(list(planner_ontology.relationship_types.keys()))
                != sorted(list(cfg.ontology.relationship_types.keys()))
            ),
            "graph_edge_audit_note": (
                "The existing knowledge graph contains exactly zero NATIONALITY edges, as established by audit. "
                "Query planning uses the updated ontology to permit the query path, but graph retrieval relies on "
                "direct seed provenance (hop 0) rather than multi-hop nationality graph traversal."
            ),
            "question_id": args.qid,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answers_included": args.generate_answers,
            "shared_preparation_timing": {
                "query_planning_seconds": t_plan_seconds,
                "seed_resolution_seconds": t_seed_seconds,
                "total_shared_prep_seconds": t_plan_seconds + t_seed_seconds,
                "shared_prep_execution_count": 1,
                "note": "Shared preparation (query planning and seed resolution) is executed exactly once and reused across routes, not executed independently four times.",
            },
            "shared_query_plan": shared_plan.model_dump(mode="json"),
            "resolved_seed_entity_ids": resolved_seed_ids,
            "ambiguous_seeds": {k: list(v) for k, v in seed_res.ambiguous_seeds.items()},
        },
        "routes": {},
    }

    # --- Step 7: True Retrieval-Only Execution for All 4 Routes ---
    routes_results: Dict[str, RetrievalResult] = {}
    route_timings: Dict[str, Dict[str, float]] = {}
    route_embedding_stats: Dict[str, Dict[str, int]] = {}

    # Route 1: vector_only
    logger.info("Executing Route 1: vector_only (retrieval-only)...")
    emb_before = instrumented_embedder.snapshot()
    t0 = time.perf_counter()
    vector_ev = vector_retriever.retrieve(question, shared_plan, kb_id=cfg.kb_id)
    t_vec = time.perf_counter() - t0
    emb_after = instrumented_embedder.snapshot()

    route_timings["vector_only"] = {
        "query_planning_seconds": t_plan_seconds,
        "seed_resolution_seconds": 0.0,
        "route_retrieval_seconds": t_vec,
        "route_plus_shared_prep_seconds": t_vec + t_plan_seconds,
    }
    route_embedding_stats["vector_only"] = {
        k: emb_after[k] - emb_before[k] for k in emb_before
    }
    routes_results["vector_only"] = RetrievalResult(
        kb_id=cfg.kb_id,
        question=question,
        query_plan=shared_plan,
        evidence_list=vector_ev,
        community_reports=[],
        abstained=(len(vector_ev) == 0),
        abstention_reason="No relevant text chunks found." if len(vector_ev) == 0 else None,
        route_diagnostics={
            "vector_candidates": len(vector_ev),
            "graph_candidates": 0,
            "community_candidates": 0,
            "seed_entity_count": 0,
            "invalid_relation_count": 0,
            "evidence_after_merge": len(vector_ev),
            "evidence_after_reranking": len(vector_ev),
            "route_seconds": t_vec,
            "route_errors": {},
        },
    )

    # Route 2: graph_only
    logger.info("Executing Route 2: graph_only (retrieval-only)...")
    emb_before = instrumented_embedder.snapshot()
    t0 = time.perf_counter()
    graph_ev = graph_retriever.retrieve(question, shared_plan, kb_id=cfg.kb_id)
    t_graph = time.perf_counter() - t0
    emb_after = instrumented_embedder.snapshot()

    route_timings["graph_only"] = {
        "query_planning_seconds": t_plan_seconds,
        "seed_resolution_seconds": t_seed_seconds,
        "route_retrieval_seconds": t_graph,
        "route_plus_shared_prep_seconds": t_graph + t_plan_seconds + t_seed_seconds,
    }
    route_embedding_stats["graph_only"] = {
        k: emb_after[k] - emb_before[k] for k in emb_before
    }
    routes_results["graph_only"] = RetrievalResult(
        kb_id=cfg.kb_id,
        question=question,
        query_plan=shared_plan,
        resolved_seed_entity_ids=resolved_seed_ids,
        evidence_list=graph_ev,
        community_reports=[],
        abstained=(len(graph_ev) == 0),
        abstention_reason="No relevant graph traversal paths found." if len(graph_ev) == 0 else None,
        route_diagnostics={
            "vector_candidates": 0,
            "graph_candidates": len(graph_ev),
            "community_candidates": 0,
            "seed_entity_count": len(resolved_seed_ids),
            "invalid_relation_count": len(shared_plan.invalid_relationship_types),
            "evidence_after_merge": len(graph_ev),
            "evidence_after_reranking": len(graph_ev),
            "graph_details": dict(getattr(graph_retriever, "last_diagnostics", {})),
            "route_seconds": t_graph,
            "route_errors": {},
        },
    )

    # Route 3: hybrid (LOCAL intent dispatches graph + vector + RRF)
    logger.info("Executing Route 3: hybrid (retrieval-only)...")
    emb_before = instrumented_embedder.snapshot()
    t0 = time.perf_counter()
    hybrid_res = orchestrator.retrieve_hybrid(question, hybrid_plan, kb_id=cfg.kb_id)
    t_hybrid = time.perf_counter() - t0
    emb_after = instrumented_embedder.snapshot()

    route_timings["hybrid"] = {
        "query_planning_seconds": t_plan_seconds,
        "seed_resolution_seconds": t_seed_seconds,
        "route_retrieval_seconds": t_hybrid,
        "route_plus_shared_prep_seconds": t_hybrid + t_plan_seconds + t_seed_seconds,
    }
    route_embedding_stats["hybrid"] = {
        k: emb_after[k] - emb_before[k] for k in emb_before
    }
    hybrid_res.route_diagnostics["route_seconds"] = t_hybrid
    routes_results["hybrid"] = hybrid_res

    # Route 4: community_aware (MIXED intent dispatches graph + vector + community + RRF)
    logger.info("Executing Route 4: community_aware (retrieval-only)...")
    emb_before = instrumented_embedder.snapshot()
    t0 = time.perf_counter()
    comm_res = orchestrator.retrieve_hybrid(question, comm_plan, kb_id=cfg.kb_id)
    t_comm = time.perf_counter() - t0
    emb_after = instrumented_embedder.snapshot()

    route_timings["community_aware"] = {
        "query_planning_seconds": t_plan_seconds,
        "seed_resolution_seconds": t_seed_seconds,
        "route_retrieval_seconds": t_comm,
        "route_plus_shared_prep_seconds": t_comm + t_plan_seconds + t_seed_seconds,
    }
    route_embedding_stats["community_aware"] = {
        k: emb_after[k] - emb_before[k] for k in emb_before
    }
    comm_res.route_diagnostics["route_seconds"] = t_comm
    routes_results["community_aware"] = comm_res

    # --- Step 8: Optional Separate Answer Generation ---
    answer_generator: Optional[OllamaAnswerGenerator] = None
    if args.generate_answers:
        logger.info("Initializing AnswerGenerator for separate generation...")
        context_builder = ContextBuilder(max_chars=cfg.retrieval.max_context_chars)
        answer_generator = OllamaAnswerGenerator(llm_client=engine.llm_client, context_builder=context_builder)

    # Compute route contributions across routes
    route_chunk_ids: Dict[str, Set[str]] = {
        route: {ev.chunk_id for ev in res.evidence_list}
        for route, res in routes_results.items()
    }
    v_chunks = route_chunk_ids.get("vector_only", set())
    g_chunks = route_chunk_ids.get("graph_only", set())
    h_chunks = route_chunk_ids.get("hybrid", set())
    c_chunks = route_chunk_ids.get("community_aware", set())

    output_payload["route_contributions"] = {
        "vector_chunks": sorted(list(v_chunks)),
        "graph_chunks": sorted(list(g_chunks)),
        "hybrid_chunks": sorted(list(h_chunks)),
        "community_aware_chunks": sorted(list(c_chunks)),
        "graph_unique_vs_vector": sorted(list(g_chunks - v_chunks)),
        "hybrid_unique_vs_vector": sorted(list(h_chunks - v_chunks)),
        "community_unique_vs_hybrid": sorted(list(c_chunks - h_chunks)),
        "shared_graph_vector": sorted(list(g_chunks & v_chunks)),
        "final_hybrid_chunks": sorted(list(h_chunks)),
    }

    for mode_name, result in routes_results.items():
        logger.info("Computing metrics and provenance for route '%s'...", mode_name)
        answer = None
        canonical_pred = None
        em = None
        f1 = None
        t_gen_sec = 0.0

        if args.generate_answers and answer_generator is not None:
            t_gen_start = time.perf_counter()
            structured_ans = answer_generator.generate_structured(result)
            t_gen_sec = time.perf_counter() - t_gen_start
            answer = structured_ans.answer
            canonical_pred = extract_canonical_candidate(
                answer, getattr(result, "canonical_answer", None), question
            )
            em = normalize_answer(canonical_pred) == normalize_answer(gold_answer)
            f1 = token_f1(canonical_pred, gold_answer)
            result.route_diagnostics["answer_generation_seconds"] = t_gen_sec

        rec_2 = supporting_metrics(result.evidence_list, sample, contexts, 2)
        rec_5 = supporting_metrics(result.evidence_list, sample, contexts, 5)

        route_record: Dict[str, Any] = {
            "timing": {
                "query_planning_seconds": route_timings[mode_name]["query_planning_seconds"],
                "seed_resolution_seconds": route_timings[mode_name]["seed_resolution_seconds"],
                "route_retrieval_seconds": route_timings[mode_name]["route_retrieval_seconds"],
                "route_plus_shared_prep_seconds": route_timings[mode_name]["route_plus_shared_prep_seconds"],
                "overall_validation_elapsed_seconds": 0.0,
                "answer_generation_seconds": t_gen_sec if args.generate_answers else None,
            },
            "embedding_accounting": route_embedding_stats[mode_name],
            "abstained": result.abstained,
            "abstention_reason": result.abstention_reason,
            "resolved_seed_entity_ids": result.resolved_seed_entity_ids,
            "evidence_count": len(result.evidence_list),
            "evidence": [e.model_dump(mode="json") for e in result.evidence_list],
            "community_report_count": len(result.community_reports),
            "community_reports": [r.model_dump(mode="json") for r in result.community_reports],
            "diagnostics": result.route_diagnostics,
            "retrieval_metrics": {
                "recall_at_2": rec_2,
                "recall_at_5": rec_5,
            },
            "answer": answer,
            "canonical_answer": getattr(result, "canonical_answer", None) or canonical_pred,
            "answer_metrics": {
                "exact_match": em,
                "token_f1": f1,
            } if args.generate_answers else None,
        }
        output_payload["routes"][mode_name] = route_record

    # Calculate overall validation elapsed time and populate in timing fields
    overall_validation_elapsed_seconds = time.perf_counter() - t_validation_start
    output_payload["overall_validation_elapsed_seconds"] = overall_validation_elapsed_seconds
    output_payload["metadata"]["shared_preparation_timing"]["overall_validation_elapsed_seconds"] = (
        overall_validation_elapsed_seconds
    )
    for mode_name in output_payload["routes"]:
        output_payload["routes"][mode_name]["timing"]["overall_validation_elapsed_seconds"] = (
            overall_validation_elapsed_seconds
        )

    # Total embedding stats across the entire run
    output_payload["total_embedding_accounting"] = instrumented_embedder.snapshot()

    # --- Step 9: Safe Exclusive Creation (Preflight Guard + OS Exclusive Create) ---
    logger.info("Writing results exclusively to %s...", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Validation run completed successfully. Output saved to %s.", out_path)


if __name__ == "__main__":
    main()
