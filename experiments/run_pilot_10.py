#!/usr/bin/env python3
"""Autonomous execution script for the HotpotQA 10-Question Pilot.

Evaluates all 10 questions from experiments/hotpotqa/pilot_10_manifest.json:
- 5 comparison questions, 5 bridge questions.
- Question 2 reuses the existing validated KB (knowledge_bases/hotpotqa_eval/hotpot_full_5a8b57f25542995d1e6f1371).
- Questions 1, 3-10 are indexed into separate, isolated KBs under knowledge_bases/hotpotqa_pilot/<question_id>/.
- Exactly 1 hybrid query and 1 evidence-grounded answer generation per question.
- Saves progress immediately after each indexing and query step.
- Resumable: resumes from the last completed stage if interrupted.
- Produces:
    - experiments/hotpotqa/pilot_10_run_state.json
    - experiments/hotpotqa/pilot_10_run.log
    - experiments/hotpotqa/pilot_10_results.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hybrid_graphrag import HybridGraphRAG, QueryResult
from src.core.configuration import load_engine_config

# Setup paths
MANIFEST_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_manifest.json"
RUN_STATE_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_run_state.json"
RUN_LOG_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_run.log"
RESULTS_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_results.json"
INPUTS_BASE_DIR = REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "pilot"
PILOT_KB_ROOT = REPO_ROOT / "knowledge_bases" / "hotpotqa_pilot"
REUSED_KB_PATH = REPO_ROOT / "knowledge_bases" / "hotpotqa_eval" / "hotpot_full_5a8b57f25542995d1e6f1371"

# Setup logger
RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("hotpot_pilot")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers on resume
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_fmt)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(str(RUN_LOG_PATH), mode="a", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    f_handler.setFormatter(f_fmt)
    logger.addHandler(f_handler)


def safe_id(value: str) -> str:
    """Create a filesystem-safe identifier from a document title."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)[:48] or hashlib.sha1(value.encode()).hexdigest()[:12]


def normalize_answer(text: str) -> str:
    """Standard HotpotQA answer normalization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def token_f1(prediction: str, gold: str) -> float:
    """Standard HotpotQA token-level F1 score."""
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = sum((__import__("collections").Counter(p_tokens) & __import__("collections").Counter(g_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def exact_match(prediction: str, gold: str) -> int:
    """Standard HotpotQA Exact Match metric (0 or 1)."""
    return 1 if normalize_answer(prediction) == normalize_answer(gold) else 0


def atomic_save_json(path: Path, data: Any) -> None:
    """Save data to path atomically using a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    temp_path.replace(path)


def load_run_state() -> Dict[str, Any]:
    """Load existing run state or initialize a fresh state dictionary."""
    if RUN_STATE_PATH.exists():
        try:
            with open(RUN_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
                logger.info("Resuming pilot from existing run state: %s", RUN_STATE_PATH)
                return state
        except Exception as e:
            logger.warning("Failed to read existing run state (%s); starting fresh.", e)

    state = {
        "run_id": f"pilot_10_{int(time.time())}",
        "manifest_path": str(MANIFEST_PATH),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_count": 0,
        "failed_count": 0,
        "total_questions": 10,
        "questions": {},
    }
    atomic_save_json(RUN_STATE_PATH, state)
    return state


def update_run_state(state: Dict[str, Any]) -> None:
    """Persist run state to disk atomically."""
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed = sum(1 for q in state["questions"].values() if q.get("status") == "completed")
    failed = sum(1 for q in state["questions"].values() if q.get("status") == "failed")
    state["completed_count"] = completed
    state["failed_count"] = failed
    atomic_save_json(RUN_STATE_PATH, state)


def prepare_raw_documents(q: Dict[str, Any], raw_dir: Path) -> Dict[str, str]:
    """Write candidate context documents as individual .txt files for PDFLoader.

    Returns:
        Mapping of filename to original document title.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    title_by_filename: Dict[str, str] = {}

    for rank, doc in enumerate(q.get("candidate_documents", [])):
        title = doc["title"]
        clean_title = safe_id(title)
        filename = f"ctx_{rank:03d}_{clean_title}.txt"
        file_path = raw_dir / filename
        title_by_filename[filename] = title

        # Check if already written with same content
        full_text = doc.get("full_text") or " ".join(doc.get("sentences", [])).strip()
        expected_content = f"Title: {title}\n{full_text}\n"

        if file_path.exists():
            try:
                if file_path.read_text(encoding="utf-8") == expected_content:
                    continue
            except Exception:
                pass

        file_path.write_text(expected_content, encoding="utf-8")

    return title_by_filename


def verify_kb_persistence(kb_dir: Path) -> Tuple[bool, Optional[str]]:
    """Verify that a knowledge base contains all required persisted graph and document store files."""
    required_files = [
        kb_dir / "config_snapshot.yaml",
        kb_dir / "graph" / "entities.jsonl",
        kb_dir / "graph" / "relationships.jsonl",
        kb_dir / "graph" / "communities.jsonl",
        kb_dir / "graph" / "community_reports.jsonl",
        kb_dir / "data" / "processed" / "chunks.jsonl",
        kb_dir / "data" / "processed" / "documents.jsonl",
    ]
    for p in required_files:
        if not p.exists() or p.stat().st_size == 0:
            return False, f"Missing or empty required KB file: {p}"
    return True, None


def compute_retrieval_and_answer_metrics(
    query_result: QueryResult,
    q: Dict[str, Any],
    title_by_filename: Dict[str, str],
) -> Dict[str, Any]:
    """Compute exact match, token F1, recall@K, and citation metrics."""
    gold_answer = q.get("gold_answer", "")
    gen_answer = query_result.answer
    canon_answer = query_result.canonical_answer or ""

    em_canon = exact_match(canon_answer, gold_answer)
    f1_canon = token_f1(canon_answer, gold_answer)
    em_gen = exact_match(gen_answer, gold_answer)
    f1_gen = token_f1(gen_answer, gold_answer)

    # Retrieval metrics
    gold_facts = q.get("supporting_facts", [])
    gold_titles = sorted(list({fact[0] for fact in gold_facts if len(fact) >= 1}))

    retrieved_titles: List[str] = []
    seen_titles: Set[str] = set()

    if query_result.raw_result and query_result.raw_result.evidence_list:
        for ev in query_result.raw_result.evidence_list:
            mapped_title = title_by_filename.get(ev.filename, ev.filename)
            if mapped_title not in seen_titles:
                retrieved_titles.append(mapped_title)
                seen_titles.add(mapped_title)

    retrieved_top_2 = set(retrieved_titles[:2])
    retrieved_top_5 = set(retrieved_titles[:5])
    gold_set = set(gold_titles)

    recall_at_2 = (len(gold_set & retrieved_top_2) / len(gold_set)) if gold_set else 1.0
    recall_at_5 = (len(gold_set & retrieved_top_5) / len(gold_set)) if gold_set else 1.0
    complete_at_2 = bool(gold_set and gold_set <= retrieved_top_2)
    complete_at_5 = bool(gold_set and gold_set <= retrieved_top_5)

    # Citation metrics
    citations = query_result.citations
    total_citations = len(citations)
    valid_citations = sum(1 for c in citations if c.valid)
    citation_precision = (valid_citations / total_citations) if total_citations > 0 else None

    cited_titles: Set[str] = set()
    for c in citations:
        mapped_t = title_by_filename.get(c.filename, c.filename)
        cited_titles.add(mapped_t)

    citation_coverage = (len(gold_set & cited_titles) / len(gold_set)) if gold_set else 1.0

    return {
        "answer_metrics": {
            "exact_match": em_canon,
            "token_f1": f1_canon,
            "generated_exact_match": em_gen,
            "generated_token_f1": f1_gen,
            "abstained": query_result.abstained,
            "abstention_reason": query_result.abstention_reason,
        },
        "retrieval_metrics": {
            "gold_supporting_titles": gold_titles,
            "retrieved_document_titles": retrieved_titles,
            "recall_at_2": recall_at_2,
            "recall_at_5": recall_at_5,
            "complete_supporting_documents_at_2": complete_at_2,
            "complete_supporting_documents_at_5": complete_at_5,
        },
        "citation_metrics": {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "has_citations": (total_citations > 0),
            "citation_precision": citation_precision,
            "citation_validity_rate": citation_precision if citation_precision is not None else 1.0,
            "citation_gold_coverage": citation_coverage,
            "cited_chunk_ids": query_result.cited_chunk_ids,
            "cited_document_titles": sorted(list(cited_titles)),
        },
    }


def execute_question(
    q_idx: int,
    q: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute complete indexing and querying workflow for a single question."""
    qid = q["question_id"]
    qtype = q["question_type"]
    question_text = q["question"]
    gold_answer = q["gold_answer"]
    is_q2_control = (qid == "5a8b57f25542995d1e6f1371")

    logger.info("================================================================================")
    logger.info("[%d/10] Starting Question: %s (Type: %s, Control: %s)", q_idx, qid, qtype, is_q2_control)
    logger.info("Question: %s", question_text)
    logger.info("Gold Answer: %s", gold_answer)

    q_record = state["questions"].get(qid, {
        "question_id": qid,
        "question_index": q_idx,
        "question_type": qtype,
        "question": question_text,
        "gold_answer": gold_answer,
        "is_regression_control": is_q2_control,
        "status": "pending",
        "attempts": 0,
        "error": None,
    })
    state["questions"][qid] = q_record

    # Check if question is already fully completed
    if q_record.get("status") == "completed" and "metrics" in q_record:
        logger.info("[%d/10] Question %s is already completed. Skipping.", q_idx, qid)
        return q_record

    # Prepare input documents mapping
    raw_dir = (REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "full" / qid) if is_q2_control else (INPUTS_BASE_DIR / qid)
    title_by_filename = prepare_raw_documents(q, raw_dir)

    # --------------------------------------------------------------------------
    # Step 1: Ingestion / Knowledge Base Setup
    # --------------------------------------------------------------------------
    kb_path: Path
    engine: Optional[HybridGraphRAG] = None
    indexing_duration = 0.0
    indexing_stats: Dict[str, Any] = {}
    indexing_llm_requests = 0

    if is_q2_control:
        # Question 2: Reuse existing validated KB
        kb_path = REUSED_KB_PATH
        logger.info("[%d/10] Reusing existing validated KB at %s", q_idx, kb_path)
        valid, reason = verify_kb_persistence(kb_path)
        if not valid:
            raise RuntimeError(f"Reused KB failed verification: {reason}")
        q_record["indexing_stats"] = {"reused_existing_kb": True, "indexing_seconds": 0.0}
        q_record["status"] = "indexed"
        update_run_state(state)
    else:
        # Questions 1, 3-10: Ingest into isolated KB
        kb_path = PILOT_KB_ROOT / qid
        already_indexed = q_record.get("status") == "indexed"
        valid, _ = verify_kb_persistence(kb_path)

        if already_indexed and valid:
            logger.info("[%d/10] KB already indexed and verified at %s. Skipping re-indexing.", q_idx, kb_path)
            indexing_duration = q_record.get("indexing_stats", {}).get("indexing_seconds", 0.0)
        else:
            logger.info("[%d/10] Indexing 10 documents into isolated KB at %s...", q_idx, kb_path)
            q_record["status"] = "indexing"
            update_run_state(state)

            cfg = load_engine_config("configs/domains/academic.yaml", kb_id_override=qid)
            cfg.storage_root = str(PILOT_KB_ROOT.relative_to(REPO_ROOT))

            idx_start = time.perf_counter()
            engine_for_ingest = HybridGraphRAG.from_config(cfg, kb_id=qid)
            llm_before = getattr(engine_for_ingest._engine.llm_client, "stats", {}).get("requests", 0)

            try:
                ingest_res = engine_for_ingest.ingest(raw_dir, rebuild=True)
                indexing_duration = time.perf_counter() - idx_start
                llm_after = getattr(engine_for_ingest._engine.llm_client, "stats", {}).get("requests", 0)
                indexing_llm_requests = max(0, llm_after - llm_before)

                logger.info(
                    "[%d/10] Indexing completed in %.2fs (%.2f min). LLM requests: %d",
                    q_idx, indexing_duration, indexing_duration / 60.0, indexing_llm_requests,
                )
                indexing_stats = {
                    "indexing_seconds": indexing_duration,
                    "indexing_llm_requests": indexing_llm_requests,
                    "ingest_summary": ingest_res,
                }
            finally:
                engine_for_ingest.close()

            # Verify KB files on disk
            valid, reason = verify_kb_persistence(kb_path)
            if not valid:
                raise RuntimeError(f"Newly indexed KB failed verification: {reason}")

            q_record["indexing_stats"] = indexing_stats
            q_record["status"] = "indexed"
            update_run_state(state)

    # --------------------------------------------------------------------------
    # Step 2: Query Evaluation (Exactly 1 Hybrid Query)
    # --------------------------------------------------------------------------
    logger.info("[%d/10] Opening KB and executing hybrid query...", q_idx)
    q_record["status"] = "querying"
    update_run_state(state)

    query_start = time.perf_counter()
    with HybridGraphRAG.from_existing(kb_path) as engine:
        llm_before_query = getattr(engine._engine.llm_client, "stats", {}).get("requests", 0)
        query_result: QueryResult = engine.query(question=question_text, mode="hybrid")
        total_query_duration = time.perf_counter() - query_start
        llm_after_query = getattr(engine._engine.llm_client, "stats", {}).get("requests", 0)
        query_llm_requests = max(0, llm_after_query - llm_before_query)

    logger.info("[%d/10] Query finished in %.2fs. Answer: %s", q_idx, total_query_duration, query_result.answer)
    logger.info("[%d/10] Canonical Answer: %s | Gold: %s", q_idx, query_result.canonical_answer, gold_answer)

    # Compute metrics
    metrics = compute_retrieval_and_answer_metrics(query_result, q, title_by_filename)
    em = metrics["answer_metrics"]["exact_match"]
    f1 = metrics["answer_metrics"]["token_f1"]
    r2 = metrics["retrieval_metrics"]["recall_at_2"]
    r5 = metrics["retrieval_metrics"]["recall_at_5"]

    logger.info(
        "[%d/10] Metrics: EM=%d, F1=%.4f, Recall@2=%.2f, Recall@5=%.2f, Citations=%d/%d valid",
        q_idx, em, f1, r2, r5, metrics["citation_metrics"]["valid_citations"], metrics["citation_metrics"]["total_citations"],
    )

    q_record.update({
        "status": "completed",
        "kb_path": str(kb_path),
        "raw_docs_dir": str(raw_dir),
        "query_result": {
            "question": query_result.question,
            "answer": query_result.answer,
            "canonical_answer": query_result.canonical_answer,
            "abstained": query_result.abstained,
            "abstention_reason": query_result.abstention_reason,
            "cited_chunk_ids": query_result.cited_chunk_ids,
            "citations": [c.model_dump() for c in query_result.citations],
            "mode": query_result.mode,
            "latency_seconds": query_result.latency_seconds,
            "measured_total_query_seconds": total_query_duration,
            "timing_breakdown": query_result.timing,
            "query_llm_requests": query_llm_requests,
        },
        "metrics": metrics,
        "error": None,
    })
    update_run_state(state)
    return q_record


def compute_aggregate_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate benchmark metrics over the 9 benchmark questions and isolate Q2 control."""
    q_records = list(state["questions"].values())
    control_record = next((q for q in q_records if q.get("is_regression_control")), None)
    benchmark_records = [q for q in q_records if not q.get("is_regression_control")]

    def mean_val(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    def median_val(vals: List[float]) -> Optional[float]:
        return statistics.median(vals) if vals else None

    # Benchmark metrics
    completed_benchmarks = [q for q in benchmark_records if q.get("status") == "completed"]
    failed_benchmarks = [q for q in benchmark_records if q.get("status") == "failed"]

    # We evaluate all 9 benchmark questions in the denominator
    denom = len(benchmark_records)
    if denom == 0:
        denom = 1

    all_em = [float(q.get("metrics", {}).get("answer_metrics", {}).get("exact_match", 0)) for q in benchmark_records]
    all_f1 = [float(q.get("metrics", {}).get("answer_metrics", {}).get("token_f1", 0.0)) for q in benchmark_records]
    all_abstained = [float(q.get("metrics", {}).get("answer_metrics", {}).get("abstained", False)) for q in benchmark_records]

    all_r2 = [float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_2", 0.0)) for q in benchmark_records]
    all_r5 = [float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_5", 0.0)) for q in benchmark_records]
    all_comp2 = [float(q.get("metrics", {}).get("retrieval_metrics", {}).get("complete_supporting_documents_at_2", False)) for q in benchmark_records]
    all_comp5 = [float(q.get("metrics", {}).get("retrieval_metrics", {}).get("complete_supporting_documents_at_5", False)) for q in benchmark_records]

    # Citation precision among emitted citations
    records_with_citations = [q for q in benchmark_records if q.get("metrics", {}).get("citation_metrics", {}).get("total_citations", 0) > 0]
    total_emitted_citations = sum(q.get("metrics", {}).get("citation_metrics", {}).get("total_citations", 0) for q in benchmark_records)
    total_valid_citations = sum(q.get("metrics", {}).get("citation_metrics", {}).get("valid_citations", 0) for q in benchmark_records)
    citation_precision_among_emitted = (total_valid_citations / total_emitted_citations) if total_emitted_citations > 0 else 1.0

    # Citation coverage across questions
    citation_question_coverage = (len(records_with_citations) / len(benchmark_records)) if benchmark_records else 0.0

    # Gold supporting-document coverage across all questions
    all_cit_cov = [float(q.get("metrics", {}).get("citation_metrics", {}).get("citation_gold_coverage", 0.0)) for q in benchmark_records]
    mean_gold_doc_coverage = mean_val(all_cit_cov)

    # Conditioned on answered (non-abstained) questions
    answered_records = [q for q in benchmark_records if not q.get("metrics", {}).get("answer_metrics", {}).get("abstained", False)]
    answered_count = len(answered_records)

    answered_em = [float(q.get("metrics", {}).get("answer_metrics", {}).get("exact_match", 0)) for q in answered_records]
    answered_f1 = [float(q.get("metrics", {}).get("answer_metrics", {}).get("token_f1", 0.0)) for q in answered_records]
    answered_cit_cov = [float(q.get("metrics", {}).get("citation_metrics", {}).get("citation_gold_coverage", 0.0)) for q in answered_records]

    answered_emitted_cit = sum(q.get("metrics", {}).get("citation_metrics", {}).get("total_citations", 0) for q in answered_records)
    answered_valid_cit = sum(q.get("metrics", {}).get("citation_metrics", {}).get("valid_citations", 0) for q in answered_records)
    answered_cit_precision = (answered_valid_cit / answered_emitted_cit) if answered_emitted_cit > 0 else 1.0

    conditioned_on_answered = {
        "answered_questions_count": answered_count,
        "answered_exact_match": mean_val(answered_em),
        "answered_token_f1": mean_val(answered_f1),
        "answered_gold_document_coverage": mean_val(answered_cit_cov),
        "answered_citation_precision": answered_cit_precision,
    }

    query_latencies = [q.get("query_result", {}).get("measured_total_query_seconds", 0.0) for q in completed_benchmarks]
    indexing_latencies = [q.get("indexing_stats", {}).get("indexing_seconds", 0.0) for q in completed_benchmarks]

    # Partition by type
    bridge_records = [q for q in benchmark_records if q.get("question_type") == "bridge"]
    comp_records = [q for q in benchmark_records if q.get("question_type") == "comparison"]

    by_type = {
        "bridge": {
            "count": len(bridge_records),
            "exact_match": mean_val([float(q.get("metrics", {}).get("answer_metrics", {}).get("exact_match", 0)) for q in bridge_records]),
            "token_f1": mean_val([float(q.get("metrics", {}).get("answer_metrics", {}).get("token_f1", 0.0)) for q in bridge_records]),
            "recall_at_2": mean_val([float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_2", 0.0)) for q in bridge_records]),
            "recall_at_5": mean_val([float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_5", 0.0)) for q in bridge_records]),
        },
        "comparison": {
            "count": len(comp_records),
            "exact_match": mean_val([float(q.get("metrics", {}).get("answer_metrics", {}).get("exact_match", 0)) for q in comp_records]),
            "token_f1": mean_val([float(q.get("metrics", {}).get("answer_metrics", {}).get("token_f1", 0.0)) for q in comp_records]),
            "recall_at_2": mean_val([float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_2", 0.0)) for q in comp_records]),
            "recall_at_5": mean_val([float(q.get("metrics", {}).get("retrieval_metrics", {}).get("recall_at_5", 0.0)) for q in comp_records]),
        },
    }

    aggregates = {
        "benchmark_sample_size": len(benchmark_records),
        "completed_count": len(completed_benchmarks),
        "failed_count": len(failed_benchmarks),
        "overall_exact_match": mean_val(all_em),
        "overall_token_f1": mean_val(all_f1),
        "abstention_rate": mean_val(all_abstained),
        "retrieval": {
            "mean_recall_at_2": mean_val(all_r2),
            "mean_recall_at_5": mean_val(all_r5),
            "mean_complete_support_at_2": mean_val(all_comp2),
            "mean_complete_support_at_5": mean_val(all_comp5),
        },
        "grounding": {
            "citation_precision_among_emitted": citation_precision_among_emitted,
            "citation_question_coverage": citation_question_coverage,
            "mean_gold_supporting_document_coverage": mean_gold_doc_coverage,
            "total_emitted_citations": total_emitted_citations,
            "total_valid_citations": total_valid_citations,
            "mean_citation_validity_rate": citation_precision_among_emitted,
            "mean_citation_gold_coverage": mean_gold_doc_coverage,
        },
        "conditioned_on_answered": conditioned_on_answered,
        "performance": {
            "mean_query_latency_seconds": mean_val(query_latencies),
            "median_query_latency_seconds": median_val(query_latencies),
            "total_query_duration_seconds": sum(query_latencies),
            "mean_indexing_latency_seconds": mean_val(indexing_latencies),
            "median_indexing_latency_seconds": median_val(indexing_latencies),
            "total_indexing_duration_seconds": sum(indexing_latencies),
        },
        "by_question_type": by_type,
    }

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "HotpotQA 10-Question Pilot Autonomous Run",
        "benchmark_summary": aggregates,
        "regression_control_q2": control_record,
        "all_questions": q_records,
    }
    return final_payload


def main() -> None:
    logger.info("################################################################################")
    logger.info("STARTING AUTONOMOUS HOTPOTQA 10-QUESTION PILOT EVALUATION")
    logger.info("Manifest: %s", MANIFEST_PATH)
    logger.info("Pilot KB Root: %s", PILOT_KB_ROOT)
    logger.info("Reused Q2 KB: %s", REUSED_KB_PATH)
    logger.info("################################################################################")

    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found at %s", MANIFEST_PATH)
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    questions = manifest.get("questions", [])
    if len(questions) != 10:
        logger.error("Expected 10 questions in manifest, found %d", len(questions))
        sys.exit(1)

    state = load_run_state()

    total_start_time = time.perf_counter()

    for idx, q in enumerate(questions, start=1):
        qid = q["question_id"]
        q_entry = state["questions"].get(qid, {})
        if q_entry.get("status") == "completed":
            logger.info("[%d/10] Already completed: %s", idx, qid)
            continue

        # Retry loop for transient failures (max 2 retries = 3 attempts)
        max_attempts = 3
        attempt = q_entry.get("attempts", 0)

        while attempt < max_attempts:
            attempt += 1
            q_entry["attempts"] = attempt
            logger.info("[%d/10] Processing %s (Attempt %d/%d)...", idx, qid, attempt, max_attempts)

            try:
                execute_question(idx, q, state)
                break
            except Exception as exc:
                err_msg = f"Attempt {attempt}/{max_attempts} failed: {exc}"
                logger.error("[%d/10] %s", idx, err_msg)
                logger.error(traceback.format_exc())
                q_entry["error"] = err_msg

                if attempt >= max_attempts:
                    logger.error("[%d/10] Exhausted all %d attempts for %s. Marking FAILED.", idx, max_attempts, qid)
                    q_entry["status"] = "failed"
                    update_run_state(state)
                else:
                    logger.info("[%d/10] Retrying in 5 seconds...", idx)
                    time.sleep(5)

    total_elapsed = time.perf_counter() - total_start_time
    logger.info("================================================================================")
    logger.info("ALL QUESTIONS COMPLETED OR ATTEMPTED in %.2fs (%.2f hours)", total_elapsed, total_elapsed / 3600.0)

    # Compute final aggregate results and persist
    results = compute_aggregate_results(state)
    results["total_pilot_wallclock_seconds"] = total_elapsed
    atomic_save_json(RESULTS_PATH, results)

    logger.info("Final results saved to: %s", RESULTS_PATH)
    logger.info("======================= PILOT RUN COMPLETED =======================")


if __name__ == "__main__":
    main()
