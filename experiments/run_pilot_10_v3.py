#!/usr/bin/env python3
"""Query-only evaluation runner for HotpotQA 10-Question Pilot (V3 Final Release Candidate).

Reuses existing persisted KBs for all 10 questions without reindexing or regenerating summaries.
Evaluates answers with the stabilised multi-hop bridge, suffix preservation, and comparative reasoning.
Saves results to experiments/hotpotqa/pilot_10_results_v3.json preserving original baselines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hybrid_graphrag import HybridGraphRAG, QueryResult
from experiments.run_pilot_10 import (
    compute_aggregate_results,
    compute_retrieval_and_answer_metrics,
    prepare_raw_documents,
    verify_kb_persistence,
    atomic_save_json,
)

MANIFEST_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_manifest.json"
RESULTS_V3_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_results_v3.json"
PILOT_KB_ROOT = REPO_ROOT / "knowledge_bases" / "hotpotqa_pilot"
REUSED_KB_PATH = REPO_ROOT / "knowledge_bases" / "hotpotqa_eval" / "hotpot_full_5a8b57f25542995d1e6f1371"
INPUTS_BASE_DIR = REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "pilot"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pilot_10_v3")


def main() -> None:
    logger.info("################################################################################")
    logger.info("STARTING QUERY-ONLY HOTPOTQA 10-QUESTION PILOT EVALUATION (V3 RELEASE CANDIDATE)")
    logger.info("Manifest: %s", MANIFEST_PATH)
    logger.info("Output: %s", RESULTS_V3_PATH)
    logger.info("################################################################################")

    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found: %s", MANIFEST_PATH)
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    questions = manifest.get("questions", [])
    logger.info("Loaded %d questions from manifest.", len(questions))

    state: Dict[str, Any] = {"questions": {}}
    total_start = time.perf_counter()

    for q_idx, q in enumerate(questions, start=1):
        qid = q["question_id"]
        qtype = q["question_type"]
        question_text = q["question"]
        gold_answer = q["gold_answer"]
        is_q2_control = (qid == "5a8b57f25542995d1e6f1371")

        logger.info("--------------------------------------------------------------------------------")
        logger.info("[%d/10] Question %s (%s, Control=%s)", q_idx, qid, qtype, is_q2_control)
        logger.info("Q: %s", question_text)
        logger.info("Gold: %s", gold_answer)

        # Identify KB path
        kb_path = REUSED_KB_PATH if is_q2_control else (PILOT_KB_ROOT / qid)
        valid, reason = verify_kb_persistence(kb_path)
        if not valid:
            logger.error("[%d/10] KB validation failed for %s: %s", q_idx, kb_path, reason)
            state["questions"][qid] = {
                "question_id": qid,
                "question_index": q_idx,
                "question_type": qtype,
                "question": question_text,
                "gold_answer": gold_answer,
                "is_regression_control": is_q2_control,
                "status": "failed",
                "error": f"KB verification failed: {reason}",
            }
            continue

        raw_dir = (REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "full" / qid) if is_q2_control else (INPUTS_BASE_DIR / qid)
        title_by_filename = prepare_raw_documents(q, raw_dir)

        # Query existing KB
        query_start = time.perf_counter()
        try:
            with HybridGraphRAG.from_existing(kb_path) as engine:
                llm_before = getattr(engine._engine.llm_client, "stats", {}).get("requests", 0)
                query_result: QueryResult = engine.query(question=question_text, mode="hybrid")
                query_duration = time.perf_counter() - query_start
                llm_after = getattr(engine._engine.llm_client, "stats", {}).get("requests", 0)
                query_llm_requests = max(0, llm_after - llm_before)

            metrics = compute_retrieval_and_answer_metrics(query_result, q, title_by_filename)
            em = metrics["answer_metrics"]["exact_match"]
            f1 = metrics["answer_metrics"]["token_f1"]
            r2 = metrics["retrieval_metrics"]["recall_at_2"]
            r5 = metrics["retrieval_metrics"]["recall_at_5"]
            valid_cits = metrics["citation_metrics"]["valid_citations"]
            total_cits = metrics["citation_metrics"]["total_citations"]

            logger.info("[%d/10] Answer: %s", q_idx, query_result.answer)
            logger.info("[%d/10] Canonical: '%s' | Gold: '%s'", q_idx, query_result.canonical_answer, gold_answer)
            logger.info(
                "[%d/10] Metrics: EM=%d, F1=%.4f, Recall@2=%.2f, Recall@5=%.2f, Citations=%d/%d, Duration=%.2fs",
                q_idx, em, f1, r2, r5, valid_cits, total_cits, query_duration,
            )

            state["questions"][qid] = {
                "question_id": qid,
                "question_index": q_idx,
                "question_type": qtype,
                "question": question_text,
                "gold_answer": gold_answer,
                "is_regression_control": is_q2_control,
                "status": "completed",
                "attempts": 1,
                "error": None,
                "indexing_stats": {"reused_existing_kb": True, "indexing_seconds": 0.0},
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
                    "measured_total_query_seconds": query_duration,
                    "timing_breakdown": query_result.timing,
                    "query_llm_requests": query_llm_requests,
                },
                "metrics": metrics,
            }
        except Exception as exc:
            logger.error("[%d/10] Query failed for %s: %s", q_idx, qid, exc, exc_info=True)
            state["questions"][qid] = {
                "question_id": qid,
                "question_index": q_idx,
                "question_type": qtype,
                "question": question_text,
                "gold_answer": gold_answer,
                "is_regression_control": is_q2_control,
                "status": "failed",
                "error": str(exc),
            }

    total_elapsed = time.perf_counter() - total_start
    logger.info("================================================================================")
    logger.info("QUERY-ONLY EVALUATION FINISHED in %.2fs (%.2f min)", total_elapsed, total_elapsed / 60.0)

    # Compute aggregates and save to v3 results path
    aggregates = compute_aggregate_results(state)
    aggregates["total_pilot_wallclock_seconds"] = total_elapsed
    atomic_save_json(RESULTS_V3_PATH, aggregates)
    logger.info("Saved V3 results to %s", RESULTS_V3_PATH)


if __name__ == "__main__":
    main()
