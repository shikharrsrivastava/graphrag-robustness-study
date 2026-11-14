#!/usr/bin/env python3
"""Controlled Model Capability Evaluation: Gemma 3 4B vs Qwen 2.5 7B.

Runs identical questions, identical evidence contexts, identical validation rules,
and temperature 0 over the 10 HotpotQA evaluation questions using existing persisted KBs.

Saves:
- experiments/hotpotqa/model_comparison_report.json
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.configuration import OllamaConfig
from src.core.models import RetrievalResult
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import OllamaAnswerGenerator
from hybrid_graphrag import HybridGraphRAG, QueryResult
from experiments.run_pilot_10 import (
    compute_retrieval_and_answer_metrics,
    prepare_raw_documents,
    verify_kb_persistence,
    atomic_save_json,
)

MANIFEST_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "pilot_10_manifest.json"
REPORT_OUTPUT_PATH = REPO_ROOT / "experiments" / "hotpotqa" / "model_comparison_report.json"
PILOT_KB_ROOT = REPO_ROOT / "knowledge_bases" / "hotpotqa_pilot"
REUSED_KB_PATH = REPO_ROOT / "knowledge_bases" / "hotpotqa_eval" / "hotpot_full_5a8b57f25542995d1e6f1371"
INPUTS_BASE_DIR = REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "pilot"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_comparison")


def evaluate_with_generator(
    answer_generator: OllamaAnswerGenerator,
    retrieval_result: RetrievalResult,
    question_dict: Dict[str, Any],
    title_by_filename: Dict[str, str],
    model_name: str,
) -> Dict[str, Any]:
    """Run generation on frozen retrieval context and evaluate metrics."""
    # Deep copy retrieval result so state doesn't leak between models
    ret_copy = copy.deepcopy(retrieval_result)

    start_time = time.perf_counter()
    structured = answer_generator.generate_structured(ret_copy)
    duration = time.perf_counter() - start_time

    # Wrap in QueryResult for metric computation
    from hybrid_graphrag.models import Citation
    citations = [
        Citation(
            chunk_id=ref["chunk_id"],
            doc_id=ref["doc_id"],
            filename=ref["filename"],
            page_numbers=ref.get("page_numbers", [1]),
            text_snippet="",
            valid=True,
        )
        for ref in structured.citation_references
    ]

    q_res = QueryResult(
        question=ret_copy.question,
        answer=structured.answer,
        canonical_answer=structured.canonical_answer or None,
        abstained=structured.abstained,
        abstention_reason=structured.abstention_reason,
        citations=citations,
        cited_chunk_ids=structured.cited_chunk_ids,
        mode="hybrid",
        latency_seconds=duration,
        timing={"generation_seconds": duration},
        raw_result=ret_copy,
    )

    metrics = compute_retrieval_and_answer_metrics(q_res, question_dict, title_by_filename)

    return {
        "model": model_name,
        "answer": structured.answer,
        "canonical_answer": structured.canonical_answer,
        "reasoning": structured.reasoning,
        "extracted_attributes": [fact.model_dump() for fact in structured.extracted_attributes],
        "abstained": structured.abstained,
        "abstention_reason": structured.abstention_reason,
        "cited_chunk_ids": structured.cited_chunk_ids,
        "citation_count": len(citations),
        "valid_citation_count": metrics["citation_metrics"]["valid_citations"],
        "exact_match": metrics["answer_metrics"]["exact_match"],
        "token_f1": metrics["answer_metrics"]["token_f1"],
        "latency_seconds": duration,
    }


def main() -> None:
    logger.info("################################################################################")
    logger.info("STARTING CONTROLLED MODEL CAPABILITY EVALUATION (GEMMA 3 4B vs QWEN 2.5 7B)")
    logger.info("################################################################################")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    questions = manifest.get("questions", [])

    # Setup clients
    gemma_client = OllamaLLMClient(OllamaConfig(llm_model="gemma3:4b", temperature=0.0))
    qwen_client = OllamaLLMClient(OllamaConfig(llm_model="qwen2.5:7b", temperature=0.0))

    gemma_gen = OllamaAnswerGenerator(llm_client=gemma_client)
    qwen_gen = OllamaAnswerGenerator(llm_client=qwen_client)

    results: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions, start=1):
        qid = q["question_id"]
        qtype = q["question_type"]
        question_text = q["question"]
        gold_answer = q["gold_answer"]
        is_q2_control = (qid == "5a8b57f25542995d1e6f1371")

        logger.info("================================================================================")
        logger.info("[%d/10] Evaluating Question %s: %s", idx, qid, question_text)
        logger.info("Gold Answer: %s", gold_answer)

        kb_path = REUSED_KB_PATH if is_q2_control else (PILOT_KB_ROOT / qid)
        raw_dir = (REPO_ROOT / "experiments" / "hotpotqa" / "inputs" / "full" / qid) if is_q2_control else (INPUTS_BASE_DIR / qid)
        title_by_filename = prepare_raw_documents(q, raw_dir)

        # Retrieve frozen evidence using the existing KB
        with HybridGraphRAG.from_existing(kb_path) as engine:
            _, retrieval_result = engine._engine.query(question=question_text, mode="hybrid")

        logger.info("[%d/10] Retrieved %d chunks.", idx, len(retrieval_result.evidence_list))

        # Model A: Gemma 3 4B
        gemma_eval = evaluate_with_generator(gemma_gen, retrieval_result, q, title_by_filename, "gemma3:4b")
        logger.info(
            "[%d/10] Gemma 3 4B -> Canonical: '%s' | EM: %d | F1: %.4f | Latency: %.2fs",
            idx, gemma_eval["canonical_answer"], gemma_eval["exact_match"], gemma_eval["token_f1"], gemma_eval["latency_seconds"],
        )

        # Model B: Qwen 2.5 7B
        qwen_eval = evaluate_with_generator(qwen_gen, retrieval_result, q, title_by_filename, "qwen2.5:7b")
        logger.info(
            "[%d/10] Qwen 2.5 7B -> Canonical: '%s' | EM: %d | F1: %.4f | Latency: %.2fs",
            idx, qwen_eval["canonical_answer"], qwen_eval["exact_match"], qwen_eval["token_f1"], qwen_eval["latency_seconds"],
        )

        q_record = {
            "question_id": qid,
            "question_index": idx,
            "question_type": qtype,
            "question": question_text,
            "gold_answer": gold_answer,
            "is_regression_control": is_q2_control,
            "retrieved_chunk_count": len(retrieval_result.evidence_list),
            "gemma3_4b": gemma_eval,
            "qwen2_5_7b": qwen_eval,
        }
        results.append(q_record)

    # Compute aggregates across the 9 benchmark questions (excluding Q2 control)
    bench_records = [r for r in results if not r["is_regression_control"]]

    gemma_em = sum(r["gemma3_4b"]["exact_match"] for r in bench_records) / len(bench_records)
    gemma_f1 = sum(r["gemma3_4b"]["token_f1"] for r in bench_records) / len(bench_records)
    gemma_lat = sum(r["gemma3_4b"]["latency_seconds"] for r in bench_records) / len(bench_records)
    gemma_abst = sum(1 for r in bench_records if r["gemma3_4b"]["abstained"]) / len(bench_records)

    qwen_em = sum(r["qwen2_5_7b"]["exact_match"] for r in bench_records) / len(bench_records)
    qwen_f1 = sum(r["qwen2_5_7b"]["token_f1"] for r in bench_records) / len(bench_records)
    qwen_lat = sum(r["qwen2_5_7b"]["latency_seconds"] for r in bench_records) / len(bench_records)
    qwen_abst = sum(1 for r in bench_records if r["qwen2_5_7b"]["abstained"]) / len(bench_records)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size": len(bench_records),
        "comparison_aggregates": {
            "gemma3_4b": {
                "exact_match": gemma_em,
                "token_f1": gemma_f1,
                "abstention_rate": gemma_abst,
                "mean_latency_seconds": gemma_lat,
            },
            "qwen2_5_7b": {
                "exact_match": qwen_em,
                "token_f1": qwen_f1,
                "abstention_rate": qwen_abst,
                "mean_latency_seconds": qwen_lat,
            },
        },
        "per_question_results": results,
    }

    atomic_save_json(REPORT_OUTPUT_PATH, summary)
    logger.info("Saved model comparison report to: %s", REPORT_OUTPUT_PATH)
    logger.info("======================= COMPARISON COMPLETE =======================")


if __name__ == "__main__":
    main()
