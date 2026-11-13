"""Answer generation replay harness for Hybrid GraphRAG.

Loads pre-retrieved evidence from an existing retrieval-only validation file,
reconstructs RetrievalResult objects without re-running query planning or retrieval,
and invokes OllamaAnswerGenerator to generate structured answers, citations, and evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.configuration import load_engine_config
from src.core.models import (
    CommunityReport,
    Evidence,
    QueryPlan,
    RetrievalResult,
)
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import (
    AnswerGenerationOutput,
    OllamaAnswerGenerator,
)
from src.retrieval.context_builder import ContextBuilder
from run_hotpot import (
    extract_canonical_candidate,
    normalize_answer,
    token_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("answer_replay")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay answer generation from existing retrieval-only validation results"
    )
    p.add_argument(
        "--input",
        default="experiments/hotpotqa/validation_retrieval_only.json",
        help="Path to existing retrieval-only validation JSON (must exist)",
    )
    p.add_argument(
        "--output",
        default="experiments/hotpotqa/validation_answer_replay.json",
        help="Path to destination answer replay results (must NOT exist)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional override for engine config snapshot (defaults to kb_config_path in input JSON)",
    )
    return p


def reconstruct_retrieval_result(
    meta: Dict[str, Any],
    route_name: str,
    route_data: Dict[str, Any],
) -> RetrievalResult:
    """Reconstruct a Pydantic RetrievalResult from serialized validation JSON data."""
    shared_plan = QueryPlan.model_validate(meta["shared_query_plan"])
    evidence_list = [Evidence.model_validate(e) for e in route_data.get("evidence", [])]
    comm_reports = [CommunityReport.model_validate(r) for r in route_data.get("community_reports", [])]
    resolved_ids = route_data.get("resolved_seed_entity_ids") or meta.get("resolved_seed_entity_ids", [])

    return RetrievalResult(
        kb_id=meta["kb_id"],
        question=meta["question"],
        query_plan=shared_plan,
        resolved_seed_entity_ids=resolved_ids,
        evidence_list=evidence_list,
        community_reports=comm_reports,
        abstained=route_data.get("abstained", False),
        abstention_reason=route_data.get("abstention_reason"),
        route_diagnostics=dict(route_data.get("diagnostics", {})),
    )


def main() -> None:
    args = build_parser().parse_args()

    # --- Preflight Protection 1: Target output file must NOT exist ---
    out_path = Path(args.output).resolve()
    if out_path.exists():
        logger.error(
            "PREFLIGHT REFUSAL: Target output file '%s' already exists. Aborting immediately.",
            out_path,
        )
        raise FileExistsError(f"Target output file '{out_path}' already exists.")

    # --- Preflight Protection 2: Input retrieval file must exist ---
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        logger.error("Input retrieval file '%s' not found.", in_path)
        raise FileNotFoundError(f"Input retrieval file '{in_path}' not found.")

    logger.info("Loading existing retrieval results from %s...", in_path)
    raw_data: Dict[str, Any] = json.loads(in_path.read_text(encoding="utf-8"))

    if "metadata" not in raw_data or "routes" not in raw_data:
        raise ValueError(f"Input file '{in_path}' is missing 'metadata' or 'routes' structures.")

    meta = raw_data["metadata"]
    question = meta["question"]
    gold_answer = meta.get("gold_answer", "")

    # --- Preflight Protection 3: Locate and load engine configuration strictly for LLM settings ---
    config_path_str = args.config or meta.get("kb_config_path")
    if not config_path_str or not Path(config_path_str).exists():
        config_path_str = "knowledge_bases/hotpotqa_eval/hotpot_full_5a8b57f25542995d1e6f1371/config_snapshot.yaml"

    config_path = Path(config_path_str).resolve()
    if not config_path.exists():
        logger.error("Configuration file '%s' not found.", config_path)
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    logger.info("Loading generation settings from config: %s...", config_path)
    cfg = load_engine_config(config_path)

    # Initialize strictly generation dependencies (Zero retrieval/storage/index calls)
    llm_client = OllamaLLMClient(config=cfg.ollama)
    context_builder = ContextBuilder(max_chars=cfg.retrieval.max_context_chars)
    answer_generator = OllamaAnswerGenerator(llm_client=llm_client, context_builder=context_builder)

    # Reconstruct RetrievalResult objects and verify exact evidence ordering
    reconstructed_results: Dict[str, RetrievalResult] = {}
    for mode_name, route_data in raw_data["routes"].items():
        reconstructed = reconstruct_retrieval_result(meta, mode_name, route_data)

        orig_cids = [e["chunk_id"] for e in route_data.get("evidence", [])]
        recon_cids = [e.chunk_id for e in reconstructed.evidence_list]
        if orig_cids != recon_cids:
            raise RuntimeError(
                f"Evidence ordering mismatch for route '{mode_name}': expected {orig_cids}, got {recon_cids}"
            )

        orig_reps = len(route_data.get("community_reports", []))
        recon_reps = len(reconstructed.community_reports)
        if orig_reps != recon_reps:
            raise RuntimeError(
                f"Community report count mismatch for route '{mode_name}': expected {orig_reps}, got {recon_reps}"
            )

        reconstructed_results[mode_name] = reconstructed
        logger.info(
            "Reconstructed %s: %d evidence chunks, %d community reports (provenance preserved).",
            mode_name,
            len(reconstructed.evidence_list),
            len(reconstructed.community_reports),
        )

    # Prepare output payload preserving all prior retrieval metadata and timing
    output_payload: Dict[str, Any] = json.loads(json.dumps(raw_data))
    output_payload["metadata"]["answer_replay"] = {
        "replay_executed": True,
        "source_retrieval_file": str(in_path),
        "retrieval_reused_without_reexecution": True,
        "llm_model": cfg.ollama.llm_model,
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_payload["metadata"]["generated_answers_included"] = True

    route_generation_seconds: Dict[str, float] = {}

    # Execute answer generation separately for each reconstructed route
    for mode_name, result in reconstructed_results.items():
        logger.info("Generating answer for route '%s' from pre-retrieved evidence...", mode_name)
        t_start = time.perf_counter()
        structured_ans: AnswerGenerationOutput = answer_generator.generate_structured(result)
        t_gen = time.perf_counter() - t_start
        route_generation_seconds[mode_name] = t_gen

        answer = structured_ans.answer
        canonical_pred = extract_canonical_candidate(
            answer,
            getattr(result, "canonical_answer", None) or structured_ans.canonical_answer,
            question,
        )
        em = normalize_answer(canonical_pred) == normalize_answer(gold_answer)
        f1 = token_f1(canonical_pred, gold_answer)

        logger.info(
            "Route '%s' generated in %.2fs: canonical='%s', EM=%s, F1=%.2f, cited_chunks=%s",
            mode_name,
            t_gen,
            canonical_pred,
            em,
            f1,
            result.cited_chunk_ids,
        )

        route_record = output_payload["routes"][mode_name]
        route_record["timing"]["answer_generation_seconds"] = t_gen
        route_record["answer"] = answer
        route_record["canonical_answer"] = getattr(result, "canonical_answer", None) or canonical_pred
        route_record["cited_chunk_ids"] = result.cited_chunk_ids
        route_record["citation_references"] = result.citation_references
        route_record["answer_metrics"] = {
            "exact_match": em,
            "token_f1": f1,
        }
        route_record["abstained"] = result.abstained
        route_record["abstention_reason"] = result.abstention_reason
        route_record["diagnostics"]["answer_generation_seconds"] = t_gen

    output_payload["metadata"]["answer_replay"]["generation_seconds_by_route"] = route_generation_seconds
    output_payload["metadata"]["answer_replay"]["total_generation_seconds"] = sum(
        route_generation_seconds.values()
    )

    # --- Safe Exclusive Creation ---
    logger.info("Writing answer replay results exclusively to %s...", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Answer replay completed successfully. Results saved to %s.", out_path)


if __name__ == "__main__":
    main()
