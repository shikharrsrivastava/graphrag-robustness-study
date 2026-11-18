#!/usr/bin/env python3
"""Runner for Milestone 3 Generator Evaluation across experimental conditions.

Conditions:
- A_v11: Gold paragraphs only (2 paragraphs), Prompt v1.1
- A_v10: Gold paragraphs only (2 paragraphs), Prompt v1.0
- B: All 10 paragraphs (2 gold + 8 distractors), Prompt v1.1
- C: Dense retrieval top-3 paragraphs using embeddinggemma, Prompt v1.1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.models import Evidence, ProvenanceRoute, QueryIntent, QueryPlan, RetrievalResult
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaConfig, OllamaLLMClient
from src.generation.answer_generator import OllamaAnswerGenerator
from src.retrieval.context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def safe_filename(title: str, max_len: int = 50) -> str:
    """Produce safe ASCII filename for a title."""
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)
    return clean[:max_len]


def make_chunk_text(title: str, sentences: List[str]) -> str:
    """Format paragraph text identically to ContextBuilder expectations."""
    joined_sentences = " ".join(s.strip() for s in sentences if s.strip())
    return f"Title: {title}\n{joined_sentences}"


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def build_evidence_chunks(
    question_id: str,
    paragraphs: List[Tuple[str, List[str]]],
    scores: Optional[List[float]] = None,
) -> List[Evidence]:
    """Construct Evidence objects for a list of (title, sentences) paragraphs."""
    chunks: List[Evidence] = []
    for idx, (title, sents) in enumerate(paragraphs):
        chunk_text = make_chunk_text(title, sents)
        fname = f"{safe_filename(title)}.txt"
        dist = (1.0 - scores[idx]) if scores is not None and idx < len(scores) else None
        rank = (idx + 1) if scores is not None else None

        ev = Evidence(
            kb_id="hotpotqa_m3",
            chunk_id=f"{question_id}_P{idx}",
            doc_id=f"D{idx:03d}",
            filename=fname,
            text=chunk_text,
            routes=[ProvenanceRoute.VECTOR],
            vector_rank=rank,
            vector_distance=dist,
            page_numbers=[1],
        )
        chunks.append(ev)
    return chunks


def log_progress(progress_file: Path, step_name: str, output_path: str) -> None:
    """Append one line to PROGRESS.md with ISO timestamp."""
    now_iso = datetime.now().astimezone().isoformat()
    line = f"| {now_iso} | {step_name} | `{output_path}` |\n"
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(line)


def run_condition_for_question(
    condition: str,
    q: Dict[str, Any],
    generator: OllamaAnswerGenerator,
    embedding_service: Optional[OllamaEmbeddingService] = None,
) -> Dict[str, Any]:
    """Execute evaluation on a single question under the specified condition."""
    qid = q["_id"]
    q_text = q["question"]
    gold_titles: Set[str] = {t for t, _ in q["supporting_facts"]}
    all_context: List[Tuple[str, List[str]]] = [(t, s) for t, s in q["context"]]

    t0 = time.time()
    gold_recall_bucket = None
    gold_recall_count = None
    retrieved_chunk_records = []

    if condition in ("A_v11", "A_v10"):
        # Gold paragraphs only
        selected_paragraphs = [(t, s) for t, s in all_context if t in gold_titles]
        evidence_chunks = build_evidence_chunks(qid, selected_paragraphs)

    elif condition == "B":
        # All 10 candidate paragraphs in dataset order
        evidence_chunks = build_evidence_chunks(qid, all_context)

    elif condition == "C":
        # Dense retrieval top 3 using embeddinggemma
        if embedding_service is None:
            raise ValueError("Embedding service is required for condition C")

        q_vec = embedding_service.embed_text(q_text)
        cand_texts = [make_chunk_text(t, s) for t, s in all_context]
        cand_vecs = embedding_service.embed_batch(cand_texts)

        scored_candidates = []
        for i, (title, sents) in enumerate(all_context):
            sim = cosine_similarity(q_vec, cand_vecs[i])
            is_gold = title in gold_titles
            scored_candidates.append({
                "index": i,
                "title": title,
                "sentences": sents,
                "similarity": sim,
                "is_gold": is_gold,
            })

        # Sort descending by similarity
        scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top3 = scored_candidates[:3]

        top3_titles = {c["title"] for c in top3}
        retrieved_gold_count = sum(1 for gt in gold_titles if gt in top3_titles)
        gold_recall_count = retrieved_gold_count
        if retrieved_gold_count == len(gold_titles):
            gold_recall_bucket = "both"
        elif retrieved_gold_count > 0:
            gold_recall_bucket = "one"
        else:
            gold_recall_bucket = "none"

        retrieved_chunk_records = [
            {
                "rank": rank + 1,
                "title": c["title"],
                "similarity": round(c["similarity"], 4),
                "is_gold": c["is_gold"],
            }
            for rank, c in enumerate(top3)
        ]

        top3_paragraphs = [(c["title"], c["sentences"]) for c in top3]
        top3_scores = [c["similarity"] for c in top3]
        evidence_chunks = build_evidence_chunks(qid, top3_paragraphs, scores=top3_scores)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    retrieval_result = RetrievalResult(
        kb_id="hotpotqa_m3",
        question=q_text,
        query_plan=QueryPlan(
            question=q_text,
            intent=QueryIntent.LOCAL,
            seed_entities=[],
            relationship_path=[],
            target_type=None,
            requires_clarification=False,
            clarification_reason=None,
            invalid_relationship_types=[],
        ),
        evidence_list=evidence_chunks,
        community_reports=[],
        abstained=False,
    )

    output = generator.generate_structured(retrieval_result)
    t_elapsed = time.time() - t0

    return {
        "question_id": qid,
        "type": q.get("type", "unknown"),
        "level": q.get("level", "unknown"),
        "question": q_text,
        "gold_answer": q.get("answer", ""),
        "supporting_facts": q.get("supporting_facts", []),
        "reasoning": output.reasoning,
        "answer": output.answer,
        "canonical_answer": output.canonical_answer,
        "abstained": output.abstained,
        "abstention_reason": output.abstention_reason,
        "cited_chunk_ids": output.cited_chunk_ids,
        "citation_references": output.citation_references,
        "generation_error": output.generation_error,
        "time_seconds": round(t_elapsed, 2),
        "gold_recall_bucket": gold_recall_bucket,
        "gold_recall_count": gold_recall_count,
        "retrieved_chunks": retrieved_chunk_records,
    }


def run_condition(
    condition: str,
    questions: List[Dict[str, Any]],
    output_dir: Path,
    progress_file: Path,
    embedding_service: Optional[OllamaEmbeddingService] = None,
) -> Path:
    """Run full benchmark for one condition across all questions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}_{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    prompt_version = "v1.0" if condition == "A_v10" else "v1.1"
    llm_client = OllamaLLMClient(config=OllamaConfig(llm_model="gemma3:4b", temperature=0.0))
    context_builder = ContextBuilder(max_chars=12000, graph_context_mode="off")
    generator = OllamaAnswerGenerator(
        llm_client=llm_client,
        context_builder=context_builder,
        prompt_version=prompt_version,
        comparative_reasoning_mode="off",
    )

    logger.info("Starting run for Condition '%s' (Prompt %s, total %d questions)", condition, prompt_version, len(questions))

    with open(results_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions, start=1):
            logger.info("[%d/%d] Processing %s (%s): %s", i, len(questions), q["_id"][:8], q.get("type"), q["question"][:60])
            record = run_condition_for_question(condition, q, generator, embedding_service)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            logger.info(
                "    Result: ans='%s' canon='%s' abs=%s err=%s (%.2fs)",
                (record["answer"][:40] + "...") if len(record["answer"]) > 40 else record["answer"],
                record["canonical_answer"],
                record["abstained"],
                record["generation_error"],
                record["time_seconds"],
            )

    logger.info("Completed Condition '%s'. Saved to %s", condition, results_path)
    log_progress(progress_file, f"Step 2: Completed Condition {condition} run", str(results_path))
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3 Condition Runner")
    parser.add_argument(
        "--condition",
        choices=["A_v11", "A_v10", "B", "C", "all"],
        default="all",
        help="Condition to execute",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("experiments/hotpotqa_m3/selected_questions.json"),
        help="Path to selected_questions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/hotpotqa_m3/runs"),
        help="Directory to save run results",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("experiments/hotpotqa_m3/PROGRESS.md"),
        help="Path to PROGRESS.md log",
    )
    args = parser.parse_args()

    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Loaded %d selected questions from %s", len(questions), args.questions_file)

    conditions_to_run = ["A_v11", "A_v10", "B", "C"] if args.condition == "all" else [args.condition]

    embedding_service = None
    if "C" in conditions_to_run:
        logger.info("Initializing OllamaEmbeddingService for Condition C...")
        embedding_service = OllamaEmbeddingService(config=OllamaConfig())

    for cond in conditions_to_run:
        run_condition(
            condition=cond,
            questions=questions,
            output_dir=args.output_dir,
            progress_file=args.progress_file,
            embedding_service=embedding_service,
        )


if __name__ == "__main__":
    main()
