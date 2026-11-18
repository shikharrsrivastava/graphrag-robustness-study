#!/usr/bin/env python3
"""Calibration script: Evaluates dense retrieval recall on synthetic corpus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaConfig


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson 95% score confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    centre = p_hat + (z**2) / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))
    lower = max(0.0, (centre - spread) / denom)
    upper = min(1.0, (centre + spread) / denom)
    return lower, upper


def evaluate_dense_recall(corpus_dir: Path, batch_size: int = 25) -> Dict[str, Any]:
    """Run dense embedding retrieval on corpus_dir and compute recall@k."""
    docs_dir = corpus_dir / "docs"
    questions_path = corpus_dir / "questions.jsonl"

    # Load questions (filter out unanswerable)
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if q["group"] != "unanswerable":
                    questions.append(q)

    # Load documents in sorted order
    doc_files = sorted(docs_dir.glob("*.txt"), key=lambda p: p.name)
    doc_ids = [p.stem for p in doc_files]
    doc_texts = [p.read_text(encoding="utf-8").strip() for p in doc_files]

    print(f"Embedding {len(doc_texts)} documents and {len(questions)} questions using embeddinggemma...")
    service = OllamaEmbeddingService(config=OllamaConfig())

    t0 = time.time()
    # Embed docs
    doc_vecs = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        doc_vecs.extend(service.embed_batch(batch))

    # Embed questions
    q_texts = [q["question"] for q in questions]
    q_vecs = []
    for i in range(0, len(q_texts), batch_size):
        batch = q_texts[i : i + batch_size]
        q_vecs.extend(service.embed_batch(batch))
    t_embed = time.time() - t0
    print(f"Embedding finished in {t_embed:.2f}s")

    doc_mat = np.array(doc_vecs, dtype=np.float32)
    q_mat = np.array(q_vecs, dtype=np.float32)

    # Normalize
    doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True)
    doc_norms[doc_norms == 0.0] = 1.0
    doc_mat = doc_mat / doc_norms

    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_norms[q_norms == 0.0] = 1.0
    q_mat = q_mat / q_norms

    # Cosine similarities: shape (len(questions), len(doc_texts))
    sim_mat = np.dot(q_mat, doc_mat.T)

    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    ks = [1, 2, 3, 4, 6]
    per_question_results = []

    for q_idx, q in enumerate(questions):
        sims = sim_mat[q_idx]
        sorted_indices = np.argsort(-sims)
        # 1-based rank for each doc index
        idx_to_rank = {doc_idx: rank for rank, doc_idx in enumerate(sorted_indices, start=1)}

        gold_doc_ids = q["gold_supporting_doc_ids"]
        gold_ranks = [idx_to_rank[doc_id_to_idx[did]] for did in gold_doc_ids]
        r_hard = max(gold_ranks)

        per_question_results.append({
            "id": q["id"],
            "group": q["group"],
            "hop_count": q["hop_count"],
            "gold_ranks": gold_ranks,
            "r_hard": r_hard,
            "all_gold_in_top": {k: (r_hard <= k) for k in ks},
        })

    # Group statistics
    groups = ["hop1", "hop2", "hop3", "alias"]
    summary: Dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "num_docs": len(doc_texts),
        "num_questions": len(questions),
        "embed_time_seconds": round(t_embed, 2),
        "groups": {},
    }

    for grp in groups:
        recs = [r for r in per_question_results if r["group"] == grp]
        n = len(recs)
        hard_ranks = [r["r_hard"] for r in recs]
        med_rank = float(np.median(hard_ranks)) if n > 0 else 0.0

        grp_stats: Dict[str, Any] = {
            "n": n,
            "median_hard_rank": med_rank,
            "recall_at_k": {},
        }
        for k in ks:
            cnt = sum(1 for r in recs if r["all_gold_in_top"][k])
            ci = wilson_ci(cnt, n)
            grp_stats["recall_at_k"][k] = {
                "count": cnt,
                "acc": cnt / n if n > 0 else 0.0,
                "ci": ci,
            }
        summary["groups"][grp] = grp_stats

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    print(f"\nCorpus: {summary['corpus_dir']} ({summary['num_docs']} docs, {summary['num_questions']} answerable questions)")
    print(f"Embedding time: {summary['embed_time_seconds']}s")
    print("\n| Group | N | Hard Gold Median Rank | k=1 | k=2 | k=3 | k=4 | k=6 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    ks = [1, 2, 3, 4, 6]
    for grp, st in summary["groups"].items():
        row = [grp, str(st["n"]), f"{st['median_hard_rank']:.1f}"]
        for k in ks:
            m = st["recall_at_k"][k]
            row.append(f"{m['acc']*100:.1f}% ({m['count']}/{st['n']})<br>[{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]")
        print("| " + " | ".join(row) + " |")

    hop2_r4 = summary["groups"]["hop2"]["recall_at_k"][4]["acc"]
    hop3_r4 = summary["groups"]["hop3"]["recall_at_k"][4]["acc"]
    print(f"\nPre-registered Target Check: k=4 recall in [30%, 85%]:")
    print(f"  hop2 recall@4 = {hop2_r4*100:.1f}% (target: 30% - 85%) -> {'PASS' if 0.30 <= hop2_r4 <= 0.85 else 'FAIL'}")
    print(f"  hop3 recall@4 = {hop3_r4*100:.1f}% (target: 30% - 85%) -> {'PASS' if 0.30 <= hop3_r4 <= 0.85 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval recall for synthetic corpus")
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/synth_kg/corpus_S"))
    args = parser.parse_args()

    summary = evaluate_dense_recall(args.corpus_dir)
    print_summary_table(summary)


if __name__ == "__main__":
    main()
