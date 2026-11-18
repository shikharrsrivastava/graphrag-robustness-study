#!/usr/bin/env python3
"""Part 2: Pooled Dense Retrieval Evaluation over 200 HotpotQA Questions.

Performs:
1. Builds deduplicated pooled corpus from the 10 context paragraphs of 200 questions.
2. Embeds pooled corpus and questions with embeddinggemma (OllamaEmbeddingService).
3. Saves embeddings to experiments/hotpotqa_m4/embeddings/.
4. Computes cosine similarities and ranks of gold paragraphs.
5. Evaluates recall@k (both gold, at least one gold, hard gold) for k in {3, 5, 10, 20}
   overall, bridge, comparison with Wilson 95% CIs.
6. Computes hard-gold median rank.
7. Contrasts with own-10 recall@3 to quantify the drop from own-10 to pooled setting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaConfig


def make_chunk_text(title: str, sentences: List[str]) -> str:
    """Format paragraph text identically to M3 condition C."""
    joined = " ".join(s.strip() for s in sentences if s.strip())
    return f"Title: {title}\n{joined}"


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


def log_progress(progress_file: Path, step_name: str, output_path: str) -> None:
    """Append one line to PROGRESS.md with ISO timestamp."""
    now_iso = datetime.now().astimezone().isoformat()
    line = f"| {now_iso} | {step_name} | `{output_path}` |\n"
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2: Pooled Dense Retrieval Evaluation")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("experiments/hotpotqa_m4/pooled_questions.json"),
        help="Path to 200 sampled questions JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/hotpotqa_m4"),
        help="Root output directory",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("experiments/hotpotqa_m4/embeddings"),
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for embedding generation",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.output_dir / "PROGRESS.md"

    # 1. Load questions
    print(f"Loading questions from {args.questions_file}...")
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions.")

    # 2. Build Pooled Corpus with Deduplication
    print("\nBuilding pooled corpus from context paragraphs...")
    pooled_corpus: List[Dict[str, Any]] = []
    text_to_pid: Dict[str, int] = {}
    total_paragraphs_seen = 0
    duplicates_removed = 0

    # For each question, record its own 10 pids and its 2 gold pids
    question_records: List[Dict[str, Any]] = []

    for q_idx, q in enumerate(questions):
        qid = q["_id"]
        q_type = q["type"]
        q_text = q["question"]
        gold_titles = set(title for title, _ in q["supporting_facts"])

        own_pids: List[int] = []
        gold_pids: List[int] = []

        for title, sents in q["context"]:
            total_paragraphs_seen += 1
            chunk_txt = make_chunk_text(title, sents)
            dedup_key = f"{title.strip()}:::{chunk_txt.strip()}"

            if dedup_key in text_to_pid:
                pid = text_to_pid[dedup_key]
                duplicates_removed += 1
            else:
                pid = len(pooled_corpus)
                text_to_pid[dedup_key] = pid
                pooled_corpus.append({
                    "pid": pid,
                    "title": title,
                    "sentences": sents,
                    "chunk_text": chunk_txt,
                })

            own_pids.append(pid)
            if title in gold_titles and pid not in gold_pids:
                gold_pids.append(pid)

        assert len(gold_pids) == 2, f"Question {qid} does not have exactly 2 gold paragraphs in its context! Found: {len(gold_pids)}"

        question_records.append({
            "qid": qid,
            "type": q_type,
            "question": q_text,
            "own_pids": own_pids,
            "gold_pids": gold_pids,
        })

    pooled_corpus_size = len(pooled_corpus)
    print(f"Total context paragraphs processed: {total_paragraphs_seen}")
    print(f"Duplicates removed:                 {duplicates_removed}")
    print(f"Unique pooled corpus size (N):      {pooled_corpus_size}")

    # Save corpus metadata
    corpus_meta_path = args.output_dir / "pooled_corpus.json"
    with open(corpus_meta_path, "w", encoding="utf-8") as f:
        json.dump([{"pid": p["pid"], "title": p["title"]} for p in pooled_corpus], f, indent=2)
    print(f"Saved corpus metadata to {corpus_meta_path}")

    # 3. Embed Corpus and Questions
    print("\nInitializing OllamaEmbeddingService (model: embeddinggemma)...")
    service = OllamaEmbeddingService(config=OllamaConfig())

    t0_embed = time.time()

    # Embed Questions
    print(f"Embedding {len(questions)} questions...")
    q_texts = [q["question"] for q in questions]
    q_vecs_list = []
    for i in range(0, len(q_texts), args.batch_size):
        batch = q_texts[i : i + args.batch_size]
        q_vecs_list.extend(service.embed_batch(batch))
        print(f"  Embedded questions [{min(i + args.batch_size, len(q_texts))}/{len(q_texts)}]")
    q_embeddings = np.array(q_vecs_list, dtype=np.float32)

    # Embed Corpus Paragraphs
    print(f"\nEmbedding {pooled_corpus_size} pooled corpus paragraphs in batches of {args.batch_size}...")
    corpus_texts = [p["chunk_text"] for p in pooled_corpus]
    corpus_vecs_list = []
    for i in range(0, len(corpus_texts), args.batch_size):
        batch = corpus_texts[i : i + args.batch_size]
        corpus_vecs_list.extend(service.embed_batch(batch))
        print(f"  Embedded corpus [{min(i + args.batch_size, len(corpus_texts))}/{len(corpus_texts)}]")
    corpus_embeddings = np.array(corpus_vecs_list, dtype=np.float32)

    t_embed_wall = time.time() - t0_embed
    print(f"\nEmbedding completed in {t_embed_wall:.2f}s ({t_embed_wall / 60:.2f} min).")

    # Normalize vectors for cosine similarity
    q_norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
    q_norms[q_norms == 0.0] = 1.0
    q_normed = q_embeddings / q_norms

    c_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    c_norms[c_norms == 0.0] = 1.0
    c_normed = corpus_embeddings / c_norms

    # Save embeddings to disk
    q_emb_path = args.embeddings_dir / "question_embeddings.npy"
    c_emb_path = args.embeddings_dir / "corpus_embeddings.npy"
    np.save(q_emb_path, q_normed)
    np.save(c_emb_path, c_normed)
    print(f"Saved question embeddings ({q_normed.shape}) to {q_emb_path}")
    print(f"Saved corpus embeddings ({c_normed.shape}) to {c_emb_path}")

    log_progress(
        progress_file,
        f"Part 2(b,c): Built corpus ({pooled_corpus_size} paras, {duplicates_removed} dups) and generated embeddings ({t_embed_wall:.1f}s)",
        str(c_emb_path),
    )

    # 4. Compute Similarities and Gold Paragraph Ranks
    print("\nComputing cosine similarity matrices and gold paragraph rankings...")
    # Matrix of similarities: shape (200, N)
    sim_matrix = np.dot(q_normed, c_normed.T)

    ks = [3, 5, 10, 20]
    eval_results: List[Dict[str, Any]] = []

    for q_idx, q_meta in enumerate(question_records):
        qid = q_meta["qid"]
        q_type = q_meta["type"]
        gold_pids = q_meta["gold_pids"]
        own_pids = q_meta["own_pids"]

        q_sims = sim_matrix[q_idx]

        # Pooled setting ranks: sort all N paragraphs descending
        sorted_pids = np.argsort(-q_sims)
        # pid -> 1-based rank
        pid_to_rank = {pid: rank for rank, pid in enumerate(sorted_pids, start=1)}

        gold_ranks = [pid_to_rank[g_pid] for g_pid in gold_pids]
        r_easy = min(gold_ranks)
        r_hard = max(gold_ranks)

        # Own-10 setting: rank only within own_pids
        own_sims = [(pid, q_sims[pid]) for pid in own_pids]
        own_sims.sort(key=lambda x: x[1], reverse=True)
        own_pid_to_rank = {pid: rank for rank, (pid, _) in enumerate(own_sims, start=1)}
        own_gold_ranks = [own_pid_to_rank[g_pid] for g_pid in gold_pids]
        own_both_top3 = max(own_gold_ranks) <= 3

        eval_results.append({
            "qid": qid,
            "type": q_type,
            "gold_ranks": gold_ranks,
            "r_easy": r_easy,
            "r_hard": r_hard,
            "both_in_top": {k: (r_hard <= k) for k in ks},
            "at_least_one_in_top": {k: (r_easy <= k) for k in ks},
            "hard_in_top": {k: (r_hard <= k) for k in ks},
            "own_10_both_top3": own_both_top3,
            "own_gold_ranks": own_gold_ranks,
        })

    # Save detailed evaluation per question
    detailed_eval_path = args.output_dir / "pooled_retrieval_results.json"
    with open(detailed_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved detailed retrieval results to {detailed_eval_path}")

    # 5. Aggregate Metrics: Overall, Bridge, Comparison
    def aggregate_subset(recs: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n = len(recs)
        hard_ranks = [r["r_hard"] for r in recs]
        median_hard_rank = float(np.median(hard_ranks)) if n > 0 else 0.0

        stats: Dict[str, Any] = {
            "subset": label,
            "n": n,
            "median_hard_rank": median_hard_rank,
            "hard_rank_p25": float(np.percentile(hard_ranks, 25)),
            "hard_rank_p75": float(np.percentile(hard_ranks, 75)),
            "both_in_top": {},
            "at_least_one_in_top": {},
            "hard_in_top": {},
            "own_10_recall3": {},
            "drop_recall3": {},
        }

        for k in ks:
            # Both in top-k
            both_cnt = sum(1 for r in recs if r["both_in_top"][k])
            b_ci = wilson_ci(both_cnt, n)
            stats["both_in_top"][k] = {
                "count": both_cnt,
                "acc": both_cnt / n,
                "ci": b_ci,
            }

            # At least one in top-k
            one_cnt = sum(1 for r in recs if r["at_least_one_in_top"][k])
            o_ci = wilson_ci(one_cnt, n)
            stats["at_least_one_in_top"][k] = {
                "count": one_cnt,
                "acc": one_cnt / n,
                "ci": o_ci,
            }

            # Hard gold in top-k
            hard_cnt = sum(1 for r in recs if r["hard_in_top"][k])
            h_ci = wilson_ci(hard_cnt, n)
            stats["hard_in_top"][k] = {
                "count": hard_cnt,
                "acc": hard_cnt / n,
                "ci": h_ci,
            }

        # Own-10 recall@3
        own_both_cnt = sum(1 for r in recs if r["own_10_both_top3"])
        stats["own_10_recall3"] = {
            "count": own_both_cnt,
            "acc": own_both_cnt / n,
            "ci": wilson_ci(own_both_cnt, n),
        }

        # Drop: Own-10 recall@3 minus Pooled recall@3
        pooled_r3 = stats["both_in_top"][3]["acc"]
        own_r3 = stats["own_10_recall3"]["acc"]
        stats["drop_recall3"] = {
            "own_10": own_r3,
            "pooled": pooled_r3,
            "delta": own_r3 - pooled_r3,
            "drop_pct": ((own_r3 - pooled_r3) / own_r3 * 100.0) if own_r3 > 0 else 0.0,
        }

        return stats

    overall_stats = aggregate_subset(eval_results, "Overall")
    bridge_stats = aggregate_subset([r for r in eval_results if r["type"] == "bridge"], "Bridge")
    comp_stats = aggregate_subset([r for r in eval_results if r["type"] == "comparison"], "Comparison")

    # 6. Format and Print Markdown Tables
    print("\n" + "=" * 90)
    print("PART 2 RESULTS SUMMARY: POOLED DENSE RECALL (EMBEDDINGGEMMA)")
    print("=" * 90)

    print(f"\nCorpus Size: {pooled_corpus_size} unique paragraphs ({duplicates_removed} duplicates removed from {total_paragraphs_seen})")
    print(f"Embedding Time: {t_embed_wall:.2f}s ({len(questions)} questions + {pooled_corpus_size} paragraphs)")

    print("\n### Table 1: Pooled Recall@k Metrics (Both Gold, At Least One Gold, Hard Gold)")
    print("| Metric | Subset | N | k=3 | k=5 | k=10 | k=20 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for subset_name, st in [("Overall", overall_stats), ("Bridge", bridge_stats), ("Comparison", comp_stats)]:
        # (i) Both gold in top-k
        row_both = [f"Both Gold in Top-k", subset_name, str(st["n"])]
        for k in ks:
            m = st["both_in_top"][k]
            row_both.append(f"{m['acc']*100:.1f}% ({m['count']}/{st['n']})<br>[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
        print("| " + " | ".join(row_both) + " |")

        # (ii) At least one gold in top-k
        row_one = [f"At Least One Gold in Top-k", subset_name, str(st["n"])]
        for k in ks:
            m = st["at_least_one_in_top"][k]
            row_one.append(f"{m['acc']*100:.1f}% ({m['count']}/{st['n']})<br>[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
        print("| " + " | ".join(row_one) + " |")

        # Hard gold recall (same as both gold, but reported explicitly)
        row_hard = [f"Hard Gold in Top-k", subset_name, str(st["n"])]
        for k in ks:
            m = st["hard_in_top"][k]
            row_hard.append(f"{m['acc']*100:.1f}% ({m['count']}/{st['n']})<br>[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
        print("| " + " | ".join(row_hard) + " |")

    print("\n### Table 2: Hard Gold Rank Statistics & Drop from Own-10 to Pooled Setting")
    print("| Subset | N | Hard Gold Median Rank [IQR] | Own-10 Recall@3 (95% CI) | Pooled Recall@3 (95% CI) | Absolute Drop (Δ) | Relative Drop |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for subset_name, st in [("Overall", overall_stats), ("Bridge", bridge_stats), ("Comparison", comp_stats)]:
        med_str = f"{st['median_hard_rank']:.1f} [{st['hard_rank_p25']:.1f} - {st['hard_rank_p75']:.1f}]"
        own_m = st["own_10_recall3"]
        own_str = f"{own_m['acc']*100:.1f}% ({own_m['count']}/{st['n']}) [{own_m['ci'][0]:.3f}, {own_m['ci'][1]:.3f}]"
        pool_m = st["both_in_top"][3]
        pool_str = f"{pool_m['acc']*100:.1f}% ({pool_m['count']}/{st['n']}) [{pool_m['ci'][0]:.3f}, {pool_m['ci'][1]:.3f}]"
        delta_str = f"-{st['drop_recall3']['delta']*100:.1f} pts"
        rel_str = f"-{st['drop_recall3']['drop_pct']:.1f}%"
        print(f"| {subset_name} | {st['n']} | {med_str} | {own_str} | {pool_str} | {delta_str} | {rel_str} |")

    # Save summary stats JSON
    summary_stats_path = args.output_dir / "pooled_recall_summary.json"
    with open(summary_stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "pooled_corpus_size": pooled_corpus_size,
            "total_context_paragraphs": total_paragraphs_seen,
            "duplicates_removed": duplicates_removed,
            "embedding_wall_time_seconds": t_embed_wall,
            "overall": overall_stats,
            "bridge": bridge_stats,
            "comparison": comp_stats,
        }, f, indent=2)
    print(f"\nSaved summary statistics to {summary_stats_path}")

    log_progress(
        progress_file,
        f"Part 2(d,e): Evaluated pooled retrieval metrics (Overall Recall@3: {overall_stats['both_in_top'][3]['acc']*100:.1f}% vs Own-10: {overall_stats['own_10_recall3']['acc']*100:.1f}%)",
        str(summary_stats_path),
    )


if __name__ == "__main__":
    main()
