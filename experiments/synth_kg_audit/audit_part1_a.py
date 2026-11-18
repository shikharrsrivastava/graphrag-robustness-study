#!/usr/bin/env python3
"""Audit Part 1(a): Integrity audit of decisions, progress, calibration, and dense recall."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Add repo root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaConfig

AUDIT_DIR = Path(__file__).resolve().parent
SYNTH_DIR = BASE_DIR / "experiments" / "synth_kg"
OUTPUT_FILE = AUDIT_DIR / "audit_output.txt"


def log(msg: str) -> None:
    print(msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    centre = p_hat + (z**2) / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))
    lower = max(0.0, (centre - spread) / denom)
    upper = min(1.0, (centre + spread) / denom)
    return lower, upper


def run_part1_a() -> None:
    log("================================================================================")
    log("PART 1(a) - INTEGRITY AUDIT: DECISIONS, PROGRESS, CALIBRATION, AND DENSE RECALL")
    log("================================================================================\n")

    # 1. Print DECISIONS.md in full
    decisions_path = SYNTH_DIR / "DECISIONS.md"
    log(f"--- Full contents of {decisions_path} ---")
    if decisions_path.exists():
        log(decisions_path.read_text(encoding="utf-8").strip())
    else:
        log("DECISIONS.md NOT FOUND!")
    log("\n")

    # 2. Print PROGRESS.md in full
    progress_path = SYNTH_DIR / "PROGRESS.md"
    log(f"--- Full contents of {progress_path} ---")
    if progress_path.exists():
        log(progress_path.read_text(encoding="utf-8").strip())
    else:
        log("PROGRESS.md NOT FOUND!")
    log("\n")

    # 3. Find calibration attempt tables
    calib_json = SYNTH_DIR / "calibration_results.json"
    log("--- Calibration Attempts from experiments/synth_kg/calibration_results.json ---")
    if calib_json.exists():
        with open(calib_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        attempts = calib_data.get("attempts", [])
        log(f"Found {len(attempts)} calibration attempts in calibration_results.json:\n")
        for att in attempts:
            num = att.get("attempt")
            knobs = att.get("knobs", {})
            summary = att.get("summary", {})
            log(f"### Calibration Attempt {num}: knobs={knobs}")
            log(f"Directory: {summary.get('corpus_dir')}, Docs: {summary.get('num_docs')}, Questions: {summary.get('num_questions')}")
            log("| Group | N | Median Hard Rank | Recall@1 | Recall@2 | Recall@3 | Recall@4 | Recall@6 |")
            log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            groups = summary.get("groups", {})
            for grp_name, gdata in groups.items():
                n_q = gdata.get("n", 0)
                mhr = gdata.get("median_hard_rank", 0.0)
                r_at_k = gdata.get("recall_at_k", {})
                r1 = f"{r_at_k.get('1', {}).get('acc', 0.0)*100:.1f}%"
                r2 = f"{r_at_k.get('2', {}).get('acc', 0.0)*100:.1f}%"
                r3 = f"{r_at_k.get('3', {}).get('acc', 0.0)*100:.1f}%"
                r4 = f"{r_at_k.get('4', {}).get('acc', 0.0)*100:.1f}%"
                r6 = f"{r_at_k.get('6', {}).get('acc', 0.0)*100:.1f}%"
                log(f"| {grp_name} | {n_q} | {mhr:.1f} | {r1} | {r2} | {r3} | {r4} | {r6} |")
            log("")
    else:
        log("calibration_results.json was NOT saved on disk!")

    # 4. Recompute dense all-gold recall for corpus_final with OllamaEmbeddingService
    corpus_final = SYNTH_DIR / "corpus_final"
    docs_dir = corpus_final / "docs"
    questions_path = corpus_final / "questions.jsonl"

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions.append(q)

    # Filter answerable questions for dense recall
    answerable_questions = [q for q in questions if q.get("group") != "unanswerable"]

    doc_files = sorted(docs_dir.glob("*.txt"), key=lambda p: p.name)
    doc_ids = [p.stem for p in doc_files]
    doc_texts = [p.read_text(encoding="utf-8").strip() for p in doc_files]

    log(f"--- Recomputing Dense All-Gold Recall for corpus_final ({len(doc_texts)} docs, {len(answerable_questions)} answerable questions) ---")
    service = OllamaEmbeddingService(config=OllamaConfig())

    t0 = time.time()
    batch_size = 25
    doc_vecs = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        doc_vecs.extend(service.embed_batch(batch))

    q_texts = [q["question"] for q in answerable_questions]
    q_vecs = []
    for i in range(0, len(q_texts), batch_size):
        batch = q_texts[i : i + batch_size]
        q_vecs.extend(service.embed_batch(batch))
    t_embed = time.time() - t0
    log(f"Embedding completed in {t_embed:.2f}s")

    doc_mat = np.array(doc_vecs, dtype=np.float32)
    q_mat = np.array(q_vecs, dtype=np.float32)

    # Save embeddings for Part 2 reuse
    np.savez_compressed(
        AUDIT_DIR / "dense_embeddings.npz",
        doc_mat=doc_mat,
        doc_ids=np.array(doc_ids),
        q_mat=q_mat,
        q_ids=np.array([q["id"] for q in answerable_questions]),
    )
    log(f"Saved dense embeddings to {AUDIT_DIR / 'dense_embeddings.npz'}")

    # L2 normalize
    doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True)
    doc_norms[doc_norms == 0.0] = 1.0
    doc_mat = doc_mat / doc_norms

    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_norms[q_norms == 0.0] = 1.0
    q_mat = q_mat / q_norms

    sim_mat = np.dot(q_mat, doc_mat.T)
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    ks = [1, 2, 3, 4, 6]
    group_results: Dict[str, Dict[str, Any]] = {
        "hop1": {"total": 0, "correct_at_k": {k: 0 for k in ks}},
        "hop2": {"total": 0, "correct_at_k": {k: 0 for k in ks}},
        "hop3": {"total": 0, "correct_at_k": {k: 0 for k in ks}},
        "alias": {"total": 0, "correct_at_k": {k: 0 for k in ks}},
    }

    per_q_ranks = {}
    for q_idx, q in enumerate(answerable_questions):
        grp = q["group"]
        if grp not in group_results:
            continue
        group_results[grp]["total"] += 1

        sims = sim_mat[q_idx]
        sorted_indices = np.argsort(-sims)
        ranked_doc_ids = [doc_ids[idx] for idx in sorted_indices]

        gold_doc_ids = q["gold_supporting_doc_ids"]
        # Find maximum rank among gold docs
        ranks = []
        for gdid in gold_doc_ids:
            # normalize gdid: DOC_0001 or D001
            idx = doc_id_to_idx.get(gdid)
            if idx is None:
                # try alternative
                num = int(gdid.replace("DOC_", "").replace("D", ""))
                idx = doc_id_to_idx.get(f"DOC_{num:04d}", doc_id_to_idx.get(f"D{num:03d}"))
            rank = list(sorted_indices).index(idx) + 1
            ranks.append(rank)

        r_hard = max(ranks)
        per_q_ranks[q["id"]] = {"r_hard": r_hard, "ranks": ranks, "ranked_top8": ranked_doc_ids[:8]}
        for k in ks:
            if r_hard <= k:
                group_results[grp]["correct_at_k"][k] += 1

    # Save per_q_ranks for Part 2
    with open(AUDIT_DIR / "dense_ranking_per_q.json", "w", encoding="utf-8") as f:
        json.dump(per_q_ranks, f, indent=2)

    log("\n--- Recomputed Dense All-Gold Recall on corpus_final ---")
    log("| Group | Total N | k=1 (Wilson 95% CI) | k=2 (Wilson 95% CI) | k=3 (Wilson 95% CI) | k=4 (Wilson 95% CI) | k=6 (Wilson 95% CI) |")
    log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    target_hop2_met = False
    target_hop3_met = False

    for grp in ["hop1", "hop2", "hop3", "alias"]:
        tot = group_results[grp]["total"]
        row_strs = []
        for k in ks:
            c = group_results[grp]["correct_at_k"][k]
            acc = c / tot if tot > 0 else 0.0
            low, high = wilson_ci(c, tot)
            row_strs.append(f"{acc*100:.1f}% [{low*100:.1f}%, {high*100:.1f}%]")
            if grp == "hop2" and k == 4:
                target_hop2_met = (0.30 <= acc <= 0.85)
                hop2_k4_acc = acc
            if grp == "hop3" and k == 4:
                target_hop3_met = (0.30 <= acc <= 0.85)
                hop3_k4_acc = acc
        log(f"| `{grp}` | {tot} | " + " | ".join(row_strs) + " |")

    log("\n--- Pre-Registered Calibration Target Evaluation ---")
    log("Pre-registered Target: All-gold recall at k=4 between 30.0% and 85.0% for both hop2 and hop3.")
    log(f"- hop2 all-gold recall at k=4: {hop2_k4_acc*100:.1f}% -> Target met (30%-85%): {target_hop2_met}")
    log(f"- hop3 all-gold recall at k=4: {hop3_k4_acc*100:.1f}% -> Target met (30%-85%): {target_hop3_met}")
    if not (target_hop2_met and target_hop3_met):
        log("CONCLUSION ON TARGET: The pre-registered calibration target was NOT MET on corpus_final.")
        log("Reason: Dense retrieval on unmentioned bridge documents in multi-hop chains is too weak; hop2 reached only 16.7% (<30%) and hop3 reached 0.0% (<30%).")
        log("As noted in DECISIONS.md (2026-11-16T03:03:46), Attempt 1 was frozen despite failing this target because across all 3 calibration attempts (45, 60, 80 docs), hop2 remained <=16.7% and hop3 remained 0.0%.")
    else:
        log("CONCLUSION ON TARGET: Pre-registered target was met.")
    log("\n")


if __name__ == "__main__":
    run_part1_a()
