#!/usr/bin/env python3
"""Audit Part 2: Graph-free retrieval baselines on corpus_final."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
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


def normalize_doc_id(did: str) -> str:
    digits = re.findall(r"\d+", did)
    if digits:
        return f"DOC_{int(digits[0]):04d}"
    return did.strip().upper()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class BM25Index:
    def __init__(self, doc_ids: List[str], doc_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.N = len(doc_ids)
        self.doc_tokens = [tokenize(t) for t in doc_texts]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0

        # Document frequencies
        self.df: Dict[str, int] = collections.defaultdict(int)
        self.tf: List[Dict[str, int]] = []
        for toks in self.doc_tokens:
            counts = collections.Counter(toks)
            self.tf.append(counts)
            for term in counts:
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(((self.N - n_q + 0.5) / (n_q + 0.5)) + 1.0)

    def score(self, query: str) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = []
        for idx, did in enumerate(self.doc_ids):
            dl = self.doc_lens[idx]
            tf_dict = self.tf[idx]
            score = 0.0
            for term in q_tokens:
                if term in tf_dict:
                    freq = tf_dict[term]
                    idf_val = self.idf(term)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                    score += idf_val * (numerator / denominator)
            scores.append((did, score))
        scores.sort(key=lambda x: -x[1])
        return scores


def run_part2() -> None:
    log("================================================================================")
    log("PART 2 - GRAPH-FREE RETRIEVAL BASELINES EVALUATION")
    log("================================================================================\n")

    # Load corpus documents
    docs_dir = SYNTH_DIR / "corpus_final" / "docs"
    doc_files = sorted(docs_dir.glob("*.txt"), key=lambda p: p.name)
    doc_ids = [p.stem for p in doc_files]
    doc_texts = [p.read_text(encoding="utf-8").strip() for p in doc_files]
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    # Load questions (105 answerable)
    questions_file = SYNTH_DIR / "corpus_final" / "questions.jsonl"
    with open(questions_file, "r", encoding="utf-8") as f:
        all_questions = [json.loads(l) for l in f if l.strip()]
    questions = [q for q in all_questions if q.get("group") != "unanswerable"]

    log(f"Evaluating 5 baselines on {len(questions)} answerable questions against {len(doc_ids)} documents.\n")

    # Load dense embeddings from Part 1(a)
    emb_file = AUDIT_DIR / "dense_embeddings.npz"
    if not emb_file.exists():
        raise FileNotFoundError(f"Missing {emb_file}; run Part 1(a) first.")
    data = np.load(emb_file)
    doc_mat = data["doc_mat"]
    q_mat = data["q_mat"]

    # Normalize vectors
    doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True)
    doc_norms[doc_norms == 0.0] = 1.0
    doc_mat = doc_mat / doc_norms

    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_norms[q_norms == 0.0] = 1.0
    q_mat = q_mat / q_norms

    # Cosine similarity matrix: (105, 60)
    sim_mat = np.dot(q_mat, doc_mat.T)

    # Initialize BM25
    bm25 = BM25Index(doc_ids, doc_texts, k1=1.5, b=0.75)

    # Name extraction regex for B4
    name_pattern = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")

    # Baseline methods to evaluate
    methods = ["B1_dense_top4", "B2_bm25_top4", "B3_rrf_top4", "B4_name_expansion_top4", "B6_dense_top8"]

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        m: {grp: {"total": 0, "correct": 0} for grp in ["hop1", "hop2", "hop3", "alias"]}
        for m in methods
    }

    for q_idx, q in enumerate(questions):
        qid = q["id"]
        grp = q["group"]
        qtxt = q["question"]
        gold_docs = {normalize_doc_id(d) for d in q.get("gold_supporting_doc_ids", [])}

        # 1. B1: Dense Top-4 & B6: Dense Top-8
        sims = sim_mat[q_idx]
        dense_sorted_indices = np.argsort(-sims)
        dense_ranked_doc_ids = [doc_ids[idx] for idx in dense_sorted_indices]
        b1_docs = {normalize_doc_id(d) for d in dense_ranked_doc_ids[:4]}
        b6_docs = {normalize_doc_id(d) for d in dense_ranked_doc_ids[:8]}

        # 2. B2: BM25 Top-4
        bm25_scored = bm25.score(qtxt)
        bm25_ranked_doc_ids = [did for did, sc in bm25_scored]
        b2_docs = {normalize_doc_id(d) for d in bm25_ranked_doc_ids[:4]}

        # 3. B3: RRF Top-4 (k=60)
        dense_rank_map = {did: rank for rank, did in enumerate(dense_ranked_doc_ids, start=1)}
        bm25_rank_map = {did: rank for rank, did in enumerate(bm25_ranked_doc_ids, start=1)}
        rrf_scores = []
        for did in doc_ids:
            r_d = dense_rank_map[did]
            r_b = bm25_rank_map[did]
            score = (1.0 / (60.0 + r_d)) + (1.0 / (60.0 + r_b))
            rrf_scores.append((did, score))
        rrf_scores.sort(key=lambda x: -x[1])
        b3_docs = {normalize_doc_id(did) for did, sc in rrf_scores[:4]}

        # 4. B4: Name Expansion Top-4
        # Seeds = dense top-2 docs
        seed_dids = dense_ranked_doc_ids[:2]
        seed_indices = {doc_id_to_idx[did] for did in seed_dids}
        # Extract names from seeds
        seed_texts = [doc_texts[doc_id_to_idx[did]] for did in seed_dids]
        extracted_names = set()
        for st in seed_texts:
            for match in name_pattern.findall(st):
                if len(match.strip()) > 1:
                    extracted_names.add(match.strip())

        # Candidates = other docs whose text contains any extracted name
        candidates = []
        for did in doc_ids:
            if did in seed_dids:
                continue
            dtext = doc_texts[doc_id_to_idx[did]]
            if any(name in dtext for name in extracted_names):
                candidates.append(did)

        # Rank candidates by cosine similarity to query
        candidate_sims = [(did, sims[doc_id_to_idx[did]]) for did in candidates]
        candidate_sims.sort(key=lambda x: -x[1])
        ranked_candidates = [did for did, sc in candidate_sims]

        # Return 2 seeds + top 2 candidates; if < 2, backfill from dense ranking
        b4_list = list(seed_dids)
        for cdid in ranked_candidates:
            if len(b4_list) >= 4:
                break
            if cdid not in b4_list:
                b4_list.append(cdid)
        # Backfill if fewer than 4
        if len(b4_list) < 4:
            for ddid in dense_ranked_doc_ids:
                if len(b4_list) >= 4:
                    break
                if ddid not in b4_list:
                    b4_list.append(ddid)
        b4_docs = {normalize_doc_id(d) for d in b4_list[:4]}

        # Score all methods for this question
        method_doc_sets = {
            "B1_dense_top4": b1_docs,
            "B2_bm25_top4": b2_docs,
            "B3_rrf_top4": b3_docs,
            "B4_name_expansion_top4": b4_docs,
            "B6_dense_top8": b6_docs,
        }

        for m, m_docs in method_doc_sets.items():
            results[m][grp]["total"] += 1
            if gold_docs.issubset(m_docs):
                results[m][grp]["correct"] += 1

    # Load existing engine modes evidence recall from evaluation_results.json
    eval_file = SYNTH_DIR / "evaluation_results.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    mode_sums = eval_data.get("mode_summaries", {})

    log("--- Graph-Free Retrieval Baselines vs. Engine Modes: Evidence Recall ---")
    log("| Method / Mode | Budget K | hop1 (N=30) | hop2 (N=30) | hop3 (N=30) | alias (N=15) |")
    log("| :--- | :--- | :--- | :--- | :--- | :--- |")

    groups = ["hop1", "hop2", "hop3", "alias"]
    for m in methods:
        k_val = "K=8" if "top8" in m else "K=4"
        row_strs = []
        for grp in groups:
            tot = results[m][grp]["total"]
            corr = results[m][grp]["correct"]
            acc = corr / tot if tot > 0 else 0.0
            low, high = wilson_ci(corr, tot)
            row_strs.append(f"{acc*100:.1f}% [{low*100:.1f}%, {high*100:.1f}%]")
        log(f"| `{m}` | {k_val} | " + " | ".join(row_strs) + " |")

    log("| :--- | :--- | :--- | :--- | :--- | :--- |")
    # Existing engine modes
    for em in ["vector_only", "graph_only", "hybrid", "community_aware"]:
        row_strs = []
        for grp in groups:
            gdata = mode_sums.get(em, {}).get("by_group", {}).get(grp, {})
            acc = gdata.get("evidence_recall", 0.0)
            corr = int(round(acc * (15 if grp == "alias" else 30)))
            tot = 15 if grp == "alias" else 30
            low, high = wilson_ci(corr, tot)
            row_strs.append(f"{acc*100:.1f}% [{low*100:.1f}%, {high*100:.1f}%]")
        log(f"| Engine: `{em}` | K=4 | " + " | ".join(row_strs) + " |")
    log("")

    # Evaluate Decision Rule on hop2
    H = mode_sums["hybrid"]["by_group"]["hop2"]["evidence_recall"] * 100.0  # 56.7%
    log("--- Pre-Registered Decision Rule Evaluation on hop2 ---")
    log(f"Hybrid hop2 all-gold evidence recall (H): {H:.1f}% (17/30)")

    rec_b2 = (results["B2_bm25_top4"]["hop2"]["correct"] / 30) * 100.0
    rec_b3 = (results["B3_rrf_top4"]["hop2"]["correct"] / 30) * 100.0
    rec_b4 = (results["B4_name_expansion_top4"]["hop2"]["correct"] / 30) * 100.0

    b_star = max(rec_b2, rec_b3, rec_b4)
    best_baseline = [name for name, val in [("B2 (BM25)", rec_b2), ("B3 (RRF)", rec_b3), ("B4 (Name Expansion)", rec_b4)] if val == b_star][0]

    log(f"- B2 (BM25 Top-4) hop2 recall: {rec_b2:.1f}% ({results['B2_bm25_top4']['hop2']['correct']}/30)")
    log(f"- B3 (RRF Top-4) hop2 recall: {rec_b3:.1f}% ({results['B3_rrf_top4']['hop2']['correct']}/30)")
    log(f"- B4 (Name Expansion Top-4) hop2 recall: {rec_b4:.1f}% ({results['B4_name_expansion_top4']['hop2']['correct']}/30)")
    log(f"Best graph-free baseline B* on hop2: {best_baseline} with {b_star:.1f}%")
    log(f"Comparison: H - B* = {H - b_star:.1f} percentage points (H - 10 = {H - 10:.1f}%, H - 20 = {H - 20:.1f}%)")

    if b_star >= H - 10.0:
        outcome = "graph-free methods reproduce most of the evidence-recall gain; graph-specific benefit not demonstrated"
    elif b_star < H - 20.0:
        outcome = "graph-specific benefit supported on this corpus"
    else:
        outcome = "ambiguous"

    log(f"\nDECISION RULE OUTCOME: \"{outcome}\"")
    log(f"Explanation: The best graph-free baseline reaches {b_star:.1f}%, which is {H - b_star:.1f} percentage points below Hybrid's {H:.1f}% (threshold for support was >20 points below, i.e. <{H - 20:.1f}%).")
    log("\n")


if __name__ == "__main__":
    run_part2()
