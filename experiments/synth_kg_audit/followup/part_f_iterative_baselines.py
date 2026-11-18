#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_f_iterative_baselines.py
Evaluation of Stronger Graph-Free Iterative Baselines (B7, B8, B9, B4).
"""

import json
import math
import re
from pathlib import Path
import datetime
import numpy as np

def wilson_score_interval(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2))))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def exact_binomial_sign_test(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p_val = 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, p_val)

def main():
    # Load embeddings
    emb_data = np.load("experiments/synth_kg_audit/dense_embeddings.npz")
    doc_mat = emb_data["doc_mat"]
    doc_ids = list(emb_data["doc_ids"])
    q_mat = emb_data["q_mat"]
    q_ids = list(emb_data["q_ids"])

    q2_data = np.load("experiments/synth_kg_audit/followup/q2_embeddings.npz")
    q2_mat = q2_data["q2_mat"]

    # Load questions
    questions = [json.loads(line) for line in open("experiments/synth_kg/corpus_final/questions.jsonl", encoding="utf-8")]
    q_by_id = {q["id"]: q for q in questions if q["id"] in q_ids}

    # Load doc texts
    docs = {}
    for did in doc_ids:
        docs[did] = Path(f"experiments/synth_kg/corpus_final/docs/{did}.txt").read_text(encoding="utf-8")

    # Load hybrid results
    hyb_results = {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_045328_hybrid/results.jsonl", encoding="utf-8")]}

    NAME_REGEX = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")

    # Recompute Hybrid Success
    hop2_qids = [qid for qid in q_ids if q_by_id[qid]["group"] == "hop2"]
    hyb_success = {g: [] for g in ["hop1", "hop2", "hop3", "alias", "all"]}
    hyb_success_by_qid = {}
    for qid in q_ids:
        grp = q_by_id[qid]["group"]
        gdocs = set(q_by_id[qid]["gold_supporting_doc_ids"])
        hyb_docs = {f"DOC_{int(d[1:]):04d}" for d in hyb_results[qid].get("context_doc_ids", [])}
        ok = gdocs.issubset(hyb_docs)
        hyb_success[grp].append(ok)
        hyb_success["all"].append(ok)
        hyb_success_by_qid[qid] = ok

    methods = ["B7_iter_dense", "B8_overlap_top2", "B9_overlap_top1", "B4_name_exp_top4"]
    results = {m: {g: [] for g in ["hop1", "hop2", "hop3", "alias", "all"]} for m in methods}
    success_per_q = {m: {} for m in methods}

    for i, qid in enumerate(q_ids):
        q_item = q_by_id[qid]
        grp = q_item["group"]
        gdocs = set(q_item["gold_supporting_doc_ids"])
        q_text = q_item["question"]

        dense_sims = np.dot(doc_mat, q_mat[i])
        dense_ranked_indices = np.argsort(-dense_sims)
        dense_ranked_docs = [doc_ids[idx] for idx in dense_ranked_indices]

        # B7: Iterative Dense
        d1 = dense_ranked_docs[0]
        q2_sims = np.dot(doc_mat, q2_mat[i])
        q2_ranked_indices = np.argsort(-q2_sims)
        q2_other_docs = [doc_ids[idx] for idx in q2_ranked_indices if doc_ids[idx] != d1]
        b7_retrieved = [d1] + q2_other_docs[:3]
        b7_ok = gdocs.issubset(set(b7_retrieved))
        results["B7_iter_dense"][grp].append(b7_ok)
        results["B7_iter_dense"]["all"].append(b7_ok)
        success_per_q["B7_iter_dense"][qid] = b7_ok

        def extract_names_from_seeds(seeds):
            names = set()
            for s in seeds:
                for match in NAME_REGEX.findall(docs[s]):
                    if match.lower() not in q_text.lower():
                        names.add(match)
            return names

        # B8: Overlap Expansion Top-2 Seeds
        s1, s2 = dense_ranked_docs[0], dense_ranked_docs[1]
        b8_seeds = [s1, s2]
        b8_names = extract_names_from_seeds(b8_seeds)
        b8_candidates = []
        for d in doc_ids:
            if d not in b8_seeds:
                d_text = docs[d]
                contained = [name for name in b8_names if name in d_text]
                if len(contained) > 0:
                    cos_sim = dense_sims[doc_ids.index(d)]
                    b8_candidates.append((len(contained), cos_sim, d))
        b8_candidates.sort(key=lambda x: (-x[0], -x[1]))
        b8_selected_candidates = [x[2] for x in b8_candidates[:2]]
        b8_retrieved = list(b8_seeds) + list(b8_selected_candidates)
        if len(b8_retrieved) < 4:
            for d in dense_ranked_docs:
                if d not in b8_retrieved:
                    b8_retrieved.append(d)
                    if len(b8_retrieved) == 4:
                        break
        b8_ok = gdocs.issubset(set(b8_retrieved))
        results["B8_overlap_top2"][grp].append(b8_ok)
        results["B8_overlap_top2"]["all"].append(b8_ok)
        success_per_q["B8_overlap_top2"][qid] = b8_ok

        # B9: Overlap Expansion Top-1 Seed
        b9_seeds = [s1]
        b9_names = extract_names_from_seeds(b9_seeds)
        b9_candidates = []
        for d in doc_ids:
            if d not in b9_seeds:
                d_text = docs[d]
                contained = [name for name in b9_names if name in d_text]
                if len(contained) > 0:
                    cos_sim = dense_sims[doc_ids.index(d)]
                    b9_candidates.append((len(contained), cos_sim, d))
        b9_candidates.sort(key=lambda x: (-x[0], -x[1]))
        b9_selected_candidates = [x[2] for x in b9_candidates[:3]]
        b9_retrieved = list(b9_seeds) + list(b9_selected_candidates)
        if len(b9_retrieved) < 4:
            for d in dense_ranked_docs:
                if d not in b9_retrieved:
                    b9_retrieved.append(d)
                    if len(b9_retrieved) == 4:
                        break
        b9_ok = gdocs.issubset(set(b9_retrieved))
        results["B9_overlap_top1"][grp].append(b9_ok)
        results["B9_overlap_top1"]["all"].append(b9_ok)
        success_per_q["B9_overlap_top1"][qid] = b9_ok

        # B4: Name Expansion Top-4 (Reference from previous audit)
        b4_seeds = [s1, s2]
        b4_candidates = []
        for d in doc_ids:
            if d not in b4_seeds:
                d_text = docs[d]
                contained = [name for name in b8_names if name in d_text]
                if len(contained) > 0:
                    cos_sim = dense_sims[doc_ids.index(d)]
                    b4_candidates.append((cos_sim, d))
        b4_candidates.sort(key=lambda x: -x[0])
        b4_selected = [x[1] for x in b4_candidates[:2]]
        b4_retrieved = list(b4_seeds) + list(b4_selected)
        if len(b4_retrieved) < 4:
            for d in dense_ranked_docs:
                if d not in b4_retrieved:
                    b4_retrieved.append(d)
                    if len(b4_retrieved) == 4:
                        break
        b4_ok = gdocs.issubset(set(b4_retrieved))
        results["B4_name_exp_top4"][grp].append(b4_ok)
        results["B4_name_exp_top4"]["all"].append(b4_ok)
        success_per_q["B4_name_exp_top4"][qid] = b4_ok

    report = []
    report.append("================================================================================")
    report.append("PART F: STRONGER GRAPH-FREE ITERATIVE BASELINES AUDIT")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append("Evaluated on all 105 answerable questions (primary rule evaluation: 30 hop2 questions)")
    report.append("================================================================================\n")

    # Recomputed H
    h_hop2_k = sum(hyb_success["hop2"])
    h_hop2_n = len(hyb_success["hop2"])
    H_rate = h_hop2_k / h_hop2_n
    h_low, h_high = wilson_score_interval(h_hop2_k, h_hop2_n)

    report.append("1. HYBRID RECOMPUTED RECALL ON HOP2:")
    report.append(f"- Hybrid all-gold evidence recall (H): {h_hop2_k}/{h_hop2_n} = {H_rate*100:.2f}% [Wilson 95%: {h_low*100:.1f}%, {h_high*100:.1f}%]")
    report.append(f"- Expected value from overnight REPORT.md: 56.7% (Confirmed exact match: {H_rate*100:.1f}%)\n")

    # Hop2 Performance Table
    report.append("2. PRIMARY EVALUATION ON HOP2 (N=30, BUDGET K=4):")
    report.append("| Method | Success Count | Recall Rate | Wilson 95% Confidence Interval |")
    report.append("| :--- | :---: | :---: | :---: |")
    report.append(f"| **Hybrid Mode (Recomputed H)** | {h_hop2_k}/30 | {H_rate*100:.1f}% | [{h_low*100:.1f}%, {h_high*100:.1f}%] |")
    
    for m in methods:
        k = sum(results[m]["hop2"])
        n = len(results[m]["hop2"])
        r = k / n
        low, high = wilson_score_interval(k, n)
        report.append(f"| `{m}` | {k}/{n} | {r*100:.1f}% | [{low*100:.1f}%, {high*100:.1f}%] |")
    report.append("")

    # Pre-registered Decision Rule Evaluation
    b7_rate = sum(results["B7_iter_dense"]["hop2"]) / 30.0
    b8_rate = sum(results["B8_overlap_top2"]["hop2"]) / 30.0
    b9_rate = sum(results["B9_overlap_top1"]["hop2"]) / 30.0
    b_star = max(b7_rate, b8_rate, b9_rate)

    best_method = "B9_overlap_top1" if b_star == b9_rate else ("B7_iter_dense" if b_star == b7_rate else "B8_overlap_top2")

    report.append("3. PRE-REGISTERED DECISION RULE OUTCOME (hop2 only):")
    report.append(f"- Recomputed Hybrid baseline (H): {H_rate*100:.2f}%")
    report.append(f"- Best Iterative Baseline (B*): {b_star*100:.2f}% (achieved by `{best_method}`)")
    report.append(f"- Upper threshold (H - 10.0 points): {(H_rate - 0.10)*100:.2f}%")
    report.append(f"- Lower threshold (H - 20.0 points): {(H_rate - 0.20)*100:.2f}%")
    
    if b_star >= H_rate - 0.10:
        rule_outcome = "iterative retrieval without a graph reproduces most of the hybrid gain; graph-specific benefit not demonstrated"
    elif b_star < H_rate - 0.20:
        rule_outcome = "graph-specific benefit supported against iterative baselines"
    else:
        rule_outcome = "ambiguous"

    report.append(f"\nDECISION RULE STATEMENT:")
    report.append(f"**\"{rule_outcome}\"**\n")

    # Paired comparisons against hybrid
    report.append("4. DESCRIPTIVE PAIRED COMPARISONS AGAINST HYBRID ON HOP2 (N=30):")
    report.append("| Baseline Method | Both Succeed (n11) | Hybrid-Only (n10) | Baseline-Only (n01) | Neither Succeed (n00) | Exact Sign Test p-value |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in methods:
        n11 = sum(1 for qid in hop2_qids if hyb_success_by_qid[qid] and success_per_q[m][qid])
        n10 = sum(1 for qid in hop2_qids if hyb_success_by_qid[qid] and not success_per_q[m][qid])
        n01 = sum(1 for qid in hop2_qids if not hyb_success_by_qid[qid] and success_per_q[m][qid])
        n00 = sum(1 for qid in hop2_qids if not hyb_success_by_qid[qid] and not success_per_q[m][qid])
        p_val = exact_binomial_sign_test(n10, n01)
        report.append(f"| `{m}` | {n11} | {n10} | {n01} | {n00} | p = {p_val:.4f} |")
    report.append("")

    # Full breakdown across all groups
    report.append("5. FULL BREAKDOWN ACROSS ALL QUESTION GROUPS (REPORTED FOR COMPLETENESS):")
    report.append("| Question Group | Total (N) | Hybrid | B7 (Iter Dense) | B8 (Overlap Top-2) | B9 (Overlap Top-1) | B4 (Name Exp Top-4) |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for grp in ["hop1", "hop2", "hop3", "alias", "all"]:
        n = len(results["B7_iter_dense"][grp])
        h_k = sum(hyb_success[grp])
        b7_k = sum(results["B7_iter_dense"][grp])
        b8_k = sum(results["B8_overlap_top2"][grp])
        b9_k = sum(results["B9_overlap_top1"][grp])
        b4_k = sum(results["B4_name_exp_top4"][grp])
        report.append(f"| `{grp}` | {n} | {h_k}/{n} ({h_k/n*100:.1f}%) | {b7_k}/{n} ({b7_k/n*100:.1f}%) | {b8_k}/{n} ({b8_k/n*100:.1f}%) | {b9_k}/{n} ({b9_k/n*100:.1f}%) | {b4_k}/{n} ({b4_k/n*100:.1f}%) |")
    report.append("================================================================================\n")

    report_text = "\n".join(report) + "\n"
    print(report_text)

    # Write to output.txt
    output_path = Path("experiments/synth_kg_audit/followup/output.txt")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(report_text)

    # Append to PROGRESS.md
    progress_path = Path("experiments/synth_kg_audit/followup/PROGRESS.md")
    timestamp = datetime.datetime.now().isoformat()
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | Part F: Evaluated iterative baselines B7/B8/B9/B4 and applied decision rule | `experiments/synth_kg_audit/followup/part_f_iterative_baselines.py` |\n")

if __name__ == "__main__":
    main()
