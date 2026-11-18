#!/usr/bin/env python3
"""Audit Part 1(c): Question template mapping, relation-token analysis, and hop2 recomputation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def exact_sign_test(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    # two-sided p-value under null p=0.5
    cum_prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    # two-sided
    p_val = min(1.0, 2.0 * cum_prob)
    return p_val

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(__file__).resolve().parent
SYNTH_DIR = BASE_DIR / "experiments" / "synth_kg"
OUTPUT_FILE = AUDIT_DIR / "audit_output.txt"


def log(msg: str) -> None:
    print(msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


TEMPLATES = [
    # 3-hop natural templates (most specific first)
    ("T_3H_BELONGS_DEVELOPS_BUILT", r"^What technology is used in the product developed by the department that .+ belongs to\?$"),
    ("T_3H_BELONGS_DEVELOPS_SERVES", r"^Which client is served by the product developed by the department that .+ belongs to\?$"),
    ("T_3H_MANAGES_DEVELOPS_BUILT", r"^What technology is used in the product developed by the unit managed by .+\?$"),
    ("T_3H_MANAGES_DEVELOPS_SERVES", r"^Which client is served by the product developed by the unit managed by .+\?$"),
    ("T_3H_MANAGES_BELONGS_DEVELOPS", r"^What product is developed by the department of the employee managed by .+\?$"),
    ("T_3H_MANAGES_MANAGES_BELONGS", r"^Which department does the employee managed by the staff under .+ belong to\?$"),
    ("T_3H_MANAGES_MANAGES_DEVELOPS", r"^What product is developed by the unit managed under .+\?$"),
    ("T_3H_TRAVERSING", r"^Starting from .+, traversing [A-Z_]+, [A-Z_]+, and [A-Z_]+, what is the target entity\?$"),

    # 2-hop natural templates
    ("T_2H_BELONGS_DEVELOPS", r"^What product is developed by the department that .+ belongs to\?$"),
    ("T_2H_BELONGS_SERVES", r"^Which client is served by the department that .+ belongs to\?$"),
    ("T_2H_DEVELOPS_BUILT", r"^What technology is used in the product developed by .+\?$"),
    ("T_2H_DEVELOPS_SERVES", r"^Which client is served by the product developed by .+\?$"),
    ("T_2H_MANAGES_DEVELOPS", r"^What product is developed by the department managed by .+\?$"),
    ("T_2H_MANAGES_BELONGS", r"^Which department does the employee managed by .+ belong to\?$"),
    ("T_2H_PRODUCT_BUILT", r"^What technology is used in .+\?$"),
    ("T_2H_TRAVERSING", r"^Starting from .+, traversing [A-Z_]+ then [A-Z_]+, what is the target entity\?$"),

    # 1-hop templates
    ("T_1H_BELONGS_TO", r"^Which department does .+ belong to\?$"),
    ("T_1H_DEVELOPS", r"^What product is developed by .+\?$"),
    ("T_1H_BUILT_WITH", r"^What technology is .+ built with\?$"),
    ("T_1H_SERVES", r"^Which client does .+ serve\?$"),
    ("T_1H_MANAGES", r"^What entity or unit is managed by .+\?$"),

    # Unanswerable templates
    ("T_UNANS_DEVELOPS", r"^What product is directly developed by .+\?$"),
    ("T_UNANS_SERVES", r"^Which commercial client is directly served by .+\?$"),
    ("T_UNANS_BUILT", r"^What technology framework is .+ built with\?$"),
    ("T_UNANS_PROD_DEV", r"^What product does .+ develop or manage\?$"),
    ("T_UNANS_CLIENT_MGT", r"^What client organization is managed by .+\?$"),
]


def classify_question(q_text: str) -> str:
    for tid, pat in TEMPLATES:
        if re.match(pat, q_text.strip()):
            return tid
    return "T_UNKNOWN"


def run_part1_c() -> None:
    log("================================================================================")
    log("PART 1(c) - QUESTION TEMPLATES, RELATION-TYPE TOKENS, AND HOP2 RECOMPUTATION")
    log("================================================================================\n")

    questions_file = SYNTH_DIR / "corpus_final" / "questions.jsonl"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    log(f"Total questions in {questions_file}: {len(questions)}")

    template_counts: Dict[str, Dict[str, int]] = {}
    q_to_template = {}
    relation_token_questions: List[Dict[str, Any]] = []

    relation_tokens = ["MANAGES", "BELONGS_TO", "DEVELOPS", "BUILT_WITH", "SERVES"]

    for q in questions:
        qid = q["id"]
        grp = q["group"]
        qtxt = q["question"]
        tid = classify_question(qtxt)
        q_to_template[qid] = tid
        template_counts.setdefault(grp, {})
        template_counts[grp][tid] = template_counts[grp].get(tid, 0) + 1

        # Check for relation token in template/question
        has_rel_token = any(f"traversing {r}" in qtxt or f"then {r}" in qtxt or f", {r}" in qtxt for r in relation_tokens)
        if has_rel_token or "TRAVERSING" in tid:
            relation_token_questions.append(q)

    log("--- Question Template Counts by Group ---")
    for grp in ["hop1", "hop2", "hop3", "alias", "unanswerable"]:
        log(f"\nGroup: `{grp}` (Total: {sum(template_counts.get(grp, {}).values())})")
        log("| Template ID | Count | Sample Question |")
        log("| :--- | :--- | :--- |")
        for tid, cnt in sorted(template_counts.get(grp, {}).items(), key=lambda x: -x[1]):
            # find sample
            samp = next(q["question"] for q in questions if q["group"] == grp and q_to_template[q["id"]] == tid)
            log(f"| `{tid}` | {cnt} | {samp} |")

    log("\n--- Questions Containing Relation-Type Tokens (e.g., 'traversing MANAGES') ---")
    log(f"Total questions containing relation tokens: {len(relation_token_questions)}")
    rel_by_grp: Dict[str, List[str]] = {}
    for q in relation_token_questions:
        rel_by_grp.setdefault(q["group"], []).append(q["id"])

    for grp, qids in rel_by_grp.items():
        log(f"  Group `{grp}`: {len(qids)} questions ({', '.join(qids)})")
    log("")

    # Load run evaluation results for hop2 questions
    eval_file = SYNTH_DIR / "evaluation_results.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    per_q = eval_data.get("per_question_eval", {})

    hop2_questions = [q for q in questions if q["group"] == "hop2"]
    hop2_rel_token_ids = set(rel_by_grp.get("hop2", []))
    hop2_filtered = [q for q in hop2_questions if q["id"] not in hop2_rel_token_ids]

    log(f"Hop 2 Analysis:")
    log(f"- Total hop2 questions: {len(hop2_questions)}")
    log(f"- Hop 2 questions with relation tokens: {len(hop2_rel_token_ids)} ({', '.join(sorted(hop2_rel_token_ids))})")
    log(f"- Hop 2 questions EXCLUDING relation tokens: {len(hop2_filtered)}")
    log("")

    modes = ["vector_only", "hybrid", "graph_only", "community_aware"]
    log(f"--- Hop 2 Performance: All {len(hop2_questions)} vs. Excluding Relation-Token Questions ({len(hop2_filtered)}) ---")
    log(f"| Mode | All hop2 Accuracy (N={len(hop2_questions)}) | Excl. Rel-Token Accuracy (N={len(hop2_filtered)}) | Removed Questions Accuracy (N={len(hop2_rel_token_ids)}) |")
    log("| :--- | :--- | :--- | :--- |")
    
    filtered_results_by_mode: Dict[str, Dict[str, bool]] = {m: {} for m in modes}
    for m in modes:
        c_all = sum(1 for q in hop2_questions if per_q[q["id"]][m]["correct"])
        c_filt = sum(1 for q in hop2_filtered if per_q[q["id"]][m]["correct"])
        c_rem = sum(1 for q in hop2_questions if q["id"] in hop2_rel_token_ids and per_q[q["id"]][m]["correct"])
        for q in hop2_filtered:
            filtered_results_by_mode[m][q["id"]] = per_q[q["id"]][m]["correct"]
        log(f"| `{m}` | {c_all}/{len(hop2_questions)} ({c_all/len(hop2_questions)*100:.1f}%) | {c_filt}/{len(hop2_filtered)} ({c_filt/len(hop2_filtered)*100:.1f}%) | {c_rem}/{len(hop2_rel_token_ids)} ({c_rem/len(hop2_rel_token_ids)*100:.1f}%) |")
    log("")

    # Paired sign test: hybrid vs vector_only on the remaining hop2 questions
    n11 = 0  # both correct
    n10 = 0  # hybrid correct, vector incorrect
    n01 = 0  # vector correct, hybrid incorrect
    n00 = 0  # neither correct
    for q in hop2_filtered:
        qid = q["id"]
        h_corr = filtered_results_by_mode["hybrid"][qid]
        v_corr = filtered_results_by_mode["vector_only"][qid]
        if h_corr and v_corr:
            n11 += 1
        elif h_corr and not v_corr:
            n10 += 1
        elif not h_corr and v_corr:
            n01 += 1
        else:
            n00 += 1

    disc = n10 + n01
    p_val = exact_sign_test(n10, n01)

    log(f"--- Paired Comparison: hybrid vs. vector_only on Remaining {len(hop2_filtered)} hop2 Questions ---")
    log(f"Both correct (n11): {n11}")
    log(f"Hybrid ONLY correct (n10): {n10}")
    log(f"Vector ONLY correct (n01): {n01}")
    log(f"Neither correct (n00): {n00}")
    log(f"Discordant pairs (n10 + n01): {disc}")
    log(f"Exact Two-Sided Sign Test p-value: {p_val:.4f}")
    log(f"Statistically significant at alpha=0.05: {p_val < 0.05} ({'DETECTED' if p_val < 0.05 else 'No detectable difference'})")
    log("")
    c_hyb_filt = sum(1 for q in hop2_filtered if filtered_results_by_mode["hybrid"][q["id"]])
    c_vec_filt = sum(1 for q in hop2_filtered if filtered_results_by_mode["vector_only"][q["id"]])
    log(f"Conclusion: Excluding the {len(hop2_rel_token_ids)} relation-token questions (e.g. 'Starting from X, traversing MANAGES...'):")
    log(f"- Hybrid accuracy is {c_hyb_filt}/{len(hop2_filtered)} ({c_hyb_filt/len(hop2_filtered)*100:.1f}%) vs. Vector-only {c_vec_filt}/{len(hop2_filtered)} ({c_vec_filt/len(hop2_filtered)*100:.1f}%).")
    log(f"- With n10={n10}, n01={n01}, discordant={disc}, the exact two-sided sign test p-value is {p_val:.4f}.")
    if p_val < 0.05:
        log("- The hybrid advantage on hop2 SURVIVES and remains statistically significant (p < 0.05).")
    else:
        log("- The hybrid advantage on hop2 does NOT reach statistical significance at alpha=0.05.")
    log("\n")


if __name__ == "__main__":
    run_part1_c()
