#!/usr/bin/env python3
"""Milestone 3 Evaluation and Scoring Engine for HotpotQA Generator Isolation.

Implements:
1. Standard HotpotQA normalization, Exact Match (EM), and Token F1.
2. Token-boundary 'contains' check (with affirmative/negative first-token check for Boolean questions).
3. Diagnostic counters: ABSTAINED, ERROR, Abstained-but-Answer-Present.
4. Condition C Gold Paragraph Recall @ Top 3 breakdown (both, one, none).
5. Wilson 95% Score Confidence Intervals.
6. Summary tables and compact 30-question symbol matrix (PASS, FAIL, ABS, ERR).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def normalize_answer(text: Optional[str]) -> str:
    """Standard HotpotQA answer normalization.
    
    1. Lowercase text.
    2. Remove punctuation.
    3. Remove English articles ('a', 'an', 'the') as whole words.
    4. Collapse whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove all characters that are not alphanumeric or whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split()).strip()


def exact_match(prediction: Optional[str], gold: str) -> bool:
    """Exact Match after normalization."""
    if not prediction:
        return False
    norm_p = normalize_answer(prediction)
    norm_g = normalize_answer(gold)
    if not norm_p or not norm_g:
        return False
    return norm_p == norm_g


def token_f1(prediction: Optional[str], gold: str) -> float:
    """Token-level F1 score after normalization."""
    if not prediction:
        return 0.0
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(p_tokens)
    recall = num_common / len(g_tokens)
    return 2.0 * precision * recall / (precision + recall)


def extract_first_boolean_token(text: str) -> Optional[str]:
    """Find the first standalone 'yes' or 'no' token in text."""
    norm = normalize_answer(text)
    for token in norm.split():
        if token in ("yes", "no"):
            return token
    return None


def is_subsequence(sub: Sequence[str], full: Sequence[str]) -> bool:
    """Check if sub is a contiguous subsequence of full."""
    n_sub = len(sub)
    n_full = len(full)
    if n_sub == 0:
        return True
    if n_sub > n_full:
        return False
    for i in range(n_full - n_sub + 1):
        if full[i : i + n_sub] == sub:
            return True
    return False


def check_contains(
    canonical_answer: Optional[str],
    full_answer: Optional[str],
    gold_answer: str,
) -> bool:
    """Check whether prediction contains the gold answer under HotpotQA rules.
    
    - For Boolean questions ('yes' or 'no'):
      Check if the first boolean token ('yes' or 'no') matches gold.
      Prioritize canonical_answer, then full_answer.
    - For general questions:
      Check if normalized gold tokens appear as a contiguous subsequence in
      either canonical_answer or full_answer.
    """
    norm_gold = normalize_answer(gold_answer)
    gold_tokens = norm_gold.split()
    if not gold_tokens:
        return False

    is_boolean = norm_gold in ("yes", "no")

    if is_boolean:
        # Check canonical first
        if canonical_answer:
            b_cand = extract_first_boolean_token(canonical_answer)
            if b_cand:
                return b_cand == norm_gold
        if full_answer:
            b_cand = extract_first_boolean_token(full_answer)
            if b_cand:
                return b_cand == norm_gold
        return False

    # General question: check contiguous token subsequence or substring
    for text in (canonical_answer, full_answer):
        if not text:
            continue
        norm_text = normalize_answer(text)
        text_tokens = norm_text.split()
        if is_subsequence(gold_tokens, text_tokens):
            return True
        # Also check substring match on normalized string
        if norm_gold in norm_text:
            return True

    return False


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson 95% score confidence interval for a proportion."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    centre = p_hat + (z**2) / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z**2) / (4.0 * (total**2))))
    lower = max(0.0, (centre - spread) / denom)
    upper = min(1.0, (centre + spread) / denom)
    return lower, upper


def evaluate_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question result record."""
    gold = rec.get("gold_answer") or rec.get("answer_gold") or ""
    canonical = rec.get("canonical_answer") or ""
    full_ans = rec.get("answer") or ""
    abstained = bool(rec.get("abstained", False))
    gen_error = bool(rec.get("generation_error", False))

    # Determine whether answer actually contains gold regardless of abstention
    answer_has_gold = check_contains(canonical, full_ans, gold)

    abstained_but_answer_present = abstained and not gen_error and answer_has_gold

    if gen_error:
        contains_pass = False
        em_pass = False
        f1_val = 0.0
        symbol = "ERR"
    elif abstained:
        contains_pass = False
        em_pass = False
        f1_val = 0.0
        symbol = "ABS"
    else:
        contains_pass = answer_has_gold
        # For EM and F1, evaluate on canonical candidate if available, else full answer
        eval_cand = canonical.strip() if canonical.strip() else full_ans.strip()
        # Clean citation brackets e.g. [D001_P0] from candidate
        eval_cand_clean = re.sub(r"\[[^\]]+\]", "", eval_cand).strip()
        em_pass = exact_match(eval_cand_clean, gold)
        f1_val = token_f1(eval_cand_clean, gold)
        symbol = "PASS" if contains_pass else "FAIL"

    return {
        "question_id": rec.get("question_id") or rec.get("_id"),
        "type": rec.get("type", "unknown"),
        "level": rec.get("level", "unknown"),
        "question": rec.get("question", ""),
        "gold_answer": gold,
        "canonical_answer": canonical,
        "answer": full_ans,
        "abstained": abstained,
        "generation_error": gen_error,
        "abstained_but_answer_present": abstained_but_answer_present,
        "contains": contains_pass,
        "em": em_pass,
        "f1": f1_val,
        "symbol": symbol,
        "gold_recall_bucket": rec.get("gold_recall_bucket"),
        "gold_recall_count": rec.get("gold_recall_count"),
        "retrieved_chunks": rec.get("retrieved_chunks", []),
    }


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation records into summary metrics."""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "contains_count": 0,
            "contains_acc": 0.0,
            "contains_ci": (0.0, 0.0),
            "em_count": 0,
            "em_acc": 0.0,
            "mean_f1": 0.0,
            "abstained_count": 0,
            "error_count": 0,
            "abstained_but_answer_present_count": 0,
        }

    contains_cnt = sum(1 for r in records if r["contains"])
    em_cnt = sum(1 for r in records if r["em"])
    f1_sum = sum(r["f1"] for r in records)
    abs_cnt = sum(1 for r in records if r["abstained"])
    err_cnt = sum(1 for r in records if r["generation_error"])
    abs_has_cnt = sum(1 for r in records if r["abstained_but_answer_present"])

    ci_lower, ci_upper = wilson_score_interval(contains_cnt, total)

    return {
        "total": total,
        "contains_count": contains_cnt,
        "contains_acc": contains_cnt / total,
        "contains_ci": (ci_lower, ci_upper),
        "em_count": em_cnt,
        "em_acc": em_cnt / total,
        "mean_f1": f1_sum / total,
        "abstained_count": abs_cnt,
        "error_count": err_cnt,
        "abstained_but_answer_present_count": abs_has_cnt,
    }


def load_results_file(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL results file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_summary_table(condition_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format Markdown summary table across all conditions and question subsets."""
    lines = [
        "| Condition | Subset | N | Contains Acc (Count) | 95% Wilson CI | Exact Match (EM) | Mean F1 | ABSTAINED | ERROR | Abs-but-Present |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cond_name, subsets in condition_metrics.items():
        for subset_name in ("Overall", "Bridge", "Comparison"):
            m = subsets.get(subset_name)
            if not m or m["total"] == 0:
                continue
            ci_str = f"[{m['contains_ci'][0]:.3f}, {m['contains_ci'][1]:.3f}]"
            acc_str = f"{m['contains_acc'] * 100:.1f}% ({m['contains_count']}/{m['total']})"
            em_str = f"{m['em_acc'] * 100:.1f}% ({m['em_count']}/{m['total']})"
            f1_str = f"{m['mean_f1']:.3f}"
            lines.append(
                f"| `{cond_name}` | {subset_name} | {m['total']} | {acc_str} | {ci_str} | {em_str} | {f1_str} | {m['abstained_count']} | {m['error_count']} | {m['abstained_but_answer_present_count']} |"
            )
    return "\n".join(lines)


def format_symbol_matrix(
    question_ids: List[str],
    id_to_meta: Dict[str, Dict[str, Any]],
    condition_evals: Dict[str, Dict[str, Dict[str, Any]]],
    conditions: List[str],
) -> str:
    """Format compact 30-question symbol matrix (PASS / FAIL / ABS / ERR)."""
    header = f"| # | Question ID | Type | " + " | ".join(f"`{c}`" for c in conditions) + " | Gold Answer |"
    sep = "| :---: | :--- | :---: | " + " | ".join([":---:"] * len(conditions)) + " | :--- |"
    lines = [header, sep]

    for idx, qid in enumerate(question_ids, start=1):
        meta = id_to_meta.get(qid, {})
        q_type = meta.get("type", "unknown")
        gold = meta.get("answer", "")
        # Shorten gold if too long for compact display
        gold_short = (gold[:25] + "...") if len(gold) > 28 else gold

        row = [f"{idx}", f"`{qid[:8]}`", q_type]
        for c in conditions:
            eval_rec = condition_evals.get(c, {}).get(qid)
            if eval_rec:
                sym = eval_rec["symbol"]
            else:
                sym = "-"
            row.append(sym)
        row.append(f"`{gold_short}`")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_condition_c_recall_table(c_evals: List[Dict[str, Any]]) -> str:
    """Format Condition C Gold Paragraph Recall @ Top 3 analysis."""
    buckets = {"both": [], "one": [], "none": []}
    for r in c_evals:
        b = r.get("gold_recall_bucket", "none")
        if b in buckets:
            buckets[b].append(r)
        else:
            buckets["none"].append(r)

    total_c = len(c_evals)
    lines = [
        "| Gold Recall @ Top 3 | Paragraphs Retrieved | Count (Proportion) | Contains Correct | Contains Acc within Bucket |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for b_key, b_label, b_paras in [
        ("both", "Both Gold Present", "2/2"),
        ("one", "One Gold Present", "1/2"),
        ("none", "Neither Gold Present", "0/2"),
    ]:
        recs = buckets[b_key]
        cnt = len(recs)
        prop = (cnt / total_c * 100) if total_c > 0 else 0.0
        corr = sum(1 for r in recs if r["contains"])
        acc = (corr / cnt * 100) if cnt > 0 else 0.0
        lines.append(
            f"| {b_label} | `{b_paras}` | {cnt}/{total_c} ({prop:.1f}%) | {corr}/{cnt} | {acc:.1f}% |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3 HotpotQA Scoring Engine")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/hotpotqa_m3/runs"),
        help="Path to directory containing run folders",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("experiments/hotpotqa_m3/selected_questions.json"),
        help="Path to selected_questions.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to write generated markdown scoring report",
    )
    args = parser.parse_args()

    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions_list = json.load(f)

    question_ids = [q["_id"] for q in questions_list]
    id_to_meta = {q["_id"]: q for q in questions_list}

    # Discover latest run directory per condition
    conditions = ["A_v11", "A_v10", "B", "C"]
    condition_latest_runs: Dict[str, Path] = {}

    if args.runs_dir.exists():
        for run_path in sorted(args.runs_dir.iterdir()):
            if not run_path.is_dir():
                continue
            for cond in conditions:
                if run_path.name.endswith(f"_{cond}") or run_path.name == cond:
                    results_file = run_path / "results.jsonl"
                    if results_file.exists():
                        condition_latest_runs[cond] = results_file

    print(f"Discovered runs: { {k: str(v) for k, v in condition_latest_runs.items()} }")

    condition_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    condition_evals: Dict[str, Dict[str, Dict[str, Any]]] = {}
    c_records: List[Dict[str, Any]] = []

    for cond in conditions:
        results_file = condition_latest_runs.get(cond)
        if not results_file:
            continue
        raw_records = load_results_file(results_file)
        eval_records = [evaluate_record(r) for r in raw_records]
        condition_evals[cond] = {r["question_id"]: r for r in eval_records}

        if cond == "C":
            c_records = eval_records

        overall = aggregate_metrics(eval_records)
        bridge = aggregate_metrics([r for r in eval_records if r["type"] == "bridge"])
        comparison = aggregate_metrics([r for r in eval_records if r["type"] == "comparison"])

        condition_metrics[cond] = {
            "Overall": overall,
            "Bridge": bridge,
            "Comparison": comparison,
        }

    report_sections = []
    report_sections.append("# Milestone 3 Generator Evaluation Results\n")

    if condition_metrics:
        report_sections.append("## 1. Summary Performance Across Conditions\n")
        report_sections.append(format_summary_table(condition_metrics))
        report_sections.append("\n")

        report_sections.append("## 2. Compact 30-Question Symbol Matrix\n")
        report_sections.append(format_symbol_matrix(question_ids, id_to_meta, condition_evals, list(condition_metrics.keys())))
        report_sections.append("\n")

        if "C" in condition_metrics:
            report_sections.append("## 3. Condition C Dense Retrieval Recall Breakdown\n")
            report_sections.append(format_condition_c_recall_table(c_records))
            report_sections.append("\n")

    report_text = "\n".join(report_sections)
    print(report_text)

    if args.output_report:
        args.output_report.write_text(report_text, encoding="utf-8")
        print(f"Wrote report to {args.output_report}")


if __name__ == "__main__":
    main()
