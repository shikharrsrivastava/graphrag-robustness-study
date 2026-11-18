#!/usr/bin/env python3
"""Part 1: Read-only adjudication dump from Milestone 3 run files.

Reads results.jsonl from conditions A_v11, A_v10, B, C under experiments/hotpotqa_m3/runs/.
Outputs:
(a) Non-correct question blocks across conditions.
(b) Untrimmed answer texts in A_v11 and A_v10 for the 4 focal questions.
(c) Gold answer breakdown ('yes', 'no', other) and accuracy per condition.
(d) du -sh and git status checks.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from experiments.hotpotqa_m3.score import evaluate_record, load_results_file, normalize_answer


def main() -> None:
    runs_dir = Path("experiments/hotpotqa_m3/runs")
    questions_file = Path("experiments/hotpotqa_m3/selected_questions.json")

    with open(questions_file, "r", encoding="utf-8") as f:
        selected_questions = json.load(f)

    question_order = [q["_id"] for q in selected_questions]
    id_to_meta = {q["_id"]: q for q in selected_questions}

    # Discover runs
    conditions = ["A_v11", "A_v10", "B", "C"]
    condition_files: Dict[str, Path] = {}
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        for cond in conditions:
            if run_path.name.endswith(f"_{cond}") or run_path.name == cond:
                results_file = run_path / "results.jsonl"
                if results_file.exists():
                    condition_files[cond] = results_file

    print(f"Discovered condition run files: { {k: str(v) for k, v in condition_files.items()} }\n")

    # Load evaluated records per condition
    condition_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cond in conditions:
        raw_recs = load_results_file(condition_files[cond])
        eval_recs = {r["question_id"]: evaluate_record(r) for r in raw_recs}
        condition_records[cond] = eval_recs

    # -------------------------------------------------------------------------
    # PART 1(a): Questions not correct (FAIL, ABS, or ERR) in >=1 condition
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PART 1(a): QUESTIONS NOT CORRECT (FAIL, ABS, OR ERR) IN AT LEAST ONE CONDITION")
    print("=" * 80)

    non_correct_qids = []
    for qid in question_order:
        statuses = [condition_records[c][qid]["symbol"] for c in conditions]
        if any(s != "PASS" for s in statuses):
            non_correct_qids.append(qid)

    print(f"Total questions non-correct in >=1 condition: {len(non_correct_qids)} / {len(question_order)}\n")

    for idx, qid in enumerate(non_correct_qids, 1):
        meta = id_to_meta[qid]
        q_type = meta.get("type", "unknown")
        gold = meta.get("answer", "")
        q_text = meta.get("question", "")

        print(f"[{idx}/{len(non_correct_qids)}] Question ID: {qid}")
        print(f"Type: {q_type}")
        print(f"Question: {q_text}")
        print(f"Gold Answer: {gold}")
        print("Non-correct Conditions:")

        for cond in conditions:
            rec = condition_records[cond][qid]
            sym = rec["symbol"]
            if sym != "PASS":
                full_ans = rec["answer"]
                trimmed_ans = full_ans[:350] + ("..." if len(full_ans) > 350 else "")
                print(f"  Condition: {cond}")
                print(f"    Status:           {sym}")
                print(f"    Canonical Answer: {rec['canonical_answer']!r}")
                print(f"    Full Answer:      {trimmed_ans}")
                print(f"    Abstained:        {rec['abstained']}")
                print(f"    Generation Error: {rec['generation_error']}")
        print("-" * 80)

    # -------------------------------------------------------------------------
    # PART 1(b): Full untrimmed answer texts for the 4 focal questions in A_v11 & A_v10
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1(b): UNTRIMMED ANSWER TEXTS FOR 4 FOCAL QUESTIONS IN A_v11 AND A_v10")
    print("=" * 80)

    focal_qids = ["5a8a3dd5", "5a8b82a0", "5ae26905", "5ab2b630"]
    for f_prefix in focal_qids:
        # Match full qid
        full_qid = next((q for q in question_order if q.startswith(f_prefix)), None)
        if not full_qid:
            print(f"ERROR: Could not find question starting with {f_prefix}")
            continue

        meta = id_to_meta[full_qid]
        print(f"\nQuestion ID: {full_qid} (Prefix: {f_prefix})")
        print(f"Type: {meta.get('type')} | Gold: {meta.get('answer')}")
        print(f"Question: {meta.get('question')}")

        for cond in ("A_v11", "A_v10"):
            rec = condition_records[cond][full_qid]
            print(f"\n--- Condition {cond} ---")
            print(f"Status: {rec['symbol']} | Abstained: {rec['abstained']} | Error: {rec['generation_error']}")
            print(f"Canonical Answer: {rec['canonical_answer']!r}")
            print(f"Full Answer (untrimmed):\n{rec['answer']}")

    # -------------------------------------------------------------------------
    # PART 1(c): Gold answers yes/no/other counts & accuracy per condition
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1(c): GOLD ANSWER BREAKDOWN (YES / NO / OTHER) AND SUBSET ACCURACY")
    print("=" * 80)

    category_counts = {"yes": 0, "no": 0, "other": 0}
    category_qids: Dict[str, List[str]] = {"yes": [], "no": [], "other": []}

    for qid in question_order:
        norm_gold = normalize_answer(id_to_meta[qid].get("answer", ""))
        if norm_gold == "yes":
            category_counts["yes"] += 1
            category_qids["yes"].append(qid)
        elif norm_gold == "no":
            category_counts["no"] += 1
            category_qids["no"].append(qid)
        else:
            category_counts["other"] += 1
            category_qids["other"].append(qid)

    print(f"Total Questions: {len(question_order)}")
    print(f"  Gold = 'yes':   {category_counts['yes']} questions")
    print(f"  Gold = 'no':    {category_counts['no']} questions")
    print(f"  Gold = 'other': {category_counts['other']} questions\n")

    print(f"{'Condition':<10} | {'Gold=yes (' + str(category_counts['yes']) + ')':<18} | {'Gold=no (' + str(category_counts['no']) + ')':<18} | {'Other (' + str(category_counts['other']) + ')':<18} | {'Overall (30)':<18}")
    print("-" * 80)

    for cond in conditions:
        rec_map = condition_records[cond]
        yes_corr = sum(1 for q in category_qids["yes"] if rec_map[q]["contains"])
        no_corr = sum(1 for q in category_qids["no"] if rec_map[q]["contains"])
        other_corr = sum(1 for q in category_qids["other"] if rec_map[q]["contains"])
        overall_corr = yes_corr + no_corr + other_corr

        yes_acc = f"{yes_corr}/{category_counts['yes']} ({yes_corr / category_counts['yes'] * 100:.1f}%)" if category_counts['yes'] else "N/A"
        no_acc = f"{no_corr}/{category_counts['no']} ({no_corr / category_counts['no'] * 100:.1f}%)" if category_counts['no'] else "N/A"
        other_acc = f"{other_corr}/{category_counts['other']} ({other_corr / category_counts['other'] * 100:.1f}%)" if category_counts['other'] else "N/A"
        overall_acc = f"{overall_corr}/30 ({overall_corr / 30 * 100:.1f}%)"

        print(f"{cond:<10} | {yes_acc:<18} | {no_acc:<18} | {other_acc:<18} | {overall_acc:<18}")

    # -------------------------------------------------------------------------
    # PART 1(d): du -sh data directory and git status -s
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1(d): DISK USAGE AND GIT STATUS")
    print("=" * 80)

    du_res = subprocess.run(["du", "-sh", "experiments/hotpotqa_m3/data"], capture_output=True, text=True)
    print("du -sh experiments/hotpotqa_m3/data:")
    print(du_res.stdout.strip())

    git_res = subprocess.run(["git", "status", "-s"], capture_output=True, text=True)
    print("\ngit status -s:")
    print(git_res.stdout.strip())


if __name__ == "__main__":
    main()
