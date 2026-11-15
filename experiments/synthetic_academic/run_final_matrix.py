#!/usr/bin/env python3
"""
run_final_matrix.py: Executes the 7 Part 4 query evaluation runs across KB_03 and KB_02
in fresh subprocesses, evaluates each run, logs to PROGRESS.md, and saves matrix analysis.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.synthetic_academic.analyze_runs import (
    evaluate_run,
    format_evaluation_output,
    load_questions,
    load_results,
)

ROOT = Path(".").resolve()
QUESTIONS_FILE = ROOT / "experiments" / "synthetic_academic" / "questions.jsonl"
PROGRESS_FILE = ROOT / "experiments" / "synthetic_academic" / "PROGRESS.md"
RUNS_DIR = ROOT / "experiments" / "synthetic_academic" / "runs"

TARGET_RUNS = [
    {
        "num": 1,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_03",
        "mode": "hybrid",
        "run_name": "kb03_final_run1_hybrid",
        "description": "Part 4 Run 1: KB_03, hybrid, compoff",
    },
    {
        "num": 2,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_03",
        "mode": "vector_only",
        "run_name": "kb03_final_run2_vector",
        "description": "Part 4 Run 2: KB_03, vector_only, compoff",
    },
    {
        "num": 3,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_03",
        "mode": "graph_only",
        "run_name": "kb03_final_run3_graph",
        "description": "Part 4 Run 3: KB_03, graph_only, compoff",
    },
    {
        "num": 4,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_03",
        "mode": "community_aware",
        "run_name": "kb03_final_run4_comm",
        "description": "Part 4 Run 4: KB_03, community_aware, compoff",
    },
    {
        "num": 5,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_02",
        "mode": "hybrid",
        "run_name": "kb02_final_run5_hybrid",
        "description": "Part 4 Run 5: KB_02, hybrid, compoff",
    },
    {
        "num": 6,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_02",
        "mode": "vector_only",
        "run_name": "kb02_final_run6_vector",
        "description": "Part 4 Run 6: KB_02, vector_only, compoff",
    },
    {
        "num": 7,
        "kb_dir": ROOT / "knowledge_bases" / "synthetic_academic" / "synthetic_academic_03",
        "mode": "hybrid",
        "run_name": "kb03_final_run7_hybrid_repeat",
        "description": "Part 4 Run 7: KB_03, hybrid (repeat determinism check), compoff",
    },
]


def run_single(r_info):
    num = r_info["num"]
    print("\n" + "=" * 90)
    print(f"[{num}/7] EXECUTING: {r_info['description']}")
    print("=" * 90)

    cmd = [
        sys.executable,
        "-m",
        "experiments.synthetic_academic.run_workflow",
        "query",
        "--kb-dir",
        str(r_info["kb_dir"]),
        "--questions-file",
        str(QUESTIONS_FILE),
        "--run-name",
        r_info["run_name"],
        "--mode",
        r_info["mode"],
        "--vector-top-k",
        "2",
        "--final-context-chunks",
        "3",
        "--graph-context-mode",
        "off",
        "--comparative-reasoning-mode",
        "off",
        "--prompt-version",
        "v1.1",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(cmd, env=env, check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed with code {proc.returncode}")

    # Find the newly created run directory
    matching = sorted(list(RUNS_DIR.glob(f"*_{r_info['run_name']}")))
    if not matching:
        raise RuntimeError(f"Could not find run directory for {r_info['run_name']}")
    run_dir = matching[-1]
    results_path = run_dir / "results.jsonl"

    questions = load_questions(str(QUESTIONS_FILE))
    results = load_results(str(results_path))
    eval_summary = evaluate_run(results, questions)
    eval_summary["run_name"] = r_info["run_name"]
    eval_summary["description"] = r_info["description"]
    eval_summary["run_dir"] = str(run_dir)

    mean_rec = eval_summary.get("mean_recall", 0.0)
    acc_pct = (eval_summary["correct_count"] / eval_summary["total_questions"] * 100) if eval_summary.get("total_questions") else 0.0
    eval_summary["accuracy_pct"] = acc_pct
    eval_summary["mean_doc_recall"] = mean_rec

    report_text = format_evaluation_output(str(results_path), eval_summary)
    analysis_path = run_dir / "analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    now_iso = datetime.now().isoformat()
    log_line = (
        f"| {now_iso} | {r_info['description']} | "
        f"`{results_path}` ({eval_summary['correct_count']}/16 correct, "
        f"{mean_rec:.3f} recall) |\n"
    )
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

    print(f"\n[DONE {num}/7] Correct={eval_summary['correct_count']}/16 "
          f"(Answerable={eval_summary['answerable_correct']}/{eval_summary['answerable_total']}, "
          f"Abstentions={eval_summary['abstention_correct']}/{eval_summary['abstention_total']}, "
          f"Errors={eval_summary['error_count']}), "
          f"Mean Doc Recall={mean_rec:.3f}")
    return eval_summary


def main():
    target_indices = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "-" in arg:
            start, end = map(int, arg.split("-"))
            target_indices = set(range(start, end + 1))
        else:
            target_indices = {int(arg)}

    summaries = []
    questions = load_questions(str(QUESTIONS_FILE))

    for r_info in TARGET_RUNS:
        num = r_info["num"]
        if target_indices is not None and num not in target_indices:
            continue

        # Check if already completed
        matching = sorted(list(RUNS_DIR.glob(f"*_{r_info['run_name']}")))
        if matching and (matching[-1] / "results.jsonl").exists():
            existing_results = load_results(str(matching[-1] / "results.jsonl"))
            if len(existing_results) == len(questions):
                print(f"\n[REUSING {num}/7] Run {r_info['run_name']} already completed in {matching[-1].name}")
                eval_summary = evaluate_run(existing_results, questions)
                eval_summary["run_name"] = r_info["run_name"]
                eval_summary["description"] = r_info["description"]
                eval_summary["run_dir"] = str(matching[-1])
                mean_rec = eval_summary.get("mean_recall", 0.0)
                acc_pct = (eval_summary["correct_count"] / eval_summary["total_questions"] * 100) if eval_summary.get("total_questions") else 0.0
                eval_summary["accuracy_pct"] = acc_pct
                eval_summary["mean_doc_recall"] = mean_rec
                summaries.append(eval_summary)
                continue

        eval_summary = run_single(r_info)
        summaries.append(eval_summary)

    if len(summaries) > 1:
        print("\n" + "=" * 90)
        print("ALL TARGET RUNS COMPLETE")
        print("=" * 90)
        summary_table = [
            f"{'Run':<6} | {'Description':<52} | {'Correct':<12} | {'Answerable':<12} | {'Abstention':<12} | {'Errors':<8} | {'Doc Recall':<10}",
            "-" * 125,
        ]
        for s in summaries:
            row = (
                f"{s['run_name'][:6]:<6} | "
                f"{s['description'][:52]:<52} | "
                f"{s['correct_count']}/16 ({s['accuracy_pct']:.1f}%) | "
                f"{s['answerable_correct']}/{s['answerable_total']:<12} | "
                f"{s['abstention_correct']}/{s['abstention_total']:<12} | "
                f"{s['error_count']:<8} | "
                f"{s['mean_doc_recall']:.3f}"
            )
            summary_table.append(row)
        summary_str = "\n".join(summary_table)
        print(summary_str)

        out_analysis = RUNS_DIR / f"final_7_runs_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(out_analysis, "w", encoding="utf-8") as f:
            f.write(summary_str + "\n\n")
            for s in summaries:
                f.write(format_evaluation_output(s["run_dir"], s) + "\n\n" + "=" * 80 + "\n\n")
        print(f"\nSaved consolidated summary to: {out_analysis}")


if __name__ == "__main__":
    main()
