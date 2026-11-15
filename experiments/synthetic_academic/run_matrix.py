#!/usr/bin/env python3
"""
run_matrix.py: Executes the 8 Part 4 query evaluation runs across KB_02 and KB_03,
logs progress to PROGRESS.md, and runs analyze_runs.py on each run.
"""

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(".").resolve()))

from experiments.synthetic_academic.run_workflow import run_query
from experiments.synthetic_academic.analyze_runs import evaluate_run, load_questions, load_results, format_evaluation_output

RUNS = [
    {
        "id": "a",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_02"),
        "mode": "hybrid",
        "run_name": "kb02_run_a_hybrid_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(a): KB_02, hybrid, comparative_reasoning_mode=off",
    },
    {
        "id": "b",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_02"),
        "mode": "vector_only",
        "run_name": "kb02_run_b_vector_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(b): KB_02, vector_only, comparative_reasoning_mode=off",
    },
    {
        "id": "c",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_02"),
        "mode": "graph_only",
        "run_name": "kb02_run_c_graph_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(c): KB_02, graph_only, comparative_reasoning_mode=off",
    },
    {
        "id": "d",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_03"),
        "mode": "hybrid",
        "run_name": "kb03_run_d_hybrid_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(d): KB_03, hybrid, comparative_reasoning_mode=off",
    },
    {
        "id": "e",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_03"),
        "mode": "graph_only",
        "run_name": "kb03_run_e_graph_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(e): KB_03, graph_only, comparative_reasoning_mode=off",
    },
    {
        "id": "f",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_03"),
        "mode": "community_aware",
        "run_name": "kb03_run_f_comm_compoff",
        "comparative_reasoning_mode": "off",
        "description": "Part 4(f): KB_03, community_aware, comparative_reasoning_mode=off",
    },
    {
        "id": "g",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_02"),
        "mode": "hybrid",
        "run_name": "kb02_run_g_hybrid_compenforce",
        "comparative_reasoning_mode": "enforce",
        "description": "Part 4(g): KB_02, hybrid, comparative_reasoning_mode=enforce",
    },
    {
        "id": "h",
        "kb_dir": Path("knowledge_bases/synthetic_academic/synthetic_academic_02"),
        "mode": "vector_only",
        "run_name": "kb02_run_h_vector_compenforce",
        "comparative_reasoning_mode": "enforce",
        "description": "Part 4(h): KB_02, vector_only, comparative_reasoning_mode=enforce",
    },
]

questions_file = Path("experiments/synthetic_academic/questions.jsonl")
progress_file = Path("experiments/synthetic_academic/PROGRESS.md")
questions = load_questions(str(questions_file))

print(f"Starting Part 4 Execution Matrix: {len(RUNS)} runs.")
matrix_summaries = []

for idx, r in enumerate(RUNS, start=1):
    print("\n" + "=" * 80)
    print(f"[{idx}/{len(RUNS)}] EXECUTING {r['description']}")
    print("=" * 80)
    
    # Check if run already completed
    existing_runs = sorted(list(Path("experiments/synthetic_academic/runs").glob(f"*_{r['run_name']}")))
    if existing_runs and (existing_runs[-1] / "results.jsonl").exists():
        cand_results = load_results(str(existing_runs[-1] / "results.jsonl"))
        if len(cand_results) == len(questions):
            print(f"Reusing existing completed run: {existing_runs[-1]}")
            run_dir = existing_runs[-1]
            results_path = run_dir / "results.jsonl"
        else:
            run_dir = run_query(
                kb_dir=r["kb_dir"],
                questions_file=questions_file,
                run_name=r["run_name"],
                mode=r["mode"],
                vector_top_k=2,
                final_context_chunks=3,
                task6=False,
                graph_context_mode="off",
                comparative_reasoning_mode=r["comparative_reasoning_mode"],
                prompt_version="v1.1",
            )
            results_path = run_dir / "results.jsonl"
    else:
        run_dir = run_query(
            kb_dir=r["kb_dir"],
            questions_file=questions_file,
            run_name=r["run_name"],
            mode=r["mode"],
            vector_top_k=2,
            final_context_chunks=3,
            task6=False,
            graph_context_mode="off",
            comparative_reasoning_mode=r["comparative_reasoning_mode"],
            prompt_version="v1.1",
        )
        results_path = run_dir / "results.jsonl"
    results = load_results(str(results_path))
    eval_summary = evaluate_run(results, questions)
    eval_summary["run_name"] = r["run_name"]
    eval_summary["description"] = r["description"]
    eval_summary["run_dir"] = str(run_dir)
    matrix_summaries.append(eval_summary)
    
    report_text = format_evaluation_output(str(results_path), eval_summary)
    analysis_path = run_dir / "analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    mean_rec = eval_summary.get("mean_recall", 0.0)
    acc_pct = (eval_summary["correct_count"] / eval_summary["total_questions"] * 100) if eval_summary.get("total_questions") else 0.0
    eval_summary["accuracy_pct"] = acc_pct
    eval_summary["mean_doc_recall"] = mean_rec

    print(f"\nCompleted {r['id']}: Correct={eval_summary['correct_count']}/16 "
          f"(Answerable={eval_summary['answerable_correct']}/{eval_summary['answerable_total']}, "
          f"Abstentions={eval_summary['abstention_correct']}/{eval_summary['abstention_total']}, "
          f"Errors={eval_summary['error_count']}), "
          f"Mean Doc Recall={mean_rec:.3f}")
          
    # Append to PROGRESS.md
    now_iso = datetime.now().isoformat()
    log_line = (
        f"| {now_iso} | {r['description']} | "
        f"`{results_path}` ({eval_summary['correct_count']}/16 correct, "
        f"{mean_rec:.3f} recall) |\n"
    )
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(log_line)

print("\n" + "=" * 80)
print("ALL 8 PART 4 MATRIX RUNS COMPLETE")
print("=" * 80)

summary_table = [
    f"{'Run':<6} | {'Description':<50} | {'Correct':<10} | {'Answerable':<12} | {'Abstention':<12} | {'Errors':<8} | {'Doc Recall':<10}",
    "-" * 120,
]

for s in matrix_summaries:
    row = (
        f"{s['run_name'][:6]:<6} | "
        f"{s['description'][:50]:<50} | "
        f"{s['correct_count']}/16 ({s['accuracy_pct']:.1f}%) | "
        f"{s['answerable_correct']}/{s['answerable_total']:<12} | "
        f"{s['abstention_correct']}/{s['abstention_total']:<12} | "
        f"{s['error_count']:<8} | "
        f"{s['mean_doc_recall']:.3f}"
    )
    summary_table.append(row)

summary_str = "\n".join(summary_table)
print(summary_str)

matrix_analysis_path = Path("experiments/synthetic_academic/runs") / f"matrix_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(matrix_analysis_path, "w", encoding="utf-8") as f:
    f.write(summary_str + "\n\n")
    for s in matrix_summaries:
        f.write(format_evaluation_output(s["run_dir"], s) + "\n\n" + "=" * 80 + "\n\n")

print(f"\nSaved consolidated matrix analysis to: {matrix_analysis_path}")
