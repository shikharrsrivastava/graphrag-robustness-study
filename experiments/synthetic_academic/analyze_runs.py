#!/usr/bin/env python3
"""
analyze_runs.py: Evaluates results.jsonl run files against questions.jsonl.

Computes:
1. Per-question table: ID, Category, Answer, Abstained, Correctness, Required-Doc Recall.
2. Per-run summary: Correct/16, Answerable Correct/13, Abstention Correct/3,
   Mean Required-Doc Recall over the 13 answerable questions, and Citation-ID Validity.
Outputs to stdout and writes to a new timestamped text file (never overwriting).
"""

import argparse
from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Set

VALID_SYNTHETIC_CHUNKS: Set[str] = {
    "D001_C0000",
    "D002_C0000",
    "D003_C0000",
    "D004_C0000",
    "D005_C0000",
}

DOC_ID_TO_FILENAME = {
    "D001": "D1_students.txt",
    "D002": "D2_varghese_lab.txt",
    "D003": "D3_smokewatch.txt",
    "D004": "D4_dhillon_lab.txt",
    "D005": "D5_members.txt",
}


def load_questions(questions_path: str) -> Dict[str, Dict[str, Any]]:
    questions = {}
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                questions[q["id"]] = q
    return questions


def load_results(results_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate_run(
    results: List[Dict[str, Any]], questions: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    per_question = []
    correct_count = 0
    answerable_correct = 0
    answerable_total = 0
    abstention_correct = 0
    abstention_total = 0
    error_count = 0
    doc_recalls = []

    total_citations = 0
    valid_citations = 0
    invalid_citations = []

    for r in results:
        qid = r.get("question_id") or r.get("id")
        q_meta = questions.get(qid, {})

        category = r.get("category") or q_meta.get("category", "unknown")
        answerable = r.get("answerable")
        if answerable is None:
            answerable = q_meta.get("answerable", True)

        expected_contains = r.get("expected_answer_contains")
        if expected_contains is None:
            expected_contains = q_meta.get("expected_answer_contains", [])

        required_docs = r.get("required_docs")
        if required_docs is None:
            required_docs = q_meta.get("required_docs", [])

        answer = r.get("answer") or ""
        abstained = bool(r.get("abstained", False))
        is_error = bool(r.get("generation_error", False)) or answer.startswith("Error generating answer")

        # Determine correctness: Errors are NEVER counted as abstention or correct
        is_correct = False
        if is_error:
            error_count += 1
            if not answerable:
                abstention_total += 1
            else:
                answerable_total += 1
        elif not answerable:
            abstention_total += 1
            if abstained:
                is_correct = True
                abstention_correct += 1
        else:
            answerable_total += 1
            if not abstained and answer:
                ans_lower = answer.lower()
                if all(sub.lower() in ans_lower for sub in expected_contains):
                    is_correct = True
                    answerable_correct += 1

        if is_correct:
            correct_count += 1

        # Required-doc recall
        retrieved_files: Set[str] = set()
        for ev in r.get("evidence_list", []):
            fn = ev.get("filename")
            if fn:
                retrieved_files.add(fn)
            doc_id = ev.get("doc_id")
            if doc_id and doc_id in DOC_ID_TO_FILENAME:
                retrieved_files.add(DOC_ID_TO_FILENAME[doc_id])
            chunk_id = ev.get("chunk_id", "")
            prefix = chunk_id.split("_")[0]
            if prefix in DOC_ID_TO_FILENAME:
                retrieved_files.add(DOC_ID_TO_FILENAME[prefix])

        req_set = set(required_docs)
        if answerable and req_set:
            recall = len(req_set.intersection(retrieved_files)) / len(req_set)
            doc_recalls.append(recall)
        else:
            recall = 1.0 if not req_set else 0.0

        # Citations validity
        cited_chunks = r.get("cited_chunk_ids", [])
        for c in cited_chunks:
            total_citations += 1
            if c in VALID_SYNTHETIC_CHUNKS:
                valid_citations += 1
            else:
                invalid_citations.append((qid, c))

        per_question.append({
            "id": qid,
            "category": category,
            "answer": answer,
            "abstained": abstained,
            "answerable": answerable,
            "is_correct": is_correct,
            "is_error": is_error,
            "recall": recall,
            "cited_chunks": cited_chunks,
        })

    mean_recall = (
        (sum(doc_recalls) / len(doc_recalls)) if doc_recalls else 0.0
    )

    return {
        "per_question": per_question,
        "total_questions": len(results),
        "correct_count": correct_count,
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "abstention_correct": abstention_correct,
        "abstention_total": abstention_total,
        "error_count": error_count,
        "doc_recall_count": len(doc_recalls),
        "mean_recall": mean_recall,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
    }


def format_evaluation_output(
    results_path: str, eval_data: Dict[str, Any]
) -> str:
    lines = []
    lines.append(f"==================================================")
    lines.append(f"RUN EVALUATION: {results_path}")
    lines.append(f"==================================================")

    # Per-question table
    lines.append("\n### Per-Question Table")
    header = f"| {'ID':4} | {'Category':20} | {'Abst':4} | {'Corr':4} | {'DocRec':6} | {'Answer (truncated)':50} |"
    sep = f"| {'-'*4} | {'-'*20} | {'-'*4} | {'-'*4} | {'-'*6} | {'-'*50} |"
    lines.append(header)
    lines.append(sep)

    for pq in eval_data["per_question"]:
        ans_trunc = pq["answer"].replace("\n", " ")[:48]
        if len(pq["answer"]) > 48:
            ans_trunc += ".."
        abst_str = "YES" if pq["abstained"] else "NO"
        corr_str = "ERROR" if pq.get("is_error") else ("PASS" if pq["is_correct"] else "FAIL")
        rec_str = (
            f"{pq['recall']:.2f}"
            if pq["answerable"]
            else "N/A"
        )
        row = f"| {pq['id']:4} | {pq['category']:20} | {abst_str:4} | {corr_str:5} | {rec_str:6} | {ans_trunc:50} |"
        lines.append(row)

    # Summary section
    total = eval_data["total_questions"]
    corr = eval_data["correct_count"]
    ans_corr = eval_data["answerable_correct"]
    ans_tot = eval_data["answerable_total"]
    abst_corr = eval_data["abstention_correct"]
    abst_tot = eval_data["abstention_total"]
    err_cnt = eval_data.get("error_count", 0)
    mean_rec = eval_data["mean_recall"]
    rec_count = eval_data["doc_recall_count"]
    tot_cit = eval_data["total_citations"]
    val_cit = eval_data["valid_citations"]
    inv_cit = eval_data["invalid_citations"]

    cit_pct = (val_cit / tot_cit * 100.0) if tot_cit > 0 else 100.0

    lines.append("\n### Per-Run Summary")
    lines.append(
        f"- Total Accuracy: {corr}/{total} ({corr/total*100.0:.1f}%)"
    )
    lines.append(
        f"- Answerable Correctness: {ans_corr}/{ans_tot} ({ans_corr/ans_tot*100.0:.1f}%)"
    )
    lines.append(
        f"- Abstention Correctness: {abst_corr}/{abst_tot} ({abst_corr/abst_tot*100.0:.1f}%)"
    )
    lines.append(
        f"- Error Count: {err_cnt}/{total} ({err_cnt/total*100.0:.1f}%)"
    )
    lines.append(
        f"- Mean Required-Doc Recall: {mean_rec:.4f} ({mean_rec*100.0:.1f}%) [Denominator: {rec_count} answerable questions]"
    )
    lines.append(
        f"- Citation-ID Validity: {val_cit}/{tot_cit} ({cit_pct:.1f}%) valid chunk IDs (Invalid/Hallucinated: {len(inv_cit)})"
    )
    if inv_cit:
        lines.append(f"  Invalid citations detail: {inv_cit}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hybrid-graphrag run results."
    )
    parser.add_argument(
        "results_paths", nargs="+", help="Paths to results.jsonl files"
    )
    parser.add_argument(
        "--questions",
        default="experiments/synthetic_academic/questions.jsonl",
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/synthetic_academic/runs",
        help="Directory to save timestamped analysis file",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(
        args.output_dir, f"analysis_{timestamp}.txt"
    )

    all_outputs = []
    for r_path in args.results_paths:
        if not os.path.exists(r_path):
            print(f"Warning: {r_path} not found.", file=sys.stderr)
            continue
        results = load_results(r_path)
        eval_data = evaluate_run(results, questions)
        text_out = format_evaluation_output(r_path, eval_data)
        all_outputs.append(text_out)
        print(text_out)

    combined_text = "\n\n".join(all_outputs)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"\nSaved analysis to: {out_file}")


if __name__ == "__main__":
    main()
