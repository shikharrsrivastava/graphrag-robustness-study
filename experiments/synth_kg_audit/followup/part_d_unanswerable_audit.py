#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_d_unanswerable_audit.py
Detailed per-question audit of the 15 unanswerable questions across all 4 modes.
"""

import json
import re
from pathlib import Path
from collections import Counter
import datetime

def parse_unanswerable_q(q_text: str):
    m = re.search(r"What client organization is managed by (.*?)\?", q_text)
    if m:
        return m.group(1).strip(), "CLIENT", "managed_client"
    m = re.search(r"What product is directly developed by (.*?)\?", q_text)
    if m:
        return m.group(1).strip(), "PRODUCT", "developed_product"
    m = re.search(r"What product does (.*?) develop or manage\?", q_text)
    if m:
        return m.group(1).strip(), "PRODUCT", "develop_or_manage_product"
    return "UNKNOWN", "UNKNOWN", "unknown"

def classify_mode_answer(sub: str, target_type: str, edges_from_sub: list, res: dict) -> str:
    abstained = res.get("abstained", False)
    ans = res.get("answer", "")
    
    if abstained:
        return "CORRECT_ABSTENTION"
    
    target_names = [e["target_name"] for e in edges_from_sub]
    names_mentioned = [name for name in target_names if name.lower() in ans.lower()]
    
    if len(edges_from_sub) > 0:
        if names_mentioned:
            return "NEAR_ANSWERABLE"
        else:
            return "ANSWERABLE_OTHER_ENTITY"
    else:
        return "TRULY_UNANSWERABLE_WRONG_ANSWER"

def main():
    gold_kg = json.loads(Path("experiments/synth_kg/corpus_final/gold_kg.json").read_text(encoding="utf-8"))
    gold_edges = gold_kg["edges"]

    questions = [json.loads(line) for line in open("experiments/synth_kg/corpus_final/questions.jsonl", encoding="utf-8")]
    unans = [q for q in questions if q.get("group") == "unanswerable"]

    modes = {
        "vector_only": {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_035101_vector_only/results.jsonl", encoding="utf-8")]},
        "hybrid": {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_045328_hybrid/results.jsonl", encoding="utf-8")]},
        "graph_only": {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_062038_graph_only/results.jsonl", encoding="utf-8")]},
        "community_aware": {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_074405_community_aware/results.jsonl", encoding="utf-8")]},
    }

    report = []
    report.append("================================================================================")
    report.append("PART D: UNANSWERABLE QUESTIONS AUDIT (PER QUESTION)")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append(f"Auditing {len(unans)} unanswerable questions across 4 modes")
    report.append("================================================================================\n")

    counts_per_mode = {m: Counter() for m in modes}
    counts_per_template = {"managed_client": Counter(), "developed_product": Counter(), "develop_or_manage_product": Counter()}

    for idx, q in enumerate(unans, 1):
        qid = q["id"]
        q_text = q["question"]
        sub, target_type, template = parse_unanswerable_q(q_text)

        edges_from_sub = [e for e in gold_edges if e["source_name"] == sub and e["target_type"] == target_type]

        report.append(f"### [{idx}] Question ID: `{qid}`")
        report.append(f"- **Question**: \"{q_text}\"")
        report.append(f"- **Template**: `{template}`")
        report.append(f"- **Subject Entity**: `{sub}`")
        report.append(f"- **Asked-About Target Type**: `{target_type}`")
        
        if edges_from_sub:
            report.append(f"- **Gold Edges from Subject to '{target_type}'** ({len(edges_from_sub)} found):")
            for e in edges_from_sub:
                report.append(f"  * Edge `{e['id']}`: `{e['source_name']}` -[{e['relation']}]-> `{e['target_name']}` ({e['target_type']}) [Doc: {e['supporting_doc_id']}]")
                report.append(f"    Verbatim Corpus Sentence: \"{e['evidence_quote']}\"")
        else:
            report.append(f"- **Gold Edges from Subject to '{target_type}'**: NONE")

        report.append("- **Mode Answers and Classifications**:")
        for mname, mresults in modes.items():
            res = mresults.get(qid, {})
            ans = res.get("answer", "").strip().replace("\n", " ")
            ans_trimmed = ans[:250] + ("..." if len(ans) > 250 else "")
            abstained = res.get("abstained", False)

            classification = classify_mode_answer(sub, target_type, edges_from_sub, res)
            counts_per_mode[mname][classification] += 1
            counts_per_template[template][classification] += 1

            report.append(f"  * **{mname}**: [Abstained={abstained}] -> **{classification}**")
            report.append(f"    Answer: \"{ans_trimmed}\"")
        report.append("")

    report.append("--------------------------------------------------------------------------------")
    report.append("COUNTS PER MODE:")
    for mname, c in counts_per_mode.items():
        report.append(f"- Mode `{mname}`:")
        for cls_name in ["NEAR_ANSWERABLE", "ANSWERABLE_OTHER_ENTITY", "TRULY_UNANSWERABLE_WRONG_ANSWER", "CORRECT_ABSTENTION"]:
            report.append(f"  * {cls_name:32s}: {c[cls_name]}")

    report.append("\nCOUNTS PER TEMPLATE (AGGREGATED ACROSS ALL 4 MODES, 60 TOTAL EVALUATIONS):")
    for tname, c in counts_per_template.items():
        report.append(f"- Template `{tname}` (N={sum(c.values()) // 4} questions * 4 modes = {sum(c.values())} evals):")
        for cls_name in ["NEAR_ANSWERABLE", "ANSWERABLE_OTHER_ENTITY", "TRULY_UNANSWERABLE_WRONG_ANSWER", "CORRECT_ABSTENTION"]:
            report.append(f"  * {cls_name:32s}: {c[cls_name]}")
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
        f.write(f"| {timestamp} | Part D: Audited 15 unanswerable questions per question, mode, and template | `experiments/synth_kg_audit/followup/part_d_unanswerable_audit.py` |\n")

if __name__ == "__main__":
    main()
