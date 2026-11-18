#!/usr/bin/env python3
"""Report generator for Synthetic KG Benchmark.

Generates experiments/synth_kg/REPORT.md from evaluation results and disk artifacts.
Can be executed standalone:
    python3 experiments/synth_kg/report.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def format_ci(ci: List[float]) -> str:
    """Format confidence interval as percentage range."""
    if not ci or len(ci) < 2:
        return "[0.0%, 0.0%]"
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def generate_report_markdown(
    eval_results: Dict[str, Any],
    build_stats: Optional[Dict[str, Any]] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    incomplete: bool = False,
    incomplete_reason: str = "",
) -> str:
    """Construct full Markdown report according to protocol."""
    lines: List[str] = []

    lines.append("# Synthetic KG Benchmark: Evaluation Report")
    lines.append("")
    lines.append(f"**Date Generated**: 2026-11-16  ")
    lines.append(f"**Status**: {'**INCOMPLETE (Time Cap Hit)**' if incomplete else '**COMPLETED**'}  ")
    if incomplete and incomplete_reason:
        lines.append(f"> [!WARNING]\n> Benchmark run terminated early: {incomplete_reason}\n")

    # 1. Settings Snapshot
    lines.append("## 1. Experimental Settings Snapshot")
    lines.append("")
    lines.append("| Parameter | Setting / Value |")
    lines.append("| :--- | :--- |")
    lines.append("| **Public API Facade** | `hybrid_graphrag.HybridGraphRAG` |")
    lines.append("| **Base Config** | `configs/default.yaml` |")
    lines.append("| **Domain Config** | `configs/domains/company.yaml` |")
    lines.append("| **Ontology Domain** | `company_knowledge` (5 entity types, 5 relationship types) |")
    lines.append("| **KB Storage Directory** | `experiments/synth_kg/kb/synth_company_kb` |")
    lines.append("| **LLM Model** | `gemma3:4b` (Temperature: 0.0) |")
    lines.append("| **Embedding Model** | `embeddinggemma:latest` (Dim: 768) |")
    lines.append("| **Retrieval Top-K** | `vector_top_k = 4` |")
    lines.append("| **Final Context Chunks** | `final_context_chunks = 4` |")
    lines.append("| **Comparative Reasoning Mode** | `comparative_reasoning_mode = off` |")
    lines.append("| **Graph Context Mode** | `graph_context_mode = off` (False) |")
    lines.append("| **Answer Prompt Version** | `v1.1` |")
    lines.append("| **Corpus Path** | `experiments/synth_kg/corpus_final/` (60 docs, 95 gold edges) |")
    lines.append("| **Questions Count** | 120 total (30 hop1, 30 hop2, 30 hop3, 15 alias, 15 unanswerable) |")
    lines.append("")

    # 2. Build Statistics
    lines.append("## 2. Knowledge Base Build Statistics")
    lines.append("")
    ingest_stats = (build_stats or {}).get("ingest_stats", {})
    kb_stats = (build_stats or {}).get("kb_stats", {})

    lines.append("| Metric | Count / Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Documents Ingested | {ingest_stats.get('documents', 'N/A')} |")
    lines.append(f"| Chunks Created | {ingest_stats.get('chunks', 'N/A')} |")
    doc_cnt = ingest_stats.get("documents", 0)
    chk_cnt = ingest_stats.get("chunks", 0)
    ratio_str = f"{chk_cnt / doc_cnt:.2f} chunks/doc" if doc_cnt else "N/A"
    lines.append(f"| Chunks vs Documents Ratio | {ratio_str} |")
    lines.append(f"| Extracted Entity Mentions | {ingest_stats.get('entity_mentions', 'N/A')} |")
    lines.append(f"| Resolved Canonical Entities | {ingest_stats.get('canonical_entities', 'N/A')} |")
    lines.append(f"| Total Extracted Triples | {ingest_stats.get('extracted_relationships', 'N/A')} |")
    lines.append(f"| Validated Relationships | {ingest_stats.get('validated_relationships', 'N/A')} |")
    lines.append(f"| Rejected Relationships | {ingest_stats.get('rejected_relationships', 'N/A')} |")
    lines.append(f"| Communities Detected | {ingest_stats.get('communities', 'N/A')} |")
    lines.append(f"| Community Reports Generated | {ingest_stats.get('community_reports', 'N/A')} |")
    lines.append(f"| Graph Vertices | {ingest_stats.get('graph_vertices', 'N/A')} |")
    lines.append(f"| Graph Edges | {ingest_stats.get('graph_edges', 'N/A')} |")
    lines.append(f"| Connected Components | {ingest_stats.get('graph_connected_components', 'N/A')} |")
    lines.append(f"| Isolated Nodes | {ingest_stats.get('graph_isolated_nodes', 'N/A')} |")
    lines.append(f"| Graph Degree (Min / Mean / Max) | {ingest_stats.get('graph_degree_min', 0)} / {ingest_stats.get('graph_degree_mean', 0.0):.2f} / {ingest_stats.get('graph_degree_max', 0)} |")
    lines.append("")

    # Build Stage Timings
    stage_secs = ingest_stats.get("stage_seconds", {})
    if stage_secs:
        lines.append("### Build Wall-Clock Timings by Stage")
        lines.append("")
        lines.append("| Pipeline Stage | Duration (seconds) | Duration (minutes) |")
        lines.append("| :--- | :--- | :--- |")
        for sname, ssec in stage_secs.items():
            lines.append(f"| `{sname}` | {ssec:.2f}s | {ssec/60:.2f}m |")
        total_bld = (build_stats or {}).get("build_wall_seconds", stage_secs.get("total_indexing", 0.0))
        lines.append(f"| **Total Build Wall Clock** | **{total_bld:.2f}s** | **{total_bld/60:.2f}m** |")
        lines.append("")

    # Rejected Relationships by Reason
    rej_reasons = ingest_stats.get("rejected_relationships_by_reason", {})
    if rej_reasons:
        lines.append("### Rejected Triples by Reason Code")
        lines.append("")
        lines.append("| Rejection Reason | Count | Fraction of Rejected |")
        lines.append("| :--- | :--- | :--- |")
        tot_rej = sum(rej_reasons.values())
        for rcode, rcnt in sorted(rej_reasons.items(), key=lambda x: -x[1]):
            frac = (rcnt / tot_rej * 100) if tot_rej else 0.0
            lines.append(f"| `{rcode}` | {rcnt} | {frac:.1f}% |")
        lines.append("")

    # 3. Extraction Quality Scoring
    ext_res = eval_results.get("extraction_results", {})
    ent_s = ext_res.get("entity_scoring", {})
    rel_s = ext_res.get("relationship_scoring", {})

    lines.append("## 3. Extraction Quality Scoring (against gold_kg.json)")
    lines.append("")
    lines.append("### Entity Resolution Scoring")
    lines.append("")
    lines.append("| Metric | Count / Fraction | Wilson / Notes |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| Gold Entities Total | {ent_s.get('gold_entities_total', 0)} | Ground truth entities |")
    lines.append(f"| Extracted Entities Total | {ent_s.get('extracted_entities_total', 0)} | Extracted canonical entities |")
    lines.append(f"| Recalled Gold Entities | {ent_s.get('recalled_gold_entities', 0)} | Matched by $\\ge 1$ canonical entity |")
    lines.append(f"| Valid Extracted Entities | {ent_s.get('valid_extracted_entities', 0)} | Matched to gold entity |")
    lines.append(f"| **Entity Recall** | **{ent_s.get('recall', 0.0)*100:.2f}%** | Recalled / Gold Total |")
    lines.append(f"| **Entity Precision** | **{ent_s.get('precision', 0.0)*100:.2f}%** | Valid / Extracted Total |")
    lines.append(f"| Entity Splits | {ent_s.get('splits', 0)} | 1 gold entity $\\rightarrow \\ge 2$ extracted |")
    lines.append(f"| Entity False Merges | {ent_s.get('false_merges', 0)} | 1 extracted entity $\\rightarrow \\ge 2$ gold |")
    lines.append("")

    lines.append("### Validated Relationship Scoring")
    lines.append("")
    lines.append("| Metric | Count / Score | Interpretation |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| Gold Edges Total | {rel_s.get('gold_edges_total', 0)} | Ground truth edges |")
    lines.append(f"| Validated Edges Total | {rel_s.get('validated_edges_total', 0)} | Edges in validated graph |")
    lines.append(f"| Exact Matched Validated Edges | {rel_s.get('exact_matched_validated', 0)} | Matches (source, target, relation) |")
    lines.append(f"| Exact Recalled Gold Edges | {rel_s.get('exact_recalled_gold', 0)} | Gold edges recalled exactly |")
    lines.append(f"| **Exact Relationship Precision** | **{rel_s.get('exact_precision', 0.0)*100:.2f}%** | Exact matches / Validated edges |")
    lines.append(f"| **Exact Relationship Recall** | **{rel_s.get('exact_recall', 0.0)*100:.2f}%** | Recalled exact / Gold edges |")
    lines.append(f"| Relaxed Matched Validated Edges | {rel_s.get('relaxed_matched_validated', 0)} | Matches undirected {{u, v}} |")
    lines.append(f"| Relaxed Recalled Gold Edges | {rel_s.get('relaxed_recalled_gold', 0)} | Gold edges recalled relaxed |")
    lines.append(f"| **Relaxed Relationship Precision** | **{rel_s.get('relaxed_precision', 0.0)*100:.2f}%** | Relaxed matches / Validated edges |")
    lines.append(f"| **Relaxed Relationship Recall** | **{rel_s.get('relaxed_recall', 0.0)*100:.2f}%** | Recalled relaxed / Gold edges |")
    lines.append("")

    # 4. Gold Edge Fate Breakdown
    fate_data = ext_res.get("fate_breakdown", {})
    fate_counts = fate_data.get("counts", {})
    lines.append("## 4. Gold Edge Fate Breakdown")
    lines.append("")
    lines.append("| Fate Category | Count | Percentage of Gold Edges |")
    lines.append("| :--- | :--- | :--- |")
    tot_gold_e = rel_s.get("gold_edges_total", 95)
    for fate_name, count in sorted(fate_counts.items(), key=lambda x: -x[1]):
        pct = (count / tot_gold_e * 100) if tot_gold_e else 0.0
        lines.append(f"| `{fate_name}` | {count} | {pct:.1f}% |")
    lines.append("")

    # 5. Question Answering Performance by Mode & Group
    lines.append("## 5. Question Answering Performance by Mode and Question Group")
    lines.append("")

    mode_sums = eval_results.get("mode_summaries", {})
    modes = ["vector_only", "hybrid", "graph_only", "community_aware"]
    groups = ["hop1", "hop2", "hop3", "alias", "unanswerable"]

    for mode in modes:
        if mode not in mode_sums:
            continue
        ms = mode_sums[mode]
        lines.append(f"### Mode: `{mode}`")
        lines.append("")
        lines.append(f"- **Overall Answerable Accuracy**: {ms.get('answerable_accuracy', 0.0)*100:.2f}% (95% CI: {format_ci(ms.get('answerable_ci_95', []))})")
        lines.append(f"- **Overall Corpus Accuracy (inc. Unanswerable)**: {ms.get('overall_accuracy', 0.0)*100:.2f}% (95% CI: {format_ci(ms.get('overall_ci_95', []))})")
        lines.append("")
        lines.append("| Question Group | Total | Correct | Accuracy | Wilson 95% CI | ERROR Count | Evidence Recall | Mean Latency |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        bg = ms.get("by_group", {})
        for g in groups:
            gd = bg.get(g, {})
            tot = gd.get("total", 0)
            corr = gd.get("correct", 0)
            acc = gd.get("accuracy", 0.0) * 100
            ci = format_ci(gd.get("ci_95", []))
            err = gd.get("error", 0)
            ev = gd.get("evidence_recall", 0.0) * 100
            lat = gd.get("mean_latency", 0.0)
            lines.append(f"| `{g}` | {tot} | {corr} | {acc:.1f}% | {ci} | {err} | {ev:.1f}% | {lat:.2f}s |")
        lines.append("")

    # 6. Paired Statistical Comparisons
    lines.append("## 6. Paired Statistical Comparisons vs. `vector_only`")
    lines.append("")
    paired = eval_results.get("paired_comparisons", {})

    for comp_key, comp_groups in paired.items():
        cand_name = comp_key.split("_vs_")[0]
        lines.append(f"### Candidate: `{cand_name}` vs. Reference: `vector_only`")
        lines.append("")
        lines.append("| Question Slice | Both Correct ($n_{11}$) | Only Cand ($n_{10}$) | Only Vec ($n_{01}$) | Neither ($n_{00}$) | Discordant | Exact Sign Test $p$-value | Significant ($p < 0.05$) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for grp_name in ["all_answerable", "hop1", "hop2", "hop3", "alias"]:
            cd = comp_groups.get(grp_name, {})
            n11 = cd.get("both_correct", 0)
            n10 = cd.get("only_cand_correct", 0)
            n01 = cd.get("only_ref_correct", 0)
            n00 = cd.get("neither_correct", 0)
            disc = cd.get("discordant", 0)
            pval = cd.get("p_value", 1.0)
            sig_str = "**DETECTED**" if cd.get("difference_detected", False) else "No detectable difference"
            lines.append(f"| `{grp_name}` | {n11} | {n10} | {n01} | {n00} | {disc} | {pval:.4f} | {sig_str} |")
        lines.append("")

    # 7. Evidence Document Recall
    lines.append("## 7. Gold Evidence Supporting Document Recall")
    lines.append("")
    lines.append("Fraction of questions whose retrieved context chunks contain **all** gold supporting documents:")
    lines.append("")
    lines.append("| Mode | hop1 (1 doc) | hop2 (2 docs) | hop3 (3 docs) | alias (1-2 docs) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    for mode in modes:
        if mode not in mode_sums:
            continue
        bg = mode_sums[mode].get("by_group", {})
        h1 = bg.get("hop1", {}).get("evidence_recall", 0.0) * 100
        h2 = bg.get("hop2", {}).get("evidence_recall", 0.0) * 100
        h3 = bg.get("hop3", {}).get("evidence_recall", 0.0) * 100
        al = bg.get("alias", {}).get("evidence_recall", 0.0) * 100
        lines.append(f"| `{mode}` | {h1:.1f}% | {h2:.1f}% | {h3:.1f}% | {al:.1f}% |")
    lines.append("")

    # 8. Failure Class Breakdown
    lines.append("## 8. Failure Class Breakdown (Answerable Questions)")
    lines.append("")
    lines.append("| Mode | Total Answerable | Correct | ERROR | ABSTAINED | WRONG_ENTITY | OTHER |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for mode in modes:
        if mode not in mode_sums:
            continue
        bg = mode_sums[mode].get("by_group", {})
        tot = sum(bg.get(g, {}).get("total", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        corr = sum(bg.get(g, {}).get("correct", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        err = sum(bg.get(g, {}).get("error", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        abs_cnt = sum(bg.get(g, {}).get("abstained", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        we = sum(bg.get(g, {}).get("wrong_entity", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        oth = sum(bg.get(g, {}).get("other", 0) for g in ["hop1", "hop2", "hop3", "alias"])
        lines.append(f"| `{mode}` | {tot} | {corr} | {err} | {abs_cnt} | {we} | {oth} |")
    lines.append("")

    # 9. Path Coverage & Conditional Accuracies
    lines.append("## 9. Path Coverage and Conditional Accuracy")
    lines.append("")
    path_cov = eval_results.get("path_coverage_results", {})
    grp_cov = path_cov.get("group_path_coverage", {})

    lines.append("### Graph Path Coverage on Validated Graph")
    lines.append("")
    lines.append("| Multi-Hop Group | Total Questions | Validated Path Exists (length $\\le h$) | Path Coverage Fraction |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for g in ["hop2", "hop3"]:
        gdata = grp_cov.get(g, {})
        tot = gdata.get("total", 0)
        pe = gdata.get("path_exists", 0)
        frac = gdata.get("fraction", 0.0) * 100
        lines.append(f"| `{g}` | {tot} | {pe} | {frac:.1f}% |")
    lines.append("")

    cond_accs = eval_results.get("conditional_accuracies", {})
    lines.append("### Accuracy Conditional on Path Existence in Validated Graph")
    lines.append("")
    lines.append("| Mode & Group | Path Exists (Correct / Total) | Path Exists Accuracy | Path Missing (Correct / Total) | Path Missing Accuracy |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for mode in ["graph_only", "hybrid", "vector_only"]:
        for grp in ["hop2", "hop3"]:
            key = f"{mode}_{grp}"
            ca = cond_accs.get(key, {})
            pe_data = ca.get("path_exists", {})
            npe_data = ca.get("path_missing", {})
            pe_c = pe_data.get("correct", 0)
            pe_t = pe_data.get("total", 0)
            pe_a = pe_data.get("accuracy", 0.0) * 100
            npe_c = npe_data.get("correct", 0)
            npe_t = npe_data.get("total", 0)
            npe_a = npe_data.get("accuracy", 0.0) * 100
            lines.append(f"| `{mode}` ({grp}) | {pe_c}/{pe_t} | {pe_a:.1f}% | {npe_c}/{npe_t} | {npe_a:.1f}% |")
    lines.append("")

    # 10. Per-Question Matrix
    per_q = eval_results.get("per_question_eval", {})
    lines.append("## 10. Per-Question Results Matrix")
    lines.append("")
    lines.append("| Question ID | Group | vector_only | hybrid | graph_only | community_aware |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    q_groups = {}
    q_file = Path(__file__).resolve().parent / "corpus_final" / "questions.jsonl"
    if q_file.exists():
        with open(q_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    q_groups[item["id"]] = item.get("group", "unknown")

    def symbol_for_eval(eval_data: Dict[str, Any]) -> str:
        if not eval_data:
            return "-"
        if eval_data.get("failure_class") == "ERROR":
            return "ERR"
        if eval_data.get("failure_class") == "ABSTAINED":
            return "ABS"
        if eval_data.get("correct"):
            return "✓"
        return "✗"

    # Sort questions by ID
    for qid in sorted(per_q.keys()):
        # Determine group from loaded questions or fallback
        q_modes = per_q[qid]
        grp = q_groups.get(qid, "unknown")
        if grp == "unknown":
            if "hop1" in qid.lower():
                grp = "hop1"
            elif "hop2" in qid.lower():
                grp = "hop2"
            elif "hop3" in qid.lower():
                grp = "hop3"
            elif "alias" in qid.lower():
                grp = "alias"
            elif "unans" in qid.lower():
                grp = "unanswerable"

        s_vec = symbol_for_eval(q_modes.get("vector_only", {}))
        s_hyb = symbol_for_eval(q_modes.get("hybrid", {}))
        s_gro = symbol_for_eval(q_modes.get("graph_only", {}))
        s_com = symbol_for_eval(q_modes.get("community_aware", {}))
        lines.append(f"| `{qid}` | `{grp}` | {s_vec} | {s_hyb} | {s_gro} | {s_com} |")
    lines.append("")

    # 11. Latency per Mode
    lines.append("## 11. Latency by Mode")
    lines.append("")
    lines.append("| Mode | Mean Latency per Question (s) | Total Query Time (s) | Total Query Time (m) |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for mode in modes:
        if mode not in mode_sums:
            continue
        all_lats = []
        for qid, qm in per_q.items():
            if mode in qm and qm[mode].get("latency", 0.0) > 0:
                all_lats.append(qm[mode]["latency"])
        mean_lat = (sum(all_lats) / len(all_lats)) if all_lats else 0.0
        tot_lat = sum(all_lats)
        lines.append(f"| `{mode}` | {mean_lat:.2f}s | {tot_lat:.2f}s | {tot_lat/60:.2f}m |")
    lines.append("")

    # 12. Example Wrong Answers per Mode
    lines.append("## 12. Example Wrong Answers per Mode (Up to 6 per mode)")
    lines.append("")

    # We need questions lookup for question text & gold answer
    questions_file = Path("experiments/synth_kg/corpus_final/questions.jsonl")
    q_lookup = {}
    if questions_file.exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    q_lookup[item["id"]] = item

    for mode in modes:
        if mode not in mode_sums:
            continue
        lines.append(f"### Mode: `{mode}` Examples")
        lines.append("")
        count = 0
        for qid, qm in per_q.items():
            ev_data = qm.get(mode, {})
            if not ev_data.get("correct", False) and ev_data.get("failure_class") != "ERROR":
                q_item = q_lookup.get(qid, {})
                q_text = q_item.get("question", "N/A")
                gold_ans = q_item.get("gold_answer", "N/A")
                pred = str(ev_data.get("prediction", "")).strip()[:250]
                fail_cls = ev_data.get("failure_class", "N/A")

                lines.append(f"**[{count+1}] Question `{qid}` ({q_item.get('group', '')})**: {q_text}  ")
                lines.append(f"- **Gold Answer**: {gold_ans}  ")
                lines.append(f"- **Prediction**: {pred}  ")
                lines.append(f"- **Failure Class**: `{fail_cls}`  ")
                lines.append("")
                count += 1
                if count >= 6:
                    break
        if count == 0:
            lines.append("No wrong answers recorded for this mode.\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from experiments.synth_kg.score import score_runs_against_disk

    parser = argparse.ArgumentParser(description="Generate REPORT.md for Synthetic KG Benchmark")
    parser.add_argument("--corpus_dir", default="experiments/synth_kg/corpus_final")
    parser.add_argument("--kb_dir", default="experiments/synth_kg/kb/synth_company_kb")
    parser.add_argument("--runs_dir", default="experiments/synth_kg/runs")
    parser.add_argument("--eval_json", default="experiments/synth_kg/evaluation_results.json")
    parser.add_argument("--out", default="experiments/synth_kg/REPORT.md")
    parser.add_argument("--incomplete", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    eval_json_path = Path(args.eval_json)
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
    else:
        print("Computing evaluation results from disk...")
        eval_results = score_runs_against_disk(
            corpus_final_dir=Path(args.corpus_dir),
            kb_dir=Path(args.kb_dir),
            runs_dir=Path(args.runs_dir),
        )
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

    # Load build stats if available
    build_stats_path = Path(args.kb_dir) / "build_stats.json"
    build_stats = None
    if build_stats_path.exists():
        with open(build_stats_path, "r", encoding="utf-8") as f:
            build_stats = json.load(f)

    report_md = generate_report_markdown(
        eval_results=eval_results,
        build_stats=build_stats,
        incomplete=args.incomplete,
        incomplete_reason=args.reason,
    )

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report generated successfully at {out_path}")
