#!/usr/bin/env python3
"""Part 3: Ingestion cost measurement on 10 HotpotQA context paragraphs using public API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict

import yaml

from hybrid_graphrag import HybridGraphRAG
from src.core.configuration import deep_merge_dicts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def log_progress(progress_file: Path, step_name: str, output_path: str) -> None:
    now_iso = datetime.now().astimezone().isoformat()
    line = f"| {now_iso} | {step_name} | `{output_path}` |\n"
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3: Measure Ingestion Cost")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("experiments/hotpotqa_m4/scratch_docs"),
        help="Path to directory containing input documents",
    )
    parser.add_argument(
        "--kb-root",
        type=Path,
        default=Path("experiments/hotpotqa_m4/scratch_kb"),
        help="Scratch knowledge base root directory",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("experiments/hotpotqa_m4/ingestion_stats.json"),
        help="Path to save output JSON stats",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("experiments/hotpotqa_m4/PROGRESS.md"),
        help="Path to PROGRESS.md log",
    )
    args = parser.parse_args()

    # 1. Load configs in-memory without modifying disk files
    logger.info("Loading configs/default.yaml and configs/domains/academic.yaml...")
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open("configs/domains/academic.yaml", "r", encoding="utf-8") as f:
        domain_cfg = yaml.safe_load(f)

    # Academic domain chosen because it defines general-purpose entity types:
    # PERSON, LOCATION, ORGANIZATION, EVENT, WORK, FILM, BOOK, COUNTRY, CONCEPT, UNIVERSITY, PROJECT, TECHNOLOGY
    # which best matches general Wikipedia texts.
    merged_cfg = deep_merge_dicts(base_cfg, domain_cfg)
    merged_cfg["storage_root"] = str(args.kb_root)
    merged_cfg["kb_id"] = "scratch_hotpot_10"

    logger.info("Initializing HybridGraphRAG facade (kb_id: scratch_hotpot_10, storage_root: %s)...", args.kb_root)
    rag = HybridGraphRAG.from_config(merged_cfg, kb_id="scratch_hotpot_10")

    doc_files = list(args.docs_dir.glob("*.txt"))
    num_paras = len(doc_files)
    logger.info("Found %d paragraphs in %s to ingest.", num_paras, args.docs_dir)

    # 2. Ingest documents and measure wall time
    t0 = time.perf_counter()
    ingest_result = rag.ingest(args.docs_dir, rebuild=True)
    t1 = time.perf_counter()
    wall_time_seconds = t1 - t0

    sec_per_para = wall_time_seconds / num_paras if num_paras > 0 else 0.0

    # 3. Retrieve KB stats
    kb_stats = rag.get_kb_stats()

    # 4. Linear extrapolations
    extrap_300_sec = 300 * sec_per_para
    extrap_1000_sec = 1000 * sec_per_para

    # 5. Build detailed reporting dict
    report = {
        "timestamp_iso": datetime.now().astimezone().isoformat(),
        "domain_config_used": "configs/domains/academic.yaml",
        "domain_config_justification": (
            "academic.yaml provides broad open-domain ontology coverage (PERSON, LOCATION, "
            "ORGANIZATION, EVENT, WORK, FILM, BOOK, COUNTRY, CONCEPT) matching general Wikipedia paragraphs, "
            "whereas company, legal, and medical_research ontologies are domain-restricted."
        ),
        "num_paragraphs": num_paras,
        "total_wall_time_seconds": round(wall_time_seconds, 2),
        "seconds_per_paragraph": round(sec_per_para, 2),
        "documents": ingest_result.get("documents", 0),
        "pages": ingest_result.get("pages", 0),
        "chunks": ingest_result.get("chunks", 0),
        "entity_mentions": ingest_result.get("entity_mentions", 0),
        "canonical_entities": ingest_result.get("canonical_entities", 0),
        "extracted_relationships": ingest_result.get("extracted_relationships", 0),
        "validated_relationships": ingest_result.get("validated_relationships", 0),
        "rejected_relationships": ingest_result.get("rejected_relationships", 0),
        "rejected_relationships_by_reason": ingest_result.get("rejected_relationships_by_reason", {}),
        "communities": ingest_result.get("communities", 0),
        "community_reports": ingest_result.get("community_reports", 0),
        "leiden_ran": bool(ingest_result.get("communities", 0) > 0),
        "community_reports_ran": bool(ingest_result.get("community_reports", 0) > 0),
        "graph_vertices": ingest_result.get("graph_vertices", 0),
        "graph_edges": ingest_result.get("graph_edges", 0),
        "stage_seconds": {k: round(v, 2) for k, v in ingest_result.get("stage_seconds", {}).items()},
        "linear_extrapolations": {
            "300_paragraphs": {
                "seconds": round(extrap_300_sec, 2),
                "minutes": round(extrap_300_sec / 60, 2),
                "hours": round(extrap_300_sec / 3600, 2),
                "note": "Linear extrapolation based on 10-paragraph sample (O(N) extraction + O(V+E) graph ops)",
            },
            "1000_paragraphs": {
                "seconds": round(extrap_1000_sec, 2),
                "minutes": round(extrap_1000_sec / 60, 2),
                "hours": round(extrap_1000_sec / 3600, 2),
                "note": "Linear extrapolation based on 10-paragraph sample (O(N) extraction + O(V+E) graph ops)",
            },
        },
        "kb_stats": kb_stats,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("PART 3 INGESTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Wall Time:                 {wall_time_seconds:.2f} s ({wall_time_seconds / 60:.2f} min)")
    print(f"Seconds per Paragraph:           {sec_per_para:.2f} s/para")
    print(f"Paragraphs / Chunks:             {num_paras} / {report['chunks']}")
    print(f"Entities (Canonical / Mentions): {report['canonical_entities']} / {report['entity_mentions']}")
    print(f"Relationships Extracted:         {report['extracted_relationships']}")
    print(f"  Validated:                     {report['validated_relationships']}")
    print(f"  Rejected:                      {report['rejected_relationships']}")
    print(f"  Rejected by reason:            {report['rejected_relationships_by_reason']}")
    print(f"Communities (Leiden Ran):        {report['communities']} ({report['leiden_ran']})")
    print(f"Community Reports Ran:           {report['community_reports']} ({report['community_reports_ran']})")
    print(f"Graph Vertices / Edges:          {report['graph_vertices']} / {report['graph_edges']}")
    print("\nStage Seconds Breakdown:")
    for stage, s in report["stage_seconds"].items():
        print(f"  {stage:<30}: {s:>6.2f}s ({s/wall_time_seconds*100:>5.1f}%)")
    print("\nLinear Extrapolations (Explicitly Linear):")
    print(f"  300 paragraphs:  {report['linear_extrapolations']['300_paragraphs']['minutes']} min ({report['linear_extrapolations']['300_paragraphs']['hours']} hrs)")
    print(f"  1000 paragraphs: {report['linear_extrapolations']['1000_paragraphs']['minutes']} min ({report['linear_extrapolations']['1000_paragraphs']['hours']} hrs)")
    print(f"\nSaved full results to {args.output_file}")

    log_progress(
        args.progress_file,
        f"Part 3: Ingested 10 paragraphs ({wall_time_seconds:.1f}s, {sec_per_para:.1f}s/para, {report['canonical_entities']} entities, {report['validated_relationships']} val rels, {report['rejected_relationships']} rej rels)",
        str(args.output_file),
    )

    rag.close()


if __name__ == "__main__":
    main()
