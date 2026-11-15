#!/usr/bin/env python3
"""Milestone 2 Workflow Script for Hybrid GraphRAG.

Executes build and query pipelines strictly through the public API
(hybrid_graphrag.HybridGraphRAG). Does not instantiate internal classes directly.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import yaml

from hybrid_graphrag import HybridGraphRAG, QueryResult


def run_build(
    corpus_dir: Path,
    kb_id: str,
    storage_root: Path,
    base_dir: Path = Path("."),
) -> None:
    """Execute Stage 1-10 offline build of the knowledge base through the public API."""
    print("=" * 80)
    print(f"BUILD: Initializing new Knowledge Base '{kb_id}' in '{storage_root}'")
    print(f"Corpus Directory: {corpus_dir}")
    print("=" * 80)

    start_time = time.perf_counter()

    config = {
        "kb_id": kb_id,
        "storage_root": str(storage_root),
    }

    # Initialize facade via public from_config
    rag = HybridGraphRAG.from_config(config=config, kb_id=kb_id, base_dir=base_dir)

    print(f"Active KB ID: {rag.kb_id}")
    print(f"Active KB Directory: {rag.kb_dir}")

    # Ingest corpus
    print("\nStarting ingestion and offline indexing...")
    ingest_start = time.perf_counter()
    stats = rag.ingest(input_path=corpus_dir, rebuild=True)
    ingest_duration = time.perf_counter() - ingest_start

    total_duration = time.perf_counter() - start_time

    print(f"\nIngestion and index build complete in {ingest_duration:.2f}s (Total: {total_duration:.2f}s)")
    print("\n--- Ingestion Statistics ---")
    for key, val in stats.items():
        if key == "stage_seconds":
            print("Stage Timings:")
            for s_name, s_sec in val.items():
                print(f"  - {s_name}: {s_sec:.3f}s")
        else:
            print(f"  {key}: {val}")

    print("\n--- KB State Summary (get_kb_stats) ---")
    kb_stats = rag.get_kb_stats()
    for key, val in kb_stats.items():
        print(f"  {key}: {val}")

    rag.close()
    print("\nBuild complete. Facade closed successfully.")


def run_query(
    kb_dir: Path,
    questions_file: Path,
    run_name: str,
    mode: str = "hybrid",
    vector_top_k: Optional[int] = None,
    final_context_chunks: Optional[int] = None,
    task6: bool = False,
    graph_context_mode: Union[bool, str] = False,
    comparative_reasoning_mode: Optional[str] = None,
    prompt_version: Optional[str] = "v1.1",
    base_dir: Path = Path("."),
) -> Path:
    """Execute query evaluation on persisted KB through the public API."""
    mode_str = str(graph_context_mode).strip().lower()
    has_graph_mode_override = mode_str not in ("false", "off", "0", "")

    print("=" * 80)
    print(f"QUERY RUN: {run_name} (Mode: {mode})")
    print(f"KB Directory: {kb_dir}")
    print(f"Questions File: {questions_file}")
    if vector_top_k is not None:
        print(f"Config Override: vector_top_k = {vector_top_k}")
    if final_context_chunks is not None:
        print(f"Config Override: final_context_chunks = {final_context_chunks}")
    if task6:
        print("Config Override: entity_description_discovery_enabled = True (Task 6 enabled)")
    if mode_str in ("true", "fix_e", "edges"):
        print("Config Override: graph_context_mode = True (Fix E enabled)")
    elif mode_str == "none":
        print("Config Override: graph_context_mode = 'none' (Graph chunks only, no graph text)")
    if comparative_reasoning_mode is not None:
        print(f"Config Override: comparative_reasoning_mode = {comparative_reasoning_mode}")
    if prompt_version is not None:
        print(f"Config Override: prompt_versions.answer_generation = {prompt_version}")
    print("=" * 80)

    # Prepare run output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiments/synthetic_academic/runs") / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    run_meta_path = run_dir / "run_meta.json"

    # Load questions
    questions: List[Dict[str, Any]] = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Reload engine through public API
    has_overrides = (
        vector_top_k is not None
        or final_context_chunks is not None
        or task6
        or has_graph_mode_override
        or comparative_reasoning_mode is not None
        or prompt_version is not None
    )

    if not has_overrides:
        print("\nReloading engine using HybridGraphRAG.from_existing...")
        rag = HybridGraphRAG.from_existing(kb_dir=kb_dir, base_dir=base_dir)
    else:
        print("\nReloading engine with configuration overrides via HybridGraphRAG.from_config...")
        snapshot_path = kb_dir / "config_snapshot.yaml"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"config_snapshot.yaml not found in {kb_dir}")
        with open(snapshot_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        if vector_top_k is not None:
            cfg_dict.setdefault("retrieval", {})["vector_top_k"] = vector_top_k
        if final_context_chunks is not None:
            cfg_dict.setdefault("retrieval", {})["final_context_chunks"] = final_context_chunks
        if task6:
            cfg_dict.setdefault("retrieval", {})["entity_description_discovery_enabled"] = True
        if mode_str in ("true", "fix_e", "edges"):
            cfg_dict.setdefault("retrieval", {})["graph_context_mode"] = True
        elif mode_str == "none":
            cfg_dict.setdefault("retrieval", {})["graph_context_mode"] = "none"
        else:
            cfg_dict.setdefault("retrieval", {})["graph_context_mode"] = False
        if comparative_reasoning_mode is not None:
            cfg_dict.setdefault("retrieval", {})["comparative_reasoning_mode"] = comparative_reasoning_mode
        if prompt_version is not None:
            cfg_dict.setdefault("prompt_versions", {})["answer_generation"] = prompt_version

        rag = HybridGraphRAG.from_config(config=cfg_dict, base_dir=base_dir)

    print(f"Attached to KB: {rag.kb_id}")
    active_retrieval = rag.config.retrieval
    print(
        f"Active Limits: vector_top_k={active_retrieval.vector_top_k}, "
        f"final_context_chunks={active_retrieval.final_context_chunks}, "
        f"max_graph_candidates={active_retrieval.max_graph_candidates}, "
        f"rrf_k={active_retrieval.rrf_k}, "
        f"task6_enabled={active_retrieval.entity_description_discovery_enabled}, "
        f"graph_context_mode={getattr(active_retrieval, 'graph_context_mode', False)}"
    )

    query_records: List[Dict[str, Any]] = []

    for idx, q_item in enumerate(questions, start=1):
        q_id = q_item["id"]
        q_text = q_item["question"]
        print(f"\n[{idx}/{len(questions)}] {q_id}: {q_text}")

        q_start = time.perf_counter()
        query_result: QueryResult = rag.query(question=q_text, mode=mode)
        q_duration = time.perf_counter() - q_start

        # Extract evidence list from raw_result if present
        evidence_records: List[Dict[str, Any]] = []
        route_diagnostics: Dict[str, Any] = {}
        if query_result.raw_result:
            route_diagnostics = dict(query_result.raw_result.route_diagnostics)
            for ev in query_result.raw_result.evidence_list:
                ev_data = {
                    "chunk_id": ev.chunk_id,
                    "doc_id": ev.doc_id,
                    "filename": ev.filename,
                    "page_numbers": list(ev.page_numbers),
                    "text": ev.text,
                    "routes": [r.value for r in ev.routes],
                    "graph_rank": ev.graph_rank,
                    "vector_rank": ev.vector_rank,
                    "vector_distance": ev.vector_distance,
                    "rrf_score": ev.rrf_score,
                    "graph_provenance_types": [p.value for p in ev.graph_provenance_types],
                    "supporting_graph_paths": list(ev.supporting_graph_paths),
                    "supporting_edge_ids": list(ev.supporting_edge_ids),
                    "document_title_details": [d.model_dump() for d in ev.document_title_details],
                    "entity_description_details": [d.model_dump() for d in ev.entity_description_details],
                    "cooccurrence_details": [d.model_dump() for d in ev.cooccurrence_details],
                }
                evidence_records.append(ev_data)

        record: Dict[str, Any] = {
            "question_id": q_id,
            "question": q_text,
            "category": q_item.get("category"),
            "answerable": q_item.get("answerable", True),
            "expected_answer_contains": q_item.get("expected_answer_contains", []),
            "required_docs": q_item.get("required_docs", []),
            "answer": query_result.answer,
            "canonical_answer": query_result.canonical_answer,
            "abstained": query_result.abstained,
            "abstention_reason": query_result.abstention_reason,
            "generation_error": getattr(query_result, "generation_error", False) or (getattr(query_result.raw_result, "generation_error", False) if query_result.raw_result else False),
            "citations": [c.model_dump() for c in query_result.citations],
            "cited_chunk_ids": list(query_result.cited_chunk_ids),
            "mode": query_result.mode,
            "latency_seconds": query_result.latency_seconds,
            "total_latency_seconds": q_duration,
            "timing": query_result.timing,
            "route_diagnostics": route_diagnostics,
            "evidence_list": evidence_records,
            "notes": q_item.get("notes", ""),
        }

        query_records.append(record)

        # Print quick status
        print(f"  Abstained: {record['abstained']}")
        print(f"  Answer: {record['answer'][:120]}...")
        print(f"  Citations: {record['cited_chunk_ids']}")
        print(f"  Evidence count: {len(evidence_records)}")

    # Write records
    with open(results_path, "w", encoding="utf-8") as f:
        for r in query_records:
            f.write(json.dumps(r) + "\n")

    run_meta = {
        "run_name": run_name,
        "timestamp": timestamp,
        "kb_dir": str(kb_dir),
        "mode": mode,
        "questions_count": len(questions),
        "vector_top_k": active_retrieval.vector_top_k,
        "final_context_chunks": active_retrieval.final_context_chunks,
        "task6_enabled": active_retrieval.entity_description_discovery_enabled,
        "graph_context_mode": getattr(active_retrieval, "graph_context_mode", False),
        "comparative_reasoning_mode": getattr(active_retrieval, "comparative_reasoning_mode", "enforce"),
        "prompt_versions": rag.config.prompt_versions.model_dump(),
        "results_path": str(results_path),
    }
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    rag.close()
    print(f"\nAll {len(questions)} queries complete. Results saved to: {results_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2 Workflow Runner")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: build
    build_parser = subparsers.add_parser("build", help="Build knowledge base from corpus")
    build_parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("experiments/synthetic_academic/corpus"),
        help="Directory containing corpus text files",
    )
    build_parser.add_argument(
        "--kb-id",
        type=str,
        default="synthetic_academic_01",
        help="Unique knowledge base identifier",
    )
    build_parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("knowledge_bases/synthetic_academic"),
        help="Storage root directory for the new KB",
    )

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Run queries on persisted KB")
    query_parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_bases/synthetic_academic/synthetic_academic_01"),
        help="Path to persisted knowledge base directory",
    )
    query_parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("experiments/synthetic_academic/questions.jsonl"),
        help="Path to questions JSONL file",
    )
    query_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Descriptive name for the run",
    )
    query_parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "vector_only", "graph_only", "community_aware"],
        help="Query retrieval mode",
    )
    query_parser.add_argument(
        "--vector-top-k",
        type=int,
        default=None,
        help="Override for vector_top_k",
    )
    query_parser.add_argument(
        "--final-context-chunks",
        type=int,
        default=None,
        help="Override for final_context_chunks",
    )
    query_parser.add_argument(
        "--task6",
        action="store_true",
        help="Enable Task 6 entity description discovery",
    )
    query_parser.add_argument(
        "--graph-context-mode",
        nargs="?",
        const="fix_e",
        default="off",
        help="Graph context formatting: 'off' (legacy paths), 'fix_e' (Fix E edges), or 'none' (chunks only)",
    )
    query_parser.add_argument(
        "--comparative-reasoning-mode",
        type=str,
        default="off",
        choices=["enforce", "off"],
        help="Comparative reasoning mode: 'enforce' or 'off'",
    )
    query_parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1.1",
        choices=["v1.0", "v1.1"],
        help="Answer generation prompt version: 'v1.0' or 'v1.1'",
    )

    args = parser.parse_args()

    if args.subcommand == "build":
        run_build(
            corpus_dir=args.corpus_dir,
            kb_id=args.kb_id,
            storage_root=args.storage_root,
        )
    elif args.subcommand == "query":
        run_query(
            kb_dir=args.kb_dir,
            questions_file=args.questions_file,
            run_name=args.run_name,
            mode=args.mode,
            vector_top_k=args.vector_top_k,
            final_context_chunks=args.final_context_chunks,
            task6=args.task6,
            graph_context_mode=args.graph_context_mode,
            comparative_reasoning_mode=args.comparative_reasoning_mode,
            prompt_version=args.prompt_version,
        )


if __name__ == "__main__":
    main()
