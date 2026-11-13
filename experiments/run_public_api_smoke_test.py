#!/usr/bin/env python3
"""Public API Smoke Test against existing persisted HotpotQA knowledge base.

Verifies:
1. Public import: from hybrid_graphrag import HybridGraphRAG
2. Opening existing KB via HybridGraphRAG.from_existing()
3. Executing exactly one hybrid query through the public API
4. Verification of answer correctness, canonical answer, citations, latency, KB identity
5. Integrity check: ensuring existing graph, chunks, and embeddings are untouched
6. Saving structured output to experiments/hotpotqa/public_api_smoke_test.json
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_graphrag import HybridGraphRAG, QueryResult


def compute_file_checksum(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_kb_integrity_snapshot(kb_dir: Path) -> dict:
    """Capture checksums and file sizes of all core KB assets."""
    target_files = [
        kb_dir / "data" / "processed" / "chunks.jsonl",
        kb_dir / "data" / "processed" / "documents.jsonl",
        kb_dir / "graph" / "entities.jsonl",
        kb_dir / "graph" / "relationships.jsonl",
        kb_dir / "graph" / "communities.jsonl",
        kb_dir / "graph" / "community_reports.jsonl",
        kb_dir / "config_snapshot.yaml",
    ]
    snapshot = {}
    for p in target_files:
        if p.exists():
            snapshot[str(p.relative_to(kb_dir))] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_file_checksum(p),
                "mtime": p.stat().st_mtime,
            }
    return snapshot


def main():
    kb_path = Path("knowledge_bases/hotpotqa_eval/hotpot_full_5a8b57f25542995d1e6f1371").resolve()
    if not kb_path.exists():
        print(f"ERROR: Target KB does not exist at {kb_path}", file=sys.stderr)
        sys.exit(1)

    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    gold_answer = "yes"
    expected_chunks = ["D002_C0000", "D005_C0000"]
    output_path = Path("experiments/hotpotqa/public_api_smoke_test.json")

    print(f"=== Starting Public API Smoke Test ===")
    print(f"Target KB: {kb_path}")
    print(f"Question:  {question}")
    print(f"Gold:      {gold_answer}\n")

    # Step 1: Pre-query integrity snapshot
    pre_integrity = get_kb_integrity_snapshot(kb_path)
    print(f"Recorded pre-query checksums for {len(pre_integrity)} KB files.")

    # Step 2: Open existing KB through Public API
    open_start = time.perf_counter()
    with HybridGraphRAG.from_existing(kb_path) as engine:
        open_duration = time.perf_counter() - open_start
        print(f"Successfully opened KB in {open_duration:.3f}s. KB ID: {engine.kb_id}")

        kb_stats = engine.get_kb_stats()
        print(f"KB Statistics: {kb_stats}")

        # Step 3: Execute exactly one hybrid query through public API
        print("\nExecuting hybrid query through public API...")
        query_start = time.perf_counter()
        result: QueryResult = engine.query(question=question, mode="hybrid")
        measured_latency = time.perf_counter() - query_start

    # Step 4: Post-query integrity snapshot
    post_integrity = get_kb_integrity_snapshot(kb_path)
    print("\nRecorded post-query checksums.")

    # Step 5: Verify no KB files were modified
    integrity_matches = True
    integrity_diffs = []
    for rel_path, pre_meta in pre_integrity.items():
        post_meta = post_integrity.get(rel_path)
        if not post_meta:
            integrity_matches = False
            integrity_diffs.append(f"{rel_path}: missing in post-snapshot")
        elif pre_meta["sha256"] != post_meta["sha256"]:
            integrity_matches = False
            integrity_diffs.append(
                f"{rel_path}: SHA256 mismatch (pre: {pre_meta['sha256'][:8]}, post: {post_meta['sha256'][:8]})"
            )

    # Step 6: Verify results
    em = 1 if result.canonical_answer and result.canonical_answer.strip().lower() == gold_answer.lower() else 0
    all_citations_valid = all(c.valid for c in result.citations) if result.citations else False
    expected_chunks_present = all(c in result.cited_chunk_ids for c in expected_chunks)

    print("\n=== Verification Results ===")
    print(f"Generated Answer:    {result.answer}")
    print(f"Canonical Answer:    {result.canonical_answer}")
    print(f"Exact Match (EM):    {em}")
    print(f"Abstained:           {result.abstained} (Reason: {result.abstention_reason})")
    print(f"Cited Chunk IDs:     {result.cited_chunk_ids}")
    print(f"Expected Chunks:     {expected_chunks} (Present: {expected_chunks_present})")
    print(f"Citations Count:     {len(result.citations)} (All Valid: {all_citations_valid})")
    print(f"Total Query Latency: {result.latency_seconds:.3f}s (Measured: {measured_latency:.3f}s)")
    print(f"Timing Breakdown:    {result.timing}")
    print(f"KB Integrity Intact: {integrity_matches}")
    if integrity_diffs:
        print(f"Integrity Diffs:     {integrity_diffs}")

    # Step 7: Save structured output
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_name": "public_api_smoke_test",
        "question": question,
        "gold_answer": gold_answer,
        "mode": result.mode,
        "kb_id": kb_stats["kb_id"],
        "kb_dir": str(kb_path),
        "generated_answer": result.answer,
        "canonical_answer": result.canonical_answer,
        "exact_match": em,
        "abstained": result.abstained,
        "abstention_reason": result.abstention_reason,
        "cited_chunk_ids": result.cited_chunk_ids,
        "expected_chunks": expected_chunks,
        "expected_chunks_present": expected_chunks_present,
        "citations": [c.model_dump() for c in result.citations],
        "all_citations_valid": all_citations_valid,
        "latency_seconds": result.latency_seconds,
        "measured_latency_seconds": measured_latency,
        "timing_breakdown": result.timing,
        "kb_stats": kb_stats,
        "kb_integrity": {
            "intact": integrity_matches,
            "diffs": integrity_diffs,
            "file_checksums": {k: v["sha256"] for k, v in post_integrity.items()},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved smoke test result to: {output_path}")

    # Assertions for programmatic smoke test verification
    assert integrity_matches, f"Knowledge base files were modified: {integrity_diffs}"
    assert em == 1, f"Canonical answer '{result.canonical_answer}' does not match gold answer '{gold_answer}'"
    assert not result.abstained, "Engine unexpectedly abstained"
    assert all_citations_valid, "One or more citations marked invalid"
    assert expected_chunks_present, f"Missing expected chunks in citations: {result.cited_chunk_ids}"
    assert result.latency_seconds > 0, "Latency was zero"

    print("\n>>> SMOKE TEST PASSED SUCCESSFULLY <<<")


if __name__ == "__main__":
    main()
