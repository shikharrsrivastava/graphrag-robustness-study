#!/usr/bin/env python3
"""Corpus and KG Validator for Synthetic Benchmark.

Validates:
1. Every evidence quote in gold_kg.json occurs verbatim in its supporting document.
2. All entity names are unique and contain no blacklisted real-world names.
3. Every answerable question's path exists in gold_kg.json and resolves to the gold answer.
4. Bridge document assertion: bridge documents for hop2 and hop3 questions do not leak the first entity's name.
5. Document count and chunk size assertions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

from experiments.synth_kg.generate import REAL_WORLD_NAME_BLACKLIST


def validate_corpus(corpus_dir: Path) -> Dict[str, Any]:
    """Validate all artifacts in corpus_dir."""
    kg_path = corpus_dir / "gold_kg.json"
    docs_dir = corpus_dir / "docs"
    questions_path = corpus_dir / "questions.jsonl"

    errors: List[str] = []

    # 1. Check file existence
    if not kg_path.exists():
        errors.append(f"Missing {kg_path}")
        return {"valid": False, "errors": errors}
    if not docs_dir.exists():
        errors.append(f"Missing {docs_dir}")
        return {"valid": False, "errors": errors}
    if not questions_path.exists():
        errors.append(f"Missing {questions_path}")
        return {"valid": False, "errors": errors}

    # 2. Load KG
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    entities = {e["id"]: e for e in kg["entities"]}
    edges = {e["id"]: e for e in kg["edges"]}

    # Check entity names uniqueness and blacklist
    canonical_names: Set[str] = set()
    for eid, ent in entities.items():
        cname = ent["canonical_name"]
        if cname in canonical_names:
            errors.append(f"Duplicate entity canonical name: {cname}")
        canonical_names.add(cname)

        # Check blacklist
        cname_lower = cname.lower()
        for bad in REAL_WORLD_NAME_BLACKLIST:
            if bad in cname_lower.split():
                errors.append(f"Entity name '{cname}' contains blacklisted real-world name '{bad}'")

    # 3. Check Documents & Evidence Quotes
    docs: Dict[str, str] = {}
    for doc_file in docs_dir.glob("*.txt"):
        text = doc_file.read_text(encoding="utf-8").strip()
        docs[doc_file.stem] = text
        if len(text) > 1100:
            errors.append(f"Document {doc_file.name} exceeds 1100 characters ({len(text)} chars)")

    for edge_id, edge in edges.items():
        doc_id = edge["supporting_doc_id"]
        if doc_id not in docs:
            errors.append(f"Edge {edge_id} supporting doc {doc_id} not found in docs")
            continue
        quote = edge["evidence_quote"]
        if not quote:
            errors.append(f"Edge {edge_id} has empty evidence quote")
            continue
        doc_text = docs[doc_id]
        if quote not in doc_text:
            errors.append(f"Edge {edge_id} evidence quote '{quote}' does not appear verbatim in {doc_id}")

    # 4. Check Questions
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Graph adjacency for traversal validation: source_id -> relation -> target_id
    adj: Dict[str, Dict[str, str]] = {}
    for edge in edges.values():
        adj.setdefault(edge["source"], {})[edge["relation"]] = edge["target"]

    for q in questions:
        qid = q["id"]
        group = q["group"]
        hop_count = q["hop_count"]
        gold_ans = q["gold_answer"]
        doc_ids = q["gold_supporting_doc_ids"]
        path_edges = [edges[eid] for eid in q["gold_path"] if eid in edges]

        if group == "unanswerable":
            if gold_ans is not None:
                errors.append(f"Unanswerable question {qid} has non-null gold answer: {gold_ans}")
            continue

        if len(path_edges) != hop_count:
            errors.append(f"Question {qid} hop_count={hop_count} but path has {len(path_edges)} edges")

        # Verify traversal
        curr = path_edges[0]["source"]
        for edge in path_edges:
            if edge["source"] != curr:
                errors.append(f"Question {qid} path disconnect: expected source {curr}, got {edge['source']}")
            curr = edge["target"]

        final_ent = entities.get(curr)
        if not final_ent or final_ent["canonical_name"] != gold_ans:
            errors.append(f"Question {qid} path resolves to {final_ent.get('canonical_name') if final_ent else None}, expected {gold_ans}")

        # Bridge document leakage check
        if hop_count >= 2:
            start_ent = entities[path_edges[0]["source"]]
            start_names = [start_ent["canonical_name"].lower()] + [a.lower() for a in start_ent["aliases"]]
            # Check bridge docs
            for b_doc_id in doc_ids[1:]:
                b_text = docs.get(b_doc_id, "").lower()
                for name in start_names:
                    if name in b_text:
                        errors.append(f"Question {qid} leaks start entity '{name}' into bridge document {b_doc_id}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "num_entities": len(entities),
        "num_edges": len(edges),
        "num_documents": len(docs),
        "num_questions": len(questions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus and KG Validator")
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/synth_kg/corpus_S"), help="Corpus directory to validate")
    args = parser.parse_args()

    res = validate_corpus(args.corpus_dir)
    print(f"Validation for {args.corpus_dir}:")
    print(f"  Valid:     {res['valid']}")
    print(f"  Entities:  {res.get('num_entities')}")
    print(f"  Edges:     {res.get('num_edges')}")
    print(f"  Documents: {res.get('num_documents')}")
    print(f"  Questions: {res.get('num_questions')}")

    if not res["valid"]:
        print(f"\nErrors found ({len(res['errors'])}):")
        for err in res["errors"][:20]:
            print(f"  - {err}")
        if len(res["errors"]) > 20:
            print(f"  ... and {len(res['errors']) - 20} more errors.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED successfully.")


if __name__ == "__main__":
    main()
