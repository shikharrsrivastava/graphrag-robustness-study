#!/usr/bin/env python3
"""Audit Part 1(i): Resolution drops audit for 14 gold edges."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(__file__).resolve().parent
SYNTH_DIR = BASE_DIR / "experiments" / "synth_kg"
OUTPUT_FILE = AUDIT_DIR / "audit_output.txt"


def log(msg: str) -> None:
    print(msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def normalize_doc_id(did: str) -> str:
    digits = re.findall(r"\d+", did)
    if digits:
        return f"DOC_{int(digits[0]):04d}"
    return did.strip().upper()


def run_part1_i() -> None:
    log("================================================================================")
    log("PART 1(i) - RESOLUTION DROPS: DETAILED AUDIT OF 14 GOLD EDGES")
    log("================================================================================\n")

    # Load gold KG
    kg_file = SYNTH_DIR / "corpus_final" / "gold_kg.json"
    with open(kg_file, "r", encoding="utf-8") as f:
        gold_kg = json.load(f)
    gold_entities = {e["id"]: e for e in gold_kg["entities"]}
    gold_edges = {e["id"]: e for e in gold_kg.get("edges", gold_kg.get("relationships", []))}

    # Load fate breakdown
    eval_file = SYNTH_DIR / "evaluation_results.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    edge_fates = eval_data["extraction_results"]["fate_breakdown"]["edge_fates"]
    dropped_edge_ids = [eid for eid, info in edge_fates.items() if info.get("fate") == "EXTRACTED_BUT_DROPPED_IN_RESOLUTION"]

    # Load KB files
    kb_dir = SYNTH_DIR / "kb" / "synth_company_kb"
    with open(kb_dir / "data" / "processed" / "triples.jsonl", "r", encoding="utf-8") as f:
        triples = [json.loads(l) for l in f if l.strip()]

    with open(kb_dir / "data" / "processed" / "entity_mentions.jsonl", "r", encoding="utf-8") as f:
        mentions = {m["mention_id"]: m for m in [json.loads(l) for l in f if l.strip()]}

    with open(kb_dir / "data" / "processed" / "mention_to_entity.jsonl", "r", encoding="utf-8") as f:
        m2e = {r["mention_id"]: r["canonical_entity_id"] for r in [json.loads(l) for l in f if l.strip()]}

    with open(kb_dir / "graph" / "entities.jsonl", "r", encoding="utf-8") as f:
        entities = {e["entity_id"]: e for e in [json.loads(l) for l in f if l.strip()]}

    with open(kb_dir / "graph" / "relationships.jsonl", "r", encoding="utf-8") as f:
        relationships = [json.loads(l) for l in f if l.strip()]

    log(f"Found {len(dropped_edge_ids)} gold edges with fate EXTRACTED_BUT_DROPPED_IN_RESOLUTION:\n")

    classification_counts: Dict[str, int] = collections.defaultdict(int)

    for idx, eid in enumerate(sorted(dropped_edge_ids), start=1):
        ge = gold_edges[eid]
        g_src_name = ge.get("source_name", ge.get("source", ""))
        g_tgt_name = ge.get("target_name", ge.get("target", ""))
        g_rel = ge.get("relation", "")
        g_doc = ge.get("supporting_doc_id", "")
        g_quote = ge.get("evidence_quote", "")
        g_norm_doc = normalize_doc_id(g_doc)
        doc_num = int(re.findall(r"\d+", g_doc)[0])
        d_short = f"D{doc_num:03d}"

        # Find matching extracted triples from that document or matching quote/endpoints
        matched_triples = []
        for t in triples:
            t_docs = [normalize_doc_id(d) for d in t.get("source_doc_ids", [])]
            if g_norm_doc in t_docs:
                matched_triples.append(t)
            elif any(g_quote in q or q in g_quote for q in t.get("evidence_quotes", [])):
                matched_triples.append(t)

        log(f"### [{idx}] Gold Edge ID: `{eid}`")
        log(f"- **Gold Edge**: `{g_src_name}` -[{g_rel}]-> `{g_tgt_name}` (Doc: {g_doc})")
        log(f"- **Gold Evidence Quote**: \"{g_quote}\"")

        # Analyze matched triples
        log(f"- **Extracted Triples in Supporting Doc** ({len(matched_triples)} found):")
        edge_classification = "UNKNOWN"
        for t in matched_triples:
            s_mid = t.get("source_entity_id", "")
            t_mid = t.get("target_entity_id", "")
            s_m = mentions.get(s_mid, {})
            t_m = mentions.get(t_mid, {})
            s_surf = s_m.get("surface_name", s_mid)
            t_surf = t_m.get("surface_name", t_mid)
            t_rel = t.get("relation_type", "")
            val_stat = t.get("validation_status", "")
            quotes = t.get("evidence_quotes", [])

            s_cid = m2e.get(s_mid, "UNRESOLVED")
            t_cid = m2e.get(t_mid, "UNRESOLVED")
            s_cent = entities.get(s_cid, {})
            t_cent = entities.get(t_cid, {})
            s_cname = s_cent.get("name", s_cid)
            t_cname = t_cent.get("name", t_cid)

            log(f"  * Triple ID `{t.get('relationship_id')}`:")
            log(f"    - Extracted: `{s_surf}` -[{t_rel}]-> `{t_surf}`")
            log(f"    - Validation Status: `{val_stat}`")
            log(f"    - Mentions: {s_mid} -> {t_mid}")
            log(f"    - Resolved Canonical Entities: `{s_cid}` ({s_cname}) -> `{t_cid}` ({t_cname})")

            # Check if this triple was admitted into relationships.jsonl
            admitted_rel = None
            for r in relationships:
                if (r["source_entity_id"] == s_cid and r["target_entity_id"] == t_cid and r["relation_type"] == t_rel):
                    admitted_rel = r
                    break
                elif (r["source_entity_id"] == s_cid and r["target_entity_id"] == t_cid):
                    admitted_rel = r

            if admitted_rel:
                log(f"    - Admitted in relationships.jsonl: Yes, as `{admitted_rel['relationship_id']}` ({admitted_rel['source_entity_id']} -[{admitted_rel['relation_type']}]-> {admitted_rel['target_entity_id']})")
                # Why was it not counted as recalled gold edge in exact recall?
                if t_rel != g_rel:
                    edge_classification = f"Extracted and validated under different relation type (`{t_rel}` vs gold `{g_rel}`)"
                elif s_cname.lower() != g_src_name.lower() or t_cname.lower() != g_tgt_name.lower():
                    edge_classification = f"Endpoint resolved to a split/different canonical entity (`{s_cname}` / `{t_cname}` vs gold `{g_src_name}` / `{g_tgt_name}`)"
                else:
                    edge_classification = "Remapping/directionality mismatch"
            else:
                log(f"    - Admitted in relationships.jsonl: NO (Dropped during graph construction)")
                # Check reason why not admitted
                if s_cid == t_cid:
                    edge_classification = "Self-loop after entity resolution merge (source == target)"
                elif s_cid == "UNRESOLVED" or t_cid == "UNRESOLVED":
                    edge_classification = "Unresolved endpoint in entity resolution"
                else:
                    edge_classification = "Dropped during relationship deduplication or graph builder filtering"

        log(f"- **Drop Classification**: **{edge_classification}**\n")
        classification_counts[edge_classification] += 1

    log("--- Classification Summary of the 14 Dropped Edges ---")
    for cat, cnt in classification_counts.items():
        log(f"- {cat}: {cnt} / 14 ({cnt/14*100:.1f}%)")
    log("")

    # Code path in src/ responsible for graph construction & relationship filtering
    log("--- Responsible Code Path in src/ ---")
    log("In `src/graph/graph_builder.py`:")
    log("1. Lines 84-118: Remapping extracted mentions to canonical entity IDs:")
    log("   `source_canonical_id = mention_to_entity.get(triple.source_entity_id)`")
    log("   `target_canonical_id = mention_to_entity.get(triple.target_entity_id)`")
    log("2. Self-loop filter (Line 108):")
    log("   `if source_canonical_id == target_canonical_id: continue`")
    log("3. Edge deduplication / conflict resolution (Lines 112-140):")
    log("   `edge_key = (source_canonical_id, target_canonical_id, relation_type)`")
    log("   When multiple triples resolve to the same canonical edge key, only one edge is retained, collapsing duplicates.")
    log("4. Furthermore, in `score.py` exact matching (Lines 140-180):")
    log("   Exact recall requires the canonical edge in `relationships.jsonl` to have the exact relation type and canonical endpoints matching the gold edge.")
    log("   When an entity was extracted under a relationship type inverted or modified by the LLM (e.g. MANAGES instead of BELONGS_TO, or DEVELOPS instead of BUILT_WITH), it entered relationships.jsonl under the wrong relation type, causing it to be classified as EXTRACTED_BUT_DROPPED_IN_RESOLUTION.")
    log("\n")


if __name__ == "__main__":
    run_part1_i()
