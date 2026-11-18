#!/usr/bin/env python3
"""Audit Part 1(f): Unanswerable question audit."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(__file__).resolve().parent
SYNTH_DIR = BASE_DIR / "experiments" / "synth_kg"
OUTPUT_FILE = AUDIT_DIR / "audit_output.txt"


def log(msg: str) -> None:
    print(msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def is_abstained(answer_text: str, abstained_flag: bool = False) -> bool:
    if abstained_flag:
        return True
    norm = normalize_text(answer_text)
    phrases = [
        "does not state",
        "does not contain",
        "does not provide",
        "cannot be determined",
        "cannot determine",
        "no evidence",
        "no information",
        "not mentioned",
        "not state",
        "insufficient evidence",
        "unanswerable",
        "not stated",
    ]
    for p in phrases:
        if p in norm:
            return True
    return False


def run_part1_f() -> None:
    log("================================================================================")
    log("PART 1(f) - UNANSWERABLE QUESTION AUDIT (ALL 15 QUESTIONS)")
    log("================================================================================\n")

    # Load corpus documents
    docs_dir = SYNTH_DIR / "corpus_final" / "docs"
    docs = {}
    for p in sorted(docs_dir.glob("*.txt"), key=lambda x: x.name):
        docs[p.stem] = p.read_text(encoding="utf-8").strip()

    # Load questions
    q_file = SYNTH_DIR / "corpus_final" / "questions.jsonl"
    with open(q_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    unans_questions = [q for q in questions if q.get("group") == "unanswerable"]

    # Load gold KG
    kg_file = SYNTH_DIR / "corpus_final" / "gold_kg.json"
    with open(kg_file, "r", encoding="utf-8") as f:
        gold_kg = json.load(f)

    entities = {e["id"]: e for e in gold_kg["entities"]}
    ent_by_name = {e["canonical_name"].lower(): e for e in gold_kg["entities"]}
    for e in gold_kg["entities"]:
        for a in e.get("aliases", []):
            ent_by_name[a.lower()] = e

    edges = gold_kg.get("edges", gold_kg.get("relationships", []))
    adj_out: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        adj_out.setdefault(edge["source"], []).append(edge)

    # Load run outputs
    modes = ["vector_only", "hybrid", "graph_only", "community_aware"]
    run_subdirs = sorted((SYNTH_DIR / "runs").glob("*"), key=lambda p: p.name)
    mode_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rd in run_subdirs:
        res_file = rd / "results.jsonl"
        if not res_file.exists():
            continue
        with open(res_file, "r", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if not recs:
            continue
        m = recs[0].get("mode")
        if m in modes:
            mode_results[m] = {r["question_id"]: r for r in recs}

    log(f"Auditing {len(unans_questions)} unanswerable questions:\n")

    for idx, q in enumerate(unans_questions, start=1):
        qid = q["id"]
        qtxt = q["question"]

        # Identify subject entity in question
        subject_ent = None
        for name, ent in ent_by_name.items():
            if name in qtxt.lower():
                subject_ent = ent
                break

        # Gold reason
        if subject_ent:
            eid = subject_ent["id"]
            cname = subject_ent["canonical_name"]
            etype = subject_ent["type"]
            out_rels = [e["relation"] for e in adj_out.get(eid, [])]
            reason = f"Entity '{cname}' (Type: {etype}) has outgoing relations: {out_rels or 'None'}. Gold KG contains no edge answering: '{qtxt}'."
        else:
            reason = f"No entity in gold KG matches question subject, or no relation exists in gold KG."

        # Search corpus documents for potential misleading sentences
        misleading_snippets = []
        if subject_ent:
            cname = subject_ent["canonical_name"]
            for did, dtext in docs.items():
                if cname.lower() in dtext.lower():
                    # extract sentences containing cname
                    sentences = re.split(r"(?<=[.!?])\s+", dtext)
                    for s in sentences:
                        if cname.lower() in s.lower():
                            misleading_snippets.append((did, s.strip()))

        log(f"### [{idx}] Question ID: `{qid}`")
        log(f"- **Question**: \"{qtxt}\"")
        log(f"- **Reason Unanswerable in Gold KG**: {reason}")
        if misleading_snippets:
            log(f"- **Corpus Text Mentioning Subject Entity** ({len(misleading_snippets)} sentence(s)):")
            for did, s in misleading_snippets[:3]:
                log(f"  * [{did}]: \"{s}\"")
        else:
            log(f"- **Corpus Text Mentioning Subject Entity**: None found in corpus documents.")

        log("- **Model Predictions & Abstention Status**:")
        for m in modes:
            rec = mode_results[m].get(qid, {})
            ans = rec.get("answer", "")
            raw_abst = rec.get("abstained", False)
            abst = is_abstained(ans, raw_abst)
            trimmed_ans = ans[:250] + ("..." if len(ans) > 250 else "")
            status_str = "ABSTAINED (Correct)" if abst else "FAILED TO ABSTAIN (Incorrect)"
            log(f"  * `{m:15s}` [{status_str}]: \"{trimmed_ans}\"")
        log("")


if __name__ == "__main__":
    run_part1_f()
