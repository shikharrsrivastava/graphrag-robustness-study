#!/usr/bin/env python3
"""Demo of the Hybrid GraphRAG engine on the seed-7 fixed knowledge base.

Default mode: print precomputed results from experiments/demo/precomputed.json (works offline).
--live:       run the 7 selected questions through the real engine (needs Ollama with gemma3:4b + embeddinggemma).
--live --record: same, and (re)write precomputed.json.

Question selection is a fixed rule applied to experiments/synth_kg_fix/retrieval_seed7_after.jsonl (see RULE).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
KB_DIR = REPO / "experiments/synth_kg_fix/kb_dev/synth_company_kb"
QUESTIONS = REPO / "experiments/synth_kg/corpus_S/questions.jsonl"
RECORDED = REPO / "experiments/synth_kg_fix/retrieval_seed7_after.jsonl"
PRECOMPUTED = DEMO / "precomputed.json"
MODES = ["vector_only", "hybrid"]

RULE = """QUESTION SELECTION RULE (fixed, applied to retrieval_seed7_after.jsonl, question-id order):
  1. first 2 hop1 questions
  2. first 2 hop2 questions where hybrid retrieved ALL gold docs and vector_only did not
  3. first 1 hop2 question where BOTH modes failed to retrieve all gold docs
  4. first 1 hop3 question
  5. first 1 unanswerable question (the recorded file has no unanswerable rows; first by id in questions.jsonl)
"gold retrieved" = every gold supporting doc id is among the 4 context docs."""


def doc_num(x: str) -> int:
    return int(re.sub(r"\D", "", x))


def norm(x: str) -> str:
    return f"D{doc_num(x):03d}"


def load_questions() -> dict:
    return {q["id"]: q for q in (json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip())}


def select_questions() -> list[dict]:
    Q = load_questions()
    rec = [json.loads(l) for l in RECORDED.read_text().splitlines() if l.strip()]
    got = {}
    for r in rec:
        if r["mode"] in MODES:
            got[(r["mode"], r["question_id"])] = {norm(d) for d in r["context_doc_ids"]}

    def ok(mode, qid):
        gold = {norm(g) for g in Q[qid]["gold_supporting_doc_ids"]}
        return gold.issubset(got[(mode, qid)])

    ids = sorted(Q)
    by_group = lambda g: [i for i in ids if Q[i]["group"] == g]
    picked: list[tuple[str, str]] = []
    picked += [(i, "hop1") for i in by_group("hop1")[:2]]
    hop2 = by_group("hop2")
    picked += [(i, "hop2: hybrid retrieved all gold, vector_only did not")
               for i in [i for i in hop2 if ok("hybrid", i) and not ok("vector_only", i)][:2]]
    picked += [(i, "hop2: both modes failed to retrieve all gold")
               for i in [i for i in hop2 if not ok("hybrid", i) and not ok("vector_only", i)][:1]]
    picked += [(i, "hop3") for i in by_group("hop3")[:1]]
    picked += [(i, "unanswerable") for i in by_group("unanswerable")[:1]]
    return [dict(Q[i], why_selected=why) for i, why in picked]


def is_correct(q: dict, res: dict) -> bool:
    """Strict 'contains' metric (the project's, which understates). Unanswerable: correct = abstained."""
    if q["group"] == "unanswerable":
        return bool(res["abstained"])
    if res["abstained"]:
        return False
    text = res["answer"].lower()
    return any(a.lower() in text for a in [q["gold_answer"], *q["gold_answer_aliases"]])


def run_live(questions: list[dict]) -> dict:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.dont_write_bytecode = True
    from hybrid_graphrag import HybridGraphRAG  # public API

    rag = HybridGraphRAG.from_existing(kb_dir=KB_DIR, base_dir=REPO)
    eng = rag._engine
    entities = {e.entity_id: e for e in eng.doc_store.load_canonical_entities()}
    out = {"meta": {"kb": str(KB_DIR.relative_to(REPO)), "rule": RULE,
                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "answer_prompt": eng.config.prompt_versions.answer_generation
                    if hasattr(eng.config, "prompt_versions") else "n/a",
                    "planner": "live OllamaQueryPlanner (not the cached plans used in retrieval_eval.py)"},
           "questions": []}
    for q in questions:
        entry = {"id": q["id"], "group": q["group"], "why_selected": q["why_selected"], "question": q["question"],
                 "gold_answer": q["gold_answer"], "gold_doc_ids": [norm(g) for g in q["gold_supporting_doc_ids"]],
                 "modes": {}}
        for mode in MODES:
            t0 = time.time()
            try:
                r = rag.query(question=q["question"], mode=mode)
                rr = r.raw_result
                cited_docs = []
                for c in r.citations:
                    if c.doc_id and c.doc_id not in cited_docs:
                        cited_docs.append(c.doc_id)
                res = {
                    "error": None, "answer": r.answer, "abstained": r.abstained,
                    "abstention_reason": r.abstention_reason,
                    "cited_chunk_ids": list(r.cited_chunk_ids), "cited_doc_ids": [norm(d) for d in cited_docs],
                    "context_chunk_ids": [e.chunk_id for e in rr.evidence_list],
                    "context_doc_ids": [norm(e.doc_id) for e in rr.evidence_list],
                    "context_routes": ["+".join(x.value for x in e.routes) for e in rr.evidence_list],
                    "seed_entities": [f"{entities[i].name} ({entities[i].entity_type})"
                                      for i in rr.resolved_seed_entity_ids if i in entities],
                    "graph_paths": [{"doc": norm(e.doc_id), "paths": e.supporting_graph_paths[:3]}
                                    for e in rr.evidence_list if e.supporting_graph_paths],
                    "generation_error": r.generation_error,
                }
            except Exception as exc:  # counted separately, never hidden
                res = {"error": f"{type(exc).__name__}: {exc}", "answer": "", "abstained": False,
                       "cited_chunk_ids": [], "cited_doc_ids": [], "context_doc_ids": [], "context_chunk_ids": [],
                       "context_routes": [], "seed_entities": [], "graph_paths": [], "generation_error": False}
            res["wall_seconds"] = round(time.time() - t0, 1)
            res["correct_strict"] = is_correct(q, res)
            res["gold_retrieved"] = set(entry["gold_doc_ids"]).issubset(set(res["context_doc_ids"]))
            entry["modes"][mode] = res
            print(f"  {q['id']} {mode}: {res['wall_seconds']}s", flush=True)
        out["questions"].append(entry)
    rag.close()
    return out


def show(data: dict) -> None:
    print("=" * 100)
    print(data["meta"]["rule"] if "rule" in data["meta"] else RULE)
    print(f"KB: {data['meta']['kb']}   recorded: {data['meta']['recorded_at']}   "
          f"answer prompt: {data['meta']['answer_prompt']}")
    print("Correctness uses the strict 'contains gold answer/alias' metric (understates); unanswerable: correct = abstain.")
    print("=" * 100)
    both_wrong = []
    for e in data["questions"]:
        print(f"\n### {e['id']} [{e['group']}] selected as: {e['why_selected']}")
        print(f"Question:  {e['question']}")
        print(f"Gold answer: {e['gold_answer']}    Gold supporting docs: {', '.join(e['gold_doc_ids']) or '(none)'}")
        for mode in MODES:
            m = e["modes"][mode]
            print(f"  -- {mode}  ({m['wall_seconds']} s)  strict-correct={m['correct_strict']}  "
                  f"gold-docs-retrieved={m['gold_retrieved'] if e['gold_doc_ids'] else 'n/a'}")
            if m["error"]:
                print(f"     ERROR: {m['error']}")
            print(f"     answer: {m['answer'].strip()!r}" + ("  [ABSTAINED]" if m["abstained"] else ""))
            print(f"     cited chunk ids: {m['cited_chunk_ids']}   cited docs: {m['cited_doc_ids']}")
            print(f"     retrieved context docs: {m['context_doc_ids']}")
            if mode == "hybrid":
                print(f"     seed entities: {m['seed_entities']}")
                print(f"     routes per context chunk: {m['context_routes']}")
                for gp in m["graph_paths"]:
                    for p in gp["paths"]:
                        print(f"     graph path ({gp['doc']}): {p}")
                if not m["graph_paths"]:
                    print("     graph path: (none attached to the context chunks)")
        if not e["modes"]["hybrid"]["correct_strict"] and not e["modes"]["vector_only"]["correct_strict"]:
            both_wrong.append(e["id"])
    print("\n" + "=" * 100)
    print("Questions where BOTH vector_only and hybrid answers are wrong (strict metric), kept in the demo:",
          ", ".join(both_wrong) or "none")
    print("Wall time per question (vector_only + hybrid): " +
          ", ".join(f"{e['id']}={e['modes']['vector_only']['wall_seconds'] + e['modes']['hybrid']['wall_seconds']:.0f}s"
                    for e in data["questions"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="run the engine for real (needs Ollama)")
    ap.add_argument("--record", action="store_true", help="with --live: write precomputed.json")
    args = ap.parse_args()
    qs = select_questions()
    if args.live:
        print(RULE, "\nSelected:", [q["id"] for q in qs], flush=True)
        data = run_live(qs)
        if args.record:
            PRECOMPUTED.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"wrote {PRECOMPUTED}")
    else:
        if not PRECOMPUTED.exists():
            sys.exit("precomputed.json missing; run with --live --record first")
        data = json.loads(PRECOMPUTED.read_text(encoding="utf-8"))
        assert [q["id"] for q in qs] == [e["id"] for e in data["questions"]], "selection rule no longer matches precomputed.json"
    show(data)


if __name__ == "__main__":
    main()
