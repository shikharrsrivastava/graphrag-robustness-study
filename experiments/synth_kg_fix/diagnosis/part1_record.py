#!/usr/bin/env python3
"""Diagnosis Part 1 (recording step): per question, record the graph-route list, the vector-route list, the full
merged RRF ranking, the final 4-chunk context, and whether ContextBuilder keeps all of them.

Retrieval only. Planner = cached plans, answer generator = stub, LLM client raises (same harness as
experiments/synth_kg_fix/retrieval_eval.py). src/ is NOT modified: the recording is done by wrapping three class
methods at import time inside this process (GraphEvidenceRetriever.retrieve, VectorEvidenceRetriever.retrieve,
RRFReranker.rerank). The wrapped rerank calls the original twice (top_k=None for the full ranking, then the real
top_k), so the returned context is exactly what the engine would return.

Usage: python3 experiments/synth_kg_fix/diagnosis/part1_record.py --seed 7|11
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "experiments" / "synth_kg_fix"
CFG = {
    7: dict(kb=FIX / "kb_dev/synth_company_kb", plans=FIX / "plans_seed7.json", q=ROOT / "experiments/synth_kg/corpus_final/questions.jsonl"),
    "7mo": dict(kb=FIX / "kb_dev_merge_only/synth_company_kb", plans=FIX / "plans_seed7.json", q=ROOT / "experiments/synth_kg/corpus_final/questions.jsonl"),
    11: dict(kb=FIX / "kb_seed11/synth_company_kb", plans=FIX / "plans_seed11.json", q=FIX / "corpus_seed11/questions.jsonl"),
}


def ev_row(ev):
    return {
        "chunk_id": ev.chunk_id, "doc_id": ev.doc_id,
        "graph_rank": ev.graph_rank, "vector_rank": ev.vector_rank, "vector_distance": ev.vector_distance,
        "rrf_score": ev.rrf_score, "routes": [r.value for r in ev.routes],
        "n_edges": len(ev.supporting_edge_ids), "n_paths": len(ev.supporting_graph_paths),
        "provenance": list(ev.graph_provenance_types),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="7, 11 or 7mo (seed 7 KB with merge only, no revalidation)")
    args = ap.parse_args()
    args.seed = int(args.seed) if args.seed.isdigit() else args.seed
    c = CFG[args.seed]
    plans = json.loads(c["plans"].read_text(encoding="utf-8"))
    questions = [json.loads(l) for l in c["q"].read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q.get("group") != "unanswerable"]
    out_path = FIX / "diagnosis" / f"part1_records_seed{args.seed}.jsonl"

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True
    import src.engine as engine_mod
    from hybrid_graphrag import HybridGraphRAG
    from src.core.models import QueryPlan
    from src.retrieval.context_builder import ContextBuilder
    from src.retrieval.graph_retriever import GraphEvidenceRetriever
    from src.retrieval.reranker import RRFReranker
    from src.retrieval.vector_retriever import VectorEvidenceRetriever

    plan_by_q = {q["question"].strip(): plans[q["id"]] for q in questions}

    class CachedPlanner:
        def __init__(self, *a, **k):
            pass

        def plan(self, question):
            rec = plan_by_q[question.strip()]
            if rec["error"]:
                raise RuntimeError(f"cached planner failure: {rec['error']}")
            return QueryPlan.model_validate(rec["plan"])

    class NoAnswerGenerator:
        def __init__(self, *a, **k):
            pass

        def generate_structured(self, rr):
            return SimpleNamespace(answer="[retrieval-only]", canonical_answer=None, abstained=bool(rr.abstained),
                                   abstention_reason=rr.abstention_reason, cited_chunk_ids=[], citation_references=[],
                                   generation_error=False)

    engine_mod.OllamaQueryPlanner = CachedPlanner
    engine_mod.OllamaAnswerGenerator = NoAnswerGenerator

    REC: dict = {}
    orig_graph, orig_vec, orig_rerank = GraphEvidenceRetriever.retrieve, VectorEvidenceRetriever.retrieve, RRFReranker.rerank

    def g_wrap(self, *a, **k):
        out = orig_graph(self, *a, **k)
        REC["graph"] = [ev_row(e) for e in out]
        REC["graph_details"] = dict(getattr(self, "last_diagnostics", {}))
        return out

    def v_wrap(self, *a, **k):
        out = orig_vec(self, *a, **k)
        REC["vector"] = [ev_row(e) for e in out]
        return out

    def r_wrap(self, evidence_list, top_k=8):
        full = orig_rerank(self, list(evidence_list), top_k=None)
        REC["merged_full"] = [ev_row(e) for e in full]
        REC["n_merge_inputs"] = len(evidence_list)
        return orig_rerank(self, list(evidence_list), top_k=top_k)

    GraphEvidenceRetriever.retrieve = g_wrap
    VectorEvidenceRetriever.retrieve = v_wrap
    RRFReranker.rerank = r_wrap

    rag = HybridGraphRAG.from_existing(kb_dir=c["kb"], base_dir=ROOT)
    eng = rag._engine
    assert Path(eng.kb_paths["kb_root"]).resolve() == c["kb"].resolve()

    def forbidden(*a, **k):
        raise RuntimeError("LLM call attempted")

    eng.llm_client.generate = forbidden
    eng.llm_client.generate_json = forbidden
    cb = ContextBuilder(max_chars=eng.config.retrieval.max_context_chars, graph_context_mode=eng.config.retrieval.graph_context_mode)
    rc = eng.config.retrieval
    meta = dict(vector_top_k=rc.vector_top_k, final_context_chunks=rc.final_context_chunks,
                max_graph_candidates=rc.max_graph_candidates, rrf_k=rc.rrf_k, max_context_chars=rc.max_context_chars,
                graph_context_mode=str(rc.graph_context_mode))
    entities = {e.entity_id: e for e in eng.doc_store.load_canonical_entities()}
    n_done = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta, "seed": args.seed}) + "\n")
        for q in questions:
            for mode in ("hybrid", "graph_only", "vector_only"):
                REC.clear()
                rec = {"question_id": q["id"], "group": q["group"], "mode": mode, "gold": q["gold_supporting_doc_ids"], "error": None}
                try:
                    res = rag.query(question=q["question"], mode=mode)
                    rr = res.raw_result
                    rec["intent"] = getattr(rr.query_plan.intent, "value", str(rr.query_plan.intent))
                    rec["seed_specs"] = [s.model_dump() for s in rr.query_plan.seed_entities]
                    rec["resolved_seeds"] = [{"id": i, "name": entities[i].name, "type": entities[i].entity_type}
                                             for i in rr.resolved_seed_entity_ids if i in entities]
                    rec["relationship_path"] = list(rr.query_plan.relationship_path or [])
                    rec["context"] = [ev_row(e) for e in rr.evidence_list]
                    text = cb.build_context(rr)
                    rec["context_builder_kept"] = [e.chunk_id for e in rr.evidence_list if f"[Chunk: {e.chunk_id} |" in text]
                    rec["context_chars"] = len(text)
                    for key in ("graph", "vector", "merged_full", "n_merge_inputs", "graph_details"):
                        if key in REC:
                            rec[key] = REC[key]
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(rec, default=str) + "\n")
            n_done += 1
            print(f"{q['id']} {q['group']} done ({n_done}/{len(questions)})", flush=True)
    rag.close()
    print("PART1 RECORD DONE", flush=True)


if __name__ == "__main__":
    main()
