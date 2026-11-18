#!/usr/bin/env python3
"""Retrieval-only evaluation through the public API, with cached query plans and NO answer generation.

The engine's query path (HybridGraphRAG.query -> HybridGraphRAGEngine.query) is executed unchanged, except
that two classes are replaced at import level inside the chosen code tree:
  * src.engine.OllamaQueryPlanner  -> returns the plan cached on disk for the question (plan_cache.py)
  * src.engine.OllamaAnswerGenerator -> stub that returns immediately (no LLM call)
In addition the LLM client's `generate` is replaced by a function that raises, so any accidental LLM call
during retrieval evaluation is an error instead of a silent generation.

--tree selects the code tree: the repository root (fixed code) or experiments/synth_kg_fix/head_tree
(a copy of HEAD src/configs/prompts taken from ~/graphrag_backup_20261116_e, verified equal to HEAD).

Usage:
  python3 experiments/synth_kg_fix/retrieval_eval.py --tree . --kb-dir <kb dir> --plans <plans.json> \
      --questions <questions.jsonl> --modes hybrid,graph_only,vector_only --out <results.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--kb-dir", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--modes", default="hybrid,graph_only,vector_only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tree = Path(args.tree).resolve()
    kb_dir = Path(args.kb_dir).resolve()
    plans = json.loads(Path(args.plans).resolve().read_text(encoding="utf-8"))
    questions = [json.loads(l) for l in Path(args.questions).resolve().read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q.get("group") != "unanswerable"]
    out_path = Path(args.out).resolve()

    os.chdir(tree)  # prompt paths in the engine are cwd-relative
    sys.path.insert(0, str(tree))
    sys.dont_write_bytecode = True

    import src.engine as engine_mod
    from hybrid_graphrag import HybridGraphRAG
    from src.core.models import QueryPlan

    assert Path(engine_mod.__file__).resolve().is_relative_to(tree), engine_mod.__file__
    plan_by_question = {}
    for q in questions:
        plan_by_question[q["question"].strip()] = plans[q["id"]]

    class CachedPlanner:
        def __init__(self, *a, **k) -> None:
            pass

        def plan(self, question: str):
            rec = plan_by_question[question.strip()]
            if rec["error"]:
                raise RuntimeError(f"cached planner failure: {rec['error']}")
            return QueryPlan.model_validate(rec["plan"])

    class NoAnswerGenerator:
        def __init__(self, *a, **k) -> None:
            pass

        def generate_structured(self, retrieval_result):
            return SimpleNamespace(
                answer="[retrieval-only: no answer generated]", canonical_answer=None,
                abstained=bool(retrieval_result.abstained), abstention_reason=retrieval_result.abstention_reason,
                cited_chunk_ids=[], citation_references=[], generation_error=False,
            )

    engine_mod.OllamaQueryPlanner = CachedPlanner
    engine_mod.OllamaAnswerGenerator = NoAnswerGenerator

    rag = HybridGraphRAG.from_existing(kb_dir=kb_dir, base_dir=tree)
    eng = rag._engine
    assert Path(eng.kb_paths["kb_root"]).resolve() == kb_dir, (eng.kb_paths["kb_root"], kb_dir)

    def forbidden(*a, **k):
        raise RuntimeError("LLM call attempted during retrieval-only evaluation")

    eng.llm_client.generate = forbidden
    eng.llm_client.generate_json = forbidden

    entities = {e.entity_id: e for e in eng.doc_store.load_canonical_entities()}
    retr = eng.config.retrieval
    print(f"tree={tree} kb={kb_dir} entities={len(entities)} vector_top_k={retr.vector_top_k} "
          f"final_context_chunks={retr.final_context_chunks} person_like={getattr(eng.config.ontology, 'person_like_types', 'n/a (HEAD)')}",
          flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for mode in [m for m in args.modes.split(",") if m]:
            for q in questions:
                rec = {"question_id": q["id"], "group": q["group"], "mode": mode, "error": None,
                       "plan_error": plans[q["id"]]["error"]}
                try:
                    res = rag.query(question=q["question"], mode=mode)
                    rr = res.raw_result
                    rec.update({
                        "intent": getattr(rr.query_plan.intent, "value", str(rr.query_plan.intent)),
                        "seed_specs": [s.model_dump() for s in rr.query_plan.seed_entities],
                        "resolved_seed_ids": list(rr.resolved_seed_entity_ids),
                        "resolved_seeds": [
                            {"id": i, "name": entities[i].name, "type": entities[i].entity_type}
                            for i in rr.resolved_seed_entity_ids if i in entities
                        ],
                        "context_chunk_ids": [ev.chunk_id for ev in rr.evidence_list],
                        "context_doc_ids": [ev.doc_id for ev in rr.evidence_list],
                        "context_routes": ["+".join(r.value for r in ev.routes) for ev in rr.evidence_list],
                        "route_errors": rr.route_diagnostics.get("route_errors", {}),
                    })
                except Exception as exc:  # counted as ERROR, kept separate from failures to retrieve
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec.update({"resolved_seed_ids": [], "resolved_seeds": [], "context_doc_ids": [], "context_chunk_ids": []})
                f.write(json.dumps(rec) + "\n")
            print(f"mode {mode}: done", flush=True)
    rag.close()
    print("RETRIEVAL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
