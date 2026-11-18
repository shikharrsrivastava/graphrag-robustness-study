#!/usr/bin/env python3
"""Call the engine's query planner once per answerable question and cache the plans on disk.

The seed-7 run records (experiments/synth_kg/runs/*/results.jsonl) do not store query plans,
so every plan is produced here by the planner (LLM call, T=0) and reused for all before/after
retrieval-only runs. That way only the resolver/graph differs between compared runs.

Usage: python3 experiments/synth_kg_fix/plan_cache.py <questions.jsonl> <kb_config_snapshot.yaml> <out.json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from src.core.configuration import EngineConfig  # noqa: E402
from src.extraction.schemas import OllamaLLMClient  # noqa: E402
from src.retrieval.query_planner import OllamaQueryPlanner  # noqa: E402


def main() -> None:
    q_path, snap_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cfg_data = yaml.safe_load(snap_path.read_text(encoding="utf-8"))
    cfg = EngineConfig.model_validate(cfg_data)
    planner = OllamaQueryPlanner(ontology=cfg.ontology, llm_client=OllamaLLMClient(config=cfg.ollama))

    questions = [json.loads(l) for l in q_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q.get("group") != "unanswerable"]

    cache = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        if qid in cache:
            continue
        started = time.perf_counter()
        try:
            plan = planner.plan(q["question"])
            cache[qid] = {"plan": plan.model_dump(mode="json"), "error": None}
        except Exception as exc:  # planner failure is recorded, not hidden
            cache[qid] = {"plan": None, "error": f"{type(exc).__name__}: {exc}"}
        cache[qid]["seconds"] = round(time.perf_counter() - started, 2)
        out_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"[{i}/{len(questions)}] {qid} {cache[qid]['seconds']}s error={cache[qid]['error']}", flush=True)
    print(f"DONE plans={len(cache)} errors={sum(1 for v in cache.values() if v['error'])}", flush=True)


if __name__ == "__main__":
    main()
