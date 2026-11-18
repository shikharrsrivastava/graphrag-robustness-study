#!/usr/bin/env python3
"""Ingest the seed-11 corpus through the public API (same path as experiments/synth_kg/run_overnight.py stage 1).

Config construction is copied from run_overnight.py:get_benchmark_config (lines 97-114): configs/default.yaml merged
with configs/domains/company.yaml, then the same overrides (T=0, vector_top_k=4, final_context_chunks=4,
comparative_reasoning_mode off, graph_context_mode False, answer_generation v1.1). Single attempt, no retry.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from hybrid_graphrag import HybridGraphRAG  # noqa: E402
from src.core.configuration import deep_merge_dicts  # noqa: E402

FIX = BASE / "experiments/synth_kg_fix"
KB_ROOT = FIX / "kb_seed11"
KB_ID = "synth_company_kb"
DOCS = FIX / "corpus_seed11" / "docs"


def main() -> None:
    if (KB_ROOT / KB_ID).exists():
        raise SystemExit(f"Refusing to run: {KB_ROOT / KB_ID} already exists (seed 11 is ingested once only)")
    base_cfg = yaml.safe_load((BASE / "configs/default.yaml").read_text(encoding="utf-8"))
    domain_cfg = yaml.safe_load((BASE / "configs/domains/company.yaml").read_text(encoding="utf-8"))
    cfg = deep_merge_dicts(base_cfg, domain_cfg)
    cfg["storage_root"] = str(KB_ROOT)
    cfg["kb_id"] = KB_ID
    cfg.setdefault("ollama", {})["temperature"] = 0.0
    cfg.setdefault("retrieval", {})["vector_top_k"] = 4
    cfg.setdefault("retrieval", {})["final_context_chunks"] = 4
    cfg.setdefault("retrieval", {})["comparative_reasoning_mode"] = "off"
    cfg.setdefault("retrieval", {})["graph_context_mode"] = False
    cfg.setdefault("prompt_versions", {})["answer_generation"] = "v1.1"

    start = time.perf_counter()
    rag = HybridGraphRAG.from_config(config=cfg, base_dir=BASE)
    print("resolution:", rag.config.resolution.model_dump(), flush=True)
    print("prompt_versions:", rag.config.prompt_versions.model_dump(), flush=True)
    print("person_like_types:", rag.config.ontology.person_like_types, flush=True)
    ingest_stats = rag.ingest(input_path=DOCS, rebuild=True)
    kb_stats = rag.get_kb_stats()
    wall = time.perf_counter() - start
    rag.close()
    (KB_ROOT / KB_ID / "build_stats.json").write_text(
        json.dumps({"ingest_stats": ingest_stats, "kb_stats": kb_stats, "build_wall_seconds": wall}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in ingest_stats.items() if k != "stage_seconds"}, indent=1, default=str), flush=True)
    print(f"INGEST DONE in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
