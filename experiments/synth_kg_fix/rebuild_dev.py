#!/usr/bin/env python3
"""Rebuild a KB from STORED mentions and triples (no entity or relationship re-extraction).

Reads (read-only) documents, pages, chunks, entity_mentions and triples from a source KB and runs the
current resolution (+ optional type reconciliation / re-validation), embedding, graph build, validation,
Leiden communities and community reports into a NEW KB directory. Stages 5-9 below are copied from
src/engine.py HybridGraphRAGEngine.index_directory (stages 1-4, loading/chunking/extraction, are replaced by
loading the stored files).

LLM use: the resolver's verification of ambiguous merges and the community summarizer (both part of the
ingestion path). No extraction, no answer generation.

Usage:
  python3 experiments/synth_kg_fix/rebuild_dev.py --out-root experiments/synth_kg_fix/kb_dev \
      [--merge on|off|config] [--revalidate on|off|config] [--initial on|off|config] [--skip-communities]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from hybrid_graphrag import HybridGraphRAG  # noqa: E402
from src.communities.community_summarizer import CommunitySummarizer  # noqa: E402
from src.core.configuration import deep_merge_dicts  # noqa: E402
from src.core.models import Chunk, Document, EntityMention, Page, Relationship  # noqa: E402
from src.graph.clustering import LeidenCommunityDetector  # noqa: E402
from src.graph.graph_builder import IGraphBuilder  # noqa: E402
from src.graph.graph_validator import GraphValidator  # noqa: E402
from src.resolution.entity_resolver import ConservativeEntityResolver  # noqa: E402

SRC_KB = BASE / "experiments/synth_kg/kb/synth_company_kb"
KB_ID = "synth_company_kb"


def benchmark_config(out_root: Path, overrides: dict) -> dict:
    """Same construction as experiments/synth_kg/run_overnight.py:get_benchmark_config (lines 97-114),
    with storage_root pointed at the new KB root and optional resolution-flag overrides."""
    base_cfg = yaml.safe_load((BASE / "configs/default.yaml").read_text(encoding="utf-8"))
    domain_cfg = yaml.safe_load((BASE / "configs/domains/company.yaml").read_text(encoding="utf-8"))
    cfg = deep_merge_dicts(base_cfg, domain_cfg)
    cfg["storage_root"] = str(out_root)
    cfg["kb_id"] = KB_ID
    cfg.setdefault("ollama", {})["temperature"] = 0.0
    cfg.setdefault("retrieval", {})["vector_top_k"] = 4
    cfg.setdefault("retrieval", {})["final_context_chunks"] = 4
    cfg.setdefault("retrieval", {})["comparative_reasoning_mode"] = "off"
    cfg.setdefault("retrieval", {})["graph_context_mode"] = False
    cfg.setdefault("prompt_versions", {})["answer_generation"] = "v1.1"
    for key, val in overrides.items():
        if val is not None:
            cfg.setdefault("resolution", {})[key] = val
    return cfg


def load_models(path: Path, cls):
    return [cls.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--src-kb", default=str(SRC_KB))
    for flag in ("merge", "revalidate", "initial"):
        ap.add_argument(f"--{flag}", choices=["on", "off", "config"], default="config")
    ap.add_argument("--skip-communities", action="store_true")
    args = ap.parse_args()

    tri = {"on": True, "off": False, "config": None}
    overrides = {
        "merge_exact_name_across_types": tri[args.merge],
        "revalidate_relationships_after_resolution": tri[args.revalidate],
        "enable_initial_surname_person_merge": tri[args.initial],
    }
    out_root = Path(args.out_root).resolve()
    src = Path(args.src_kb)
    if (out_root / KB_ID).exists():
        raise SystemExit(f"Refusing to overwrite existing KB directory: {out_root / KB_ID}")

    cfg = benchmark_config(out_root, overrides)
    rag = HybridGraphRAG.from_config(config=cfg, base_dir=BASE)
    eng = rag._engine
    print("resolution config:", eng.config.resolution.model_dump(), flush=True)
    print("person_like_types:", eng.config.ontology.person_like_types, flush=True)

    proc = src / "data" / "processed"
    documents = load_models(proc / "documents.jsonl", Document)
    pages = load_models(proc / "pages.jsonl", Page)
    chunks = load_models(proc / "chunks.jsonl", Chunk)
    all_mentions = load_models(proc / "entity_mentions.jsonl", EntityMention)
    all_triples = load_models(proc / "triples.jsonl", Relationship)
    print(f"loaded stored: documents={len(documents)} pages={len(pages)} chunks={len(chunks)} "
          f"mentions={len(all_mentions)} triples={len(all_triples)} "
          f"statuses={dict(Counter(t.validation_status.value for t in all_triples))}", flush=True)

    eng.vector_store.clear()
    eng.doc_store.save_documents(documents)
    eng.doc_store.save_pages(pages)
    eng.doc_store.save_chunks(chunks)
    eng.doc_store.save_entity_mentions(all_mentions)
    eng.doc_store.save_triples(all_triples)

    stage_seconds = {}
    llm_before = dict(eng.llm_client.stats) if hasattr(eng.llm_client, "stats") else {}

    # ---- Stage 5 (copied from src/engine.py index_directory)
    t0 = time.perf_counter()
    resolver = ConservativeEntityResolver(config=eng.config, embedding_provider=eng.embedding_service, llm_client=eng.llm_client)
    canonical_entities, mention_to_entity, graph_relationships = resolver.resolve_and_remap(all_mentions, all_triples)
    eng.doc_store.save_canonical_entities(canonical_entities)
    eng.doc_store.save_mention_to_entity(mention_to_entity)
    eng.doc_store.save_relationships(graph_relationships)
    eng._persist_resolution_outputs(resolver)
    stage_seconds["entity_resolution"] = time.perf_counter() - t0
    print(f"resolution: canonical_entities={len(canonical_entities)} graph_relationships={len(graph_relationships)} "
          f"({stage_seconds['entity_resolution']:.1f}s)", flush=True)

    # ---- Stage 6
    t0 = time.perf_counter()
    if canonical_entities:
        entity_texts = [f"{e.name} ({e.entity_type}): {e.description}" for e in canonical_entities]
        eng.vector_store.upsert_entities(canonical_entities, eng.embedding_service.embed_batch(entity_texts))
    if chunks:
        eng.vector_store.upsert_chunks(chunks, eng.embedding_service.embed_batch([c.text for c in chunks]))
    stage_seconds["embedding_and_vector_indexing"] = time.perf_counter() - t0

    # ---- Stage 7
    graph = IGraphBuilder.build_graph(canonical_entities, graph_relationships)
    eng._graph = graph
    manifest = GraphValidator.validate(
        graph=graph, entities=canonical_entities, relationships=graph_relationships,
        valid_chunk_ids={c.chunk_id for c in chunks}, ontology=eng.config.ontology,
        manifest_path=eng.kb_paths["kb_root"] / "manifest.json", kb_id=eng.kb_id,
        prompt_versions=eng.config.prompt_versions.model_dump(),
    )

    # ---- Stages 8-9
    communities, reports = [], []
    if not args.skip_communities:
        t0 = time.perf_counter()
        communities = LeidenCommunityDetector(config=eng.config.leiden).detect_communities(graph, kb_id=eng.kb_id)
        eng.doc_store.save_communities(communities)
        summarizer = CommunitySummarizer(
            llm_client=eng.llm_client, embedding_provider=eng.embedding_service, vector_store=eng.vector_store,
            doc_store=eng.doc_store, model_tag=eng.config.ollama.llm_model,
        )
        reports = summarizer.summarize_communities(communities, graph, doc_store=eng.doc_store)
        stage_seconds["communities_and_summaries"] = time.perf_counter() - t0

    llm_after = dict(eng.llm_client.stats) if hasattr(eng.llm_client, "stats") else {}
    stats = {
        "source_kb": str(src),
        "path_used": "stored mentions + stored triples (no re-extraction)",
        "resolution_flags": eng.config.resolution.model_dump(),
        "canonical_entities": len(canonical_entities),
        "graph_relationships": len(graph_relationships),
        "graph_vertices": graph.vcount(),
        "graph_edges": graph.ecount(),
        "relation_type_distribution": manifest["relation_type_distribution"],
        "communities": len(communities),
        "community_reports": len(reports),
        "skip_communities": args.skip_communities,
        "stage_seconds": stage_seconds,
        "llm_requests": llm_after.get("requests", 0) - llm_before.get("requests", 0),
        "resolution_report": resolver.last_resolution_report,
    }
    (eng.kb_paths["kb_root"] / "rebuild_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "resolution_report"}, indent=2, default=str), flush=True)
    rag.close()
    print("REBUILD DONE", flush=True)


if __name__ == "__main__":
    main()
