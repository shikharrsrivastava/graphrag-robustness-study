#!/usr/bin/env python3
"""Phase 2c: dev test of the multi-relation prompt (v1.1) against v1.0 on seed 7.

Only RELATIONSHIP extraction is re-run, on the STORED entity mentions of selected documents, with the same model and
parameters as the original ingestion (KB config snapshot: gemma3:4b, T=0). No entity re-extraction, no generation.

Document selection (fixed before running):
  T1  documents whose gold KG states more than one relation for one entity pair (recount in step0_reverify.py)
  T2  supporting documents of the gold edges that experiments/synth_kg/score.py labels
      EXTRACTED_BUT_DROPPED_IN_RESOLUTION on the stored KB ("the 14 gold-edge documents")
  CONTROL  6 documents drawn with random.Random(0).sample from the sorted remaining documents
           (documents in neither T1 nor T2; by construction they have one gold relation per entity pair)

Counts per prompt, over all selected documents, on the extractor's own validation status:
  (a) distinct gold edges recovered: a VALIDATED triple whose source/target mentions map to the gold source/target
      (normalized surface name = gold canonical name or alias) with the gold relation type
  (b) VALIDATED triples that match no gold edge (type, endpoints and direction)

Rule fixed before the run: enable v1.1 for seed 11 only if (a) rises by at least 3 AND (b) rises by no more than
the rise in (a); otherwise keep v1.0.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[2]
FIX = BASE / "experiments/synth_kg_fix"
SK = BASE / "experiments/synth_kg"
KB = SK / "kb/synth_company_kb"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(FIX))

import fixlib as fl  # noqa: E402
from src.core.configuration import EngineConfig, deep_merge_dicts  # noqa: E402
from src.core.models import Chunk, EntityMention  # noqa: E402
from src.extraction.relation_extractor import OllamaRelationshipExtractor  # noqa: E402
from src.extraction.schemas import OllamaLLMClient  # noqa: E402

OUT_JSON = FIX / "fix3_dev_test_results.json"
OUT_TXT = FIX / "fix3_dev_test_output.txt"
PROMPTS = {"v1.0": BASE / "prompts/relationship_extraction.txt", "v1.1": BASE / "prompts/relationship_extraction_v1_1.txt"}


def main() -> None:
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    spec = importlib.util.spec_from_file_location("synth_score", SK / "score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)

    gold = json.loads((SK / "corpus_final/gold_kg.json").read_text())
    G_EDGE = {e["id"]: e for e in gold["edges"]}
    entities = fl.read_jsonl(KB / "graph/entities.jsonl")
    rels = fl.read_jsonl(KB / "graph/relationships.jsonl")
    triples = fl.read_jsonl(KB / "data/processed/triples.jsonl")
    mentions_raw = fl.read_jsonl(KB / "data/processed/entity_mentions.jsonl")
    chunks_raw = fl.read_jsonl(KB / "data/processed/chunks.jsonl")

    by_doc_pair = collections.defaultdict(list)
    for ge in gold["edges"]:
        by_doc_pair[(ge["supporting_doc_id"], frozenset([ge["source"], ge["target"]]))].append(ge["id"])
    t1 = sorted({doc for (doc, _), ids in by_doc_pair.items() if len(ids) > 1})
    ext = score.score_extraction(gold, entities, rels, triples, mentions_raw)
    dropped = sorted(e for e, v in ext["fate_breakdown"]["edge_fates"].items() if v["fate"] == "EXTRACTED_BUT_DROPPED_IN_RESOLUTION")
    t2 = sorted({G_EDGE[e]["supporting_doc_id"] for e in dropped})
    all_docs = sorted(p.stem for p in (SK / "corpus_final/docs").glob("DOC_*.txt"))
    remaining = [d for d in all_docs if d not in set(t1) | set(t2)]
    control = sorted(random.Random(0).sample(remaining, 6))
    selected = sorted(set(t1) | set(t2)) + control
    P("=" * 100)
    P("PHASE 2c - FIX 3 DEV TEST (relationship prompt v1.0 vs v1.1), seed 7")
    P(f"T1 two-relation documents ({len(t1)}): {t1}")
    P(f"T2 supporting documents of the {len(dropped)} score.py EXTRACTED_BUT_DROPPED_IN_RESOLUTION gold edges ({len(t2)}): {t2}")
    P(f"T1 union T2: {len(set(t1) | set(t2))} documents; CONTROL (random seed 0, 6 of {len(remaining)} remaining): {control}")
    P(f"selected documents: {len(selected)}")

    snap = yaml.safe_load((KB / "config_snapshot.yaml").read_text())
    base_cfg = yaml.safe_load((BASE / "configs/default.yaml").read_text())
    domain_cfg = yaml.safe_load((BASE / "configs/domains/company.yaml").read_text())
    cfg_data = deep_merge_dicts(base_cfg, domain_cfg)
    cfg_data["ollama"] = snap["ollama"]          # model and parameters of the original ingestion
    cfg_data["validation"] = snap["validation"]
    cfg_data["kb_id"] = "synth_company_kb"
    cfg = EngineConfig.model_validate(cfg_data)
    P(f"ollama config (from the seed-7 KB snapshot): {cfg.ollama.model_dump()}")
    P(f"ontology person_like_types (current company.yaml, same for both prompts): {cfg.ontology.person_like_types}")

    n2g = {}
    for ge in gold["entities"]:
        for n in [ge["canonical_name"]] + ge.get("aliases", []):
            n2g[score.normalize_text(n)] = ge["id"]
    gold_exact = {(e["source"], e["relation"], e["target"]): e["id"] for e in gold["edges"]}

    chunks = {c["doc_id"]: Chunk.model_validate(c) for c in chunks_raw}
    assert len(chunks) == len(chunks_raw), "expected one chunk per document"
    mentions_by_chunk = collections.defaultdict(list)
    for m in mentions_raw:
        mentions_by_chunk[m["source_chunk_id"]].append(EntityMention.model_validate(m))

    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for version, prompt_path in PROMPTS.items():
        extractor = OllamaRelationshipExtractor(
            ontology=cfg.ontology, llm_client=OllamaLLMClient(config=cfg.ollama),
            prompt_template_path=prompt_path, validation_config=cfg.validation,
        )
        assert ("output one object per relation" in extractor.prompt_template) == (version == "v1.1")
        results.setdefault(version, {})
        for doc in selected:
            if doc in results[version]:
                continue
            kb_doc = "D" + doc[-3:]
            chunk = chunks[kb_doc]
            ms = mentions_by_chunk[chunk.chunk_id]
            t0 = time.perf_counter()
            try:
                out = extractor.extract_relationships(chunk, ms)
                err = None
            except Exception as exc:
                out, err = [], f"{type(exc).__name__}: {exc}"
            mid = {m.mention_id: m for m in ms}
            recs = []
            for t in out:
                s, o = mid[t.source_entity_id], mid[t.target_entity_id]
                gs, go = n2g.get(score.normalize_text(s.surface_name)), n2g.get(score.normalize_text(o.surface_name))
                recs.append({
                    "source": s.surface_name, "source_type": s.entity_type, "relation": t.relation_type,
                    "target": o.surface_name, "target_type": o.entity_type, "status": t.validation_status.value,
                    "gold_edge": gold_exact.get((gs, t.relation_type, go)),
                })
            results[version][doc] = {"triples": recs, "error": err, "seconds": round(time.perf_counter() - t0, 1)}
            OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"  {version} {doc}: {len(recs)} triples, error={err}", flush=True)

    def counts(version, docs):
        rec_gold, no_gold, n_valid, n_all, errs = set(), 0, 0, 0, 0
        for d in docs:
            r = results[version][d]
            errs += bool(r["error"])
            for t in r["triples"]:
                n_all += 1
                if t["status"] == "VALIDATED":
                    n_valid += 1
                    if t["gold_edge"]:
                        rec_gold.add(t["gold_edge"])
                    else:
                        no_gold += 1
        return rec_gold, no_gold, n_valid, n_all, errs

    gold_in_docs = lambda docs: [e["id"] for e in gold["edges"] if e["supporting_doc_id"] in set(docs)]
    P("\n| document set | docs | gold edges in docs | prompt | raw triples | VALIDATED | (a) gold edges recovered | (b) VALIDATED matching no gold edge | extraction ERRORs |")
    P("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for name, docs in [("T1 two-relation", t1), ("T1 union T2", sorted(set(t1) | set(t2))), ("CONTROL", control), ("ALL SELECTED", selected)]:
        for v in PROMPTS:
            g, b, nv, na, er = counts(v, docs)
            summary[(name, v)] = (len(g), b, g)
            P(f"| {name} | {len(docs)} | {len(gold_in_docs(docs))} | {v} | {na} | {nv} | {len(g)} | {b} | {er} |")
    a0, b0, g0 = summary[("ALL SELECTED", "v1.0")]
    a1, b1, g1 = summary[("ALL SELECTED", "v1.1")]
    da, db = a1 - a0, b1 - b0
    P(f"\nALL SELECTED: (a) v1.0={a0} v1.1={a1} rise={da}; (b) v1.0={b0} v1.1={b1} rise={db}")
    P(f"gold edges recovered only with v1.1: {sorted(g1 - g0)}")
    P(f"gold edges recovered only with v1.0: {sorted(g0 - g1)}")
    pair_multi = {}
    for v in PROMPTS:
        n = 0
        for d in t1:
            types = collections.defaultdict(set)
            for t in results[v][d]["triples"]:
                types[frozenset([t["source"], t["target"]])].add(t["relation"])
            n += any(len(x) > 1 for x in types.values())
        pair_multi[v] = n
    P(f"T1 documents in which the model emitted more than one relation type for one mention pair: v1.0={pair_multi['v1.0']} v1.1={pair_multi['v1.1']} of {len(t1)}")
    enable = (da >= 3) and (db <= da)
    P(f"RULE: enable v1.1 only if rise in (a) >= 3 AND rise in (b) <= rise in (a)  ->  (a) rise {da} >= 3: {da >= 3}; (b) rise {db} <= {da}: {db <= da}")
    P(f"OUTCOME: {'ENABLE v1.1 in configs/domains/company.yaml for seed 11' if enable else 'KEEP v1.0 (v1.1 not enabled)'}")
    (FIX / "fix3_dev_test_decision.json").write_text(json.dumps({"a_v10": a0, "a_v11": a1, "b_v10": b0, "b_v11": b1, "rise_a": da, "rise_b": db, "enable_v11": enable}, indent=1))
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
