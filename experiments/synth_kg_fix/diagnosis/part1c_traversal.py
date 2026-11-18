#!/usr/bin/env python3
"""Diagnosis Part 1c: can the graph route REACH the gold documents at all? Uses the engine's own GraphTraverser
(src/graph/traversal.py, unchanged) on the stored KB graphs, with the seeds the engine resolved (recorded in
part1_records_seed*.jsonl) and the cached plan's relationship_path. No LLM, no embeddings.

Variants (analysis only, nothing in src changes):
  T1  engine behaviour: directed, max_hops=3, max_candidates=20 (what GraphEvidenceRetriever constructs)
  T2  same traversal with the 20-path cap removed
  T3  undirected reachability within 3 hops (vertex source chunks + chunks of traversed edges), cap removed
  T4  undirected reachability at any distance (connected component of the seeds)

Reports per group the number of questions whose gold documents are ALL covered by the chunks of the traversal.
Writes part1c_tables.txt.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "experiments" / "synth_kg_fix"
DIAG = FIX / "diagnosis"
sys.dont_write_bytecode = True
sys.path.insert(0, str(FIX))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import fixlib as fl  # noqa: E402

CFG = {
    "seed 7 (kb_dev)": (FIX / "kb_dev/synth_company_kb", FIX / "plans_seed7.json", ROOT / "experiments/synth_kg/corpus_final/questions.jsonl", DIAG / "part1_records_seed7.jsonl"),
    "seed 7 (kb_dev_merge_only)": (FIX / "kb_dev_merge_only/synth_company_kb", FIX / "plans_seed7.json", ROOT / "experiments/synth_kg/corpus_final/questions.jsonl", DIAG / "part1_records_seed7mo.jsonl"),
    "seed 11 (kb_seed11)": (FIX / "kb_seed11/synth_company_kb", FIX / "plans_seed11.json", FIX / "corpus_seed11/questions.jsonl", DIAG / "part1_records_seed11.jsonl"),
}
LINES: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def undirected_chunks(graph, seeds, max_hops=3):
    seen = {}
    chunks = set()
    frontier = []
    for s in seeds:
        try:
            v = graph.vs.find(name=s)
        except ValueError:
            continue
        seen[v.index] = 0
        frontier.append(v.index)
        chunks.update(v.attributes().get("source_chunk_ids") or [])
    for hop in range(max_hops):
        nxt = []
        for vi in frontier:
            for e in graph.incident(vi, mode="all"):
                ed = graph.es[e]
                other = ed.target if ed.source == vi else ed.source
                chunks.update(ed.attributes().get("source_chunk_ids") or [])
                if other not in seen:
                    seen[other] = hop + 1
                    chunks.update(graph.vs[other].attributes().get("source_chunk_ids") or [])
                    nxt.append(other)
        frontier = nxt
    return chunks


def main():
    from hybrid_graphrag import HybridGraphRAG
    from src.graph.traversal import GraphTraverser

    P("part1c: reachability of gold documents by the graph route's traversal (engine GraphTraverser, stored KB graphs)")
    OUTJ = {}
    for label, (kbd, plans_p, q_p, rec_p) in CFG.items():
        plans = json.loads(plans_p.read_text())
        qs = [q for q in fl.read_jsonl(q_p) if q["group"] != "unanswerable"]
        recs = {r["question_id"]: r for r in fl.read_jsonl(rec_p)[1:] if r["mode"] == "hybrid"}
        rag = HybridGraphRAG.from_existing(kb_dir=kbd, base_dir=ROOT)
        graph = rag._engine.get_or_load_graph()
        P("=" * 110)
        P(f"{label}: graph vertices={graph.vcount()} edges={graph.ecount()}")
        res = collections.defaultdict(lambda: collections.Counter())
        capinfo = collections.defaultdict(list)
        detail = []
        hneed = collections.defaultdict(list)
        for q in qs:
            grp = q["group"]
            gold = {fl.norm_doc_id(d) for d in q["gold_supporting_doc_ids"]}
            seeds = [s["id"] for s in recs[q["id"]]["resolved_seeds"]]
            rel_path = plans[q["id"]]["plan"].get("relationship_path") if plans[q["id"]].get("plan") else None
            docs_of = lambda chs: {fl.norm_doc_id(c.split("_")[0]) for c in chs}
            p20 = GraphTraverser(max_hops=3, max_candidates=20).traverse(graph, seeds, rel_path)
            pU = GraphTraverser(max_hops=3, max_candidates=10**6).traverse(graph, seeds, rel_path)
            c20 = docs_of({c for p in p20 for c in p.chunk_ids})
            cU = docs_of({c for p in pU for c in p.chunk_ids})
            cT = docs_of(undirected_chunks(graph, seeds))
            cC = docs_of(undirected_chunks(graph, seeds, max_hops=50))
            hu = next((h for h in range(1, 13) if gold <= docs_of(undirected_chunks(graph, seeds, max_hops=h))), None)
            hd = next((h for h in range(1, 13) if gold <= docs_of({c for p in GraphTraverser(max_hops=h, max_candidates=10**6).traverse(graph, seeds, rel_path) for c in p.chunk_ids})), None)
            hneed[grp].append((hu, hd))
            gl = {fl.norm_doc_id(e["doc_id"]) for e in recs[q["id"]].get("graph", [])}
            res[grp]["N"] += 1
            res[grp]["T1 capped-20 directed"] += gold <= c20
            res[grp]["T2 uncapped directed"] += gold <= cU
            res[grp]["T3 undirected"] += gold <= cT
            res[grp]["T4 undirected, any distance"] += gold <= cC
            res[grp]["recorded graph list (<=20 chunks)"] += gold <= gl
            res[grp]["consistency violation (graph list covers gold but T1 does not)"] += (gold <= gl) and not (gold <= c20)
            capinfo[grp].append((len(p20), len(pU), max((p.hop_count for p in p20), default=0), max((p.hop_count for p in pU), default=0)))
            if grp == "hop3":
                detail.append((q["id"], sorted(gold), seeds, rel_path, len(p20), len(pU),
                               sorted(gold - c20), sorted(gold - cU), sorted(gold - cT)))
        P("| group | N | recorded graph list covers all gold | T1 capped-20 directed | T2 uncapped directed | T3 undirected 3-hop | T4 undirected any distance | graph list covers gold but T1 does not (seed-direct/other chunks) |")
        P("|---|---|---|---|---|---|---|---|")
        for grp in fl.GROUPS:
            r = res[grp]
            P(f"| {grp} | {r['N']} | {r['recorded graph list (<=20 chunks)']} | {r['T1 capped-20 directed']} | {r['T2 uncapped directed']} | {r['T3 undirected']} | {r['T4 undirected, any distance']} | {r['consistency violation (graph list covers gold but T1 does not)']} |")
        for grp in ("hop2", "hop3"):
            ci = capinfo[grp]
            P(f"  {grp}: questions where the capped traversal returned exactly 20 paths: {sum(1 for a in ci if a[0]==20)}/{len(ci)}; "
              f"median uncapped path count {sorted(a[1] for a in ci)[len(ci)//2]}; max hop reached in the capped traversal: {dict(collections.Counter(a[2] for a in ci))}; "
              f"max hop reached uncapped: {dict(collections.Counter(a[3] for a in ci))}")
        for grp in ("hop2", "hop3"):
            P(f"  {grp}: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {dict(sorted(collections.Counter(h[0] for h in hneed[grp]).items(), key=lambda kv: (kv[0] is None, kv[0])))}; "
              f"directed (engine traversal, cap removed): {dict(sorted(collections.Counter(h[1] for h in hneed[grp]).items(), key=lambda kv: (kv[0] is None, kv[0])))}")
        P(f"  hop3 per question (id, gold docs, n resolved seeds, relationship_path, paths capped, paths uncapped, gold NOT covered by T1 / T2 / T3):")
        for d in detail:
            P(f"    {d[0]} gold={d[1]} seeds={len(d[2])} rel_path={d[3]} paths={d[4]}/{d[5]} missing T1={d[6]} T2={d[7]} T3={d[8]}")
        OUTJ[label] = {g: dict(res[g]) for g in fl.GROUPS}
        OUTJ[label]['hneed'] = {g: [list(x) for x in hneed[g]] for g in ('hop2', 'hop3')}
        OUTJ[label]['capped_20_paths'] = {g: sum(1 for a in capinfo[g] if a[0] == 20) for g in ('hop2', 'hop3')}
        rag.close()
    (DIAG / "part1c_summary.json").write_text(json.dumps(OUTJ, indent=1), encoding="utf-8")
    (DIAG / "part1c_tables.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
