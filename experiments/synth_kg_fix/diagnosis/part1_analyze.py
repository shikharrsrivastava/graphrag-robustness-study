#!/usr/bin/env python3
"""Diagnosis Part 1 (analysis): reads part1_records_seed{7,11}.jsonl (written by part1_record.py) and the earlier
retrieval_seed7_after.jsonl / retrieval_seed11_all.jsonl, writes part1_tables.txt, part1_hop3_detail_seed{7,11}.txt and
part1_summary.json. No engine calls.

Counterfactual re-ranking uses ONLY the recorded candidate lists. Labelled: exploratory, not a fix, seed already used.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
DIAG = Path(__file__).resolve().parent
FIX = DIAG.parent
sys.path.insert(0, str(FIX))
import fixlib as fl  # noqa: E402

EARLIER = {7: FIX / "retrieval_seed7_after.jsonl", "7mo": FIX / "retrieval_seed7_merge_only.jsonl", 11: FIX / "retrieval_seed11_all.jsonl"}
GROUPS = fl.GROUPS
OUT_LINES: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT_LINES.append(s)


def docs_of(rows):
    return [fl.norm_doc_id(r["doc_id"]) for r in rows]


def rank_of(rows, doc, key=None):
    for i, r in enumerate(rows, 1):
        if fl.norm_doc_id(r["doc_id"]) == doc:
            return i
    return None


def rrf(graph, vector, wg=1.0, wv=1.0, k=60):
    """Same formula and tie-breaks as src/retrieval/reranker.py, with optional route weights."""
    by = {}
    for r in graph:
        by.setdefault(r["chunk_id"], dict(r, vector_rank=None))["graph_rank"] = r["graph_rank"]
    for r in vector:
        d = by.setdefault(r["chunk_id"], dict(r, graph_rank=None))
        d["vector_rank"] = r["vector_rank"]
        d.setdefault("n_edges", 0)
    # graph rows carry n_edges; vector-only rows have 0
    for cid, d in by.items():
        gs = wg / (k + d["graph_rank"]) if d.get("graph_rank") else 0.0
        vs = wv / (k + d["vector_rank"]) if d.get("vector_rank") else 0.0
        d["score"] = gs + vs
    return sorted(by.values(), key=lambda d: (-d["score"], -d.get("n_edges", 0), 0 if d.get("graph_rank") else 1, d["chunk_id"]))


def load(seed):
    recs = fl.read_jsonl(DIAG / f"part1_records_seed{seed}.jsonl")
    meta = recs[0]["meta"]
    by = defaultdict(dict)
    for r in recs[1:]:
        by[r["question_id"]][r["mode"]] = r
    return meta, by


def main():
    summary = {}
    for seed in (7, "7mo", 11):
        meta, by = load(seed)
        P("=" * 120)
        P(f"SEED {seed}{' (seed-7 KB built with merge only, no revalidation; 53 vs 55 relationships question)' if seed=='7mo' else ''}   engine settings: {meta}")
        # ---- fidelity check against earlier runs
        earlier = {(r["question_id"], r["mode"]): r for r in fl.read_jsonl(EARLIER[seed])}
        same = diff = 0
        diffs = []
        for qid, modes in by.items():
            for m, r in modes.items():
                e = earlier.get((qid, m))
                if e is None:
                    continue
                if e["context_chunk_ids"] == [x["chunk_id"] for x in r.get("context", [])]:
                    same += 1
                else:
                    diff += 1
                    diffs.append((qid, m))
        P(f"fidelity: context chunk-id lists identical to the earlier recorded run for {same} of {same+diff} (question, mode) pairs; differing: {diffs[:10]}")
        errs = [(q, m) for q, ms in by.items() for m, r in ms.items() if r["error"]]
        P(f"query errors in this recording: {errs}")

        # ---- per-question features
        feats = {}
        for qid, modes in by.items():
            h, g, v = modes["hybrid"], modes["graph_only"], modes["vector_only"]
            gold = [fl.norm_doc_id(d) for d in h["gold"]]
            G = h.get("graph", [])  # graph route list inside hybrid
            V = h.get("vector", [])
            M = h.get("merged_full", [])
            ctxH = h["context"]
            f = dict(qid=qid, group=h["group"], gold=gold, intent=h.get("intent"), n_seeds=len(h["resolved_seeds"]),
                     G=G, V=V, M=M, ctxH=ctxH, ctxG=g["context"], ctxV=v["context"])
            f["ok_hybrid"] = fl.all_gold(docs_of(ctxH), gold)
            f["ok_graph_only"] = fl.all_gold(docs_of(g["context"]), gold)
            f["ok_vector_only"] = fl.all_gold(docs_of(v["context"]), gold)
            f["ok_graph_top4"] = fl.all_gold(docs_of(g["context"][:4]), gold)
            f["len_graph_only"] = len(g["context"])
            f["ok_graph_list_full"] = fl.all_gold(docs_of(G), gold)
            f["ok_graph_list_top4"] = fl.all_gold(docs_of(G[:4]), gold)
            f["ok_union"] = fl.all_gold(docs_of(G) + docs_of(V), gold)
            f["ok_merged_top4"] = fl.all_gold(docs_of(M[:4]), gold)
            f["cb_dropped"] = [e["chunk_id"] for e in ctxH if e["chunk_id"] not in h["context_builder_kept"]]
            f["merged_top4_equals_context"] = [e["chunk_id"] for e in M[:4]] == [e["chunk_id"] for e in ctxH]
            f["vector_list_equals_vector_only"] = [e["chunk_id"] for e in V] == [e["chunk_id"] for e in v["context"]]
            f["graph_list_equals_graph_only"] = [e["chunk_id"] for e in G] == [e["chunk_id"] for e in g["context"]]
            f["n_unique_inputs"] = len({e["chunk_id"] for e in G} | {e["chunk_id"] for e in V})
            f["n_merged"] = len(M)
            feats[qid] = f

        # ---- Table 1: all-gold recall per group, per list
        P(f"\nTABLE 1 (seed {seed}): all-gold recall by candidate list (answerable questions)")
        cols = [("hybrid context (K=4)", "ok_hybrid"), ("graph_only as evaluated (variable length)", "ok_graph_only"),
                ("graph_only first 4 chunks", "ok_graph_top4"), ("graph list in hybrid, first 4", "ok_graph_list_top4"),
                ("graph list in hybrid, full", "ok_graph_list_full"), ("vector_only (K=4)", "ok_vector_only"),
                ("union(graph full, vector 4)", "ok_union")]
        P("| list | " + " | ".join(GROUPS) + " |")
        P("|---|" + "---|" * len(GROUPS))
        for name, key in cols:
            cells = []
            for grp in GROUPS:
                ids = [q for q, f in feats.items() if f["group"] == grp]
                cells.append(fl.fmt_rate(sum(feats[i][key] for i in ids), len(ids)))
            P(f"| {name} | " + " | ".join(cells) + " |")
        lens = defaultdict(list)
        for f in feats.values():
            lens[f["group"]].append(f["len_graph_only"])
        P("graph_only context length (chunks) per group: " + "; ".join(
            f"{g}: min {min(v)} median {sorted(v)[len(v)//2]} max {max(v)} share>4: {sum(1 for x in v if x > 4)}/{len(v)}" for g, v in lens.items()))

        # ---- structural checks (hop3 and all)
        P(f"\nSTRUCTURAL CHECKS (seed {seed}, all {len(feats)} answerable questions, hybrid mode)")
        P(f"  merged top-4 == returned context: {sum(f['merged_top4_equals_context'] for f in feats.values())}/{len(feats)}")
        P(f"  ContextBuilder (max_chars={meta['max_context_chars']}, mode={meta['graph_context_mode']}) drops a context chunk: "
          f"{sum(1 for f in feats.values() if f['cb_dropped'])}/{len(feats)} questions; max context chars "
          f"{max(by[q]['hybrid']['context_chars'] for q in by)}")
        P(f"  merge-by-chunk-id: questions where |merged| != |graph chunk ids U vector chunk ids|: "
          f"{sum(1 for f in feats.values() if f['n_unique_inputs'] != f['n_merged'])}/{len(feats)}")
        P(f"  graph list inside hybrid identical to graph_only context: {sum(f['graph_list_equals_graph_only'] for f in feats.values())}/{len(feats)}")
        P(f"  vector list inside hybrid identical to vector_only context: {sum(f['vector_list_equals_vector_only'] for f in feats.values())}/{len(feats)}")
        docs_per_chunk = Counter()
        for f in feats.values():
            for e in f["G"] + f["V"]:
                docs_per_chunk[(e["doc_id"])] += 0
        chunk_doc = {}
        for f in feats.values():
            for e in f["G"] + f["V"]:
                chunk_doc.setdefault(e["doc_id"], set()).add(e["chunk_id"])
        P(f"  distinct chunks per document id across all recorded lists: max {max(len(s) for s in chunk_doc.values())} "
          f"(1 means one chunk per document, so chunk-id collisions cannot merge different documents)")

        # ---- hop3 detail file
        det = []
        for qid, f in feats.items():
            if f["group"] != "hop3":
                continue
            det.append("-" * 110)
            det.append(f"{qid} hop3 gold docs {f['gold']} intent={f['intent']} resolved seeds={f['n_seeds']}  "
                       f"hybrid_ok={f['ok_hybrid']} graph_only_ok={f['ok_graph_only']} (len {f['len_graph_only']}) vector_only_ok={f['ok_vector_only']}")
            det.append("  graph route list (rank, doc, edges, gold?): " + ", ".join(
                f"{e['graph_rank']}:{e['doc_id']}{'*' if fl.norm_doc_id(e['doc_id']) in f['gold'] else ''}(e{e['n_edges']})" for e in f["G"]))
            det.append("  vector route list (rank, doc, gold?): " + ", ".join(
                f"{e['vector_rank']}:{e['doc_id']}{'*' if fl.norm_doc_id(e['doc_id']) in f['gold'] else ''}(dist {e['vector_distance']:.4f})" for e in f["V"]))
            det.append("  merged RRF ranking (pos, doc, routes, graph_rank, vector_rank, score, gold?): " + "; ".join(
                f"{i}:{e['doc_id']}[{'+'.join(e['routes'])} g{e['graph_rank']} v{e['vector_rank']} {e['rrf_score']:.5f}]{'*' if fl.norm_doc_id(e['doc_id']) in f['gold'] else ''}" for i, e in enumerate(f["M"][:10], 1)))
            det.append("  context (final 4): " + ", ".join(f"{e['doc_id']}{'*' if fl.norm_doc_id(e['doc_id']) in f['gold'] else ''}" for e in f["ctxH"]))
            for gd in f["gold"]:
                gr = rank_of(f["G"], gd)
                vr = rank_of(f["V"], gd)
                mr = rank_of(f["M"], gd)
                det.append(f"    gold {gd}: graph-list rank {gr}, vector-list rank {vr}, merged position {mr}, in hybrid context {gd in docs_of(f['ctxH'])}")
        (DIAG / f"part1_hop3_detail_seed{seed}.txt").write_text("\n".join(det) + "\n", encoding="utf-8")

        # ---- hop3 failure classification (hybrid fails)
        P(f"\nTABLE 2 (seed {seed}): hop3 questions, graph_only vs hybrid")
        hop3 = [f for f in feats.values() if f["group"] == "hop3"]
        n = len(hop3)
        both = sum(1 for f in hop3 if f["ok_hybrid"] and f["ok_graph_only"])
        g_only = [f for f in hop3 if f["ok_graph_only"] and not f["ok_hybrid"]]
        h_only = [f for f in hop3 if f["ok_hybrid"] and not f["ok_graph_only"]]
        P(f"  N={n}; both succeed {both}; graph_only only {len(g_only)}; hybrid only {len(h_only)}; neither {n-both-len(g_only)-len(h_only)}")

        def mechanism(f):
            """Classify why hybrid misses at least one gold doc, using only the recorded lists."""
            ctx_docs = set(docs_of(f["ctxH"]))
            missing = [d for d in f["gold"] if d not in ctx_docs]
            if not missing:
                return "hybrid succeeds", {}
            gr = {d: rank_of(f["G"], d) for d in missing}
            vr = {d: rank_of(f["V"], d) for d in missing}
            mr = {d: rank_of(f["M"], d) for d in missing}
            if any(gr[d] is None and vr[d] is None for d in missing):
                return "A: gold doc in neither route list (candidate generation)", dict(graph=gr, vector=vr, merged=mr)
            if any(gr[d] is not None and mr[d] is None for d in missing):
                return "other: gold in graph list but absent from merged list", dict(graph=gr, vector=vr, merged=mr)
            if all(gr[d] is not None and gr[d] <= 4 for d in missing):
                return "B: RRF crowding (gold within graph top-4 but displaced by other chunks)", dict(graph=gr, vector=vr, merged=mr)
            if all(gr[d] is not None for d in missing):
                return "C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)", dict(graph=gr, vector=vr, merged=mr)
            return "D: gold only in vector list or mixed, not selected", dict(graph=gr, vector=vr, merged=mr)

        mech_all = Counter()
        mech_g = Counter()
        rows = []
        for f in hop3:
            m, info = mechanism(f)
            mech_all[m] += 1
            if f in g_only:
                mech_g[m] += 1
            rows.append((f["qid"], m, info))
        P("  mechanism for every hop3 question (hybrid misses >=1 gold doc), seed", seed)
        for m, c in sorted(mech_all.items()):
            P(f"    {c:3d}  {m}")
        P("  mechanism restricted to the questions where graph_only succeeds and hybrid does not:")
        for m, c in sorted(mech_g.items()):
            P(f"    {c:3d}  {m}")
        P("  | question | graph_only len | gold ranks in graph list | gold ranks in vector list | gold merged positions | hybrid context docs | mechanism |")
        P("  |---|---|---|---|---|---|---|")
        for f in g_only:
            m, info = mechanism(f)
            gold_g = [rank_of(f["G"], d) for d in f["gold"]]
            gold_v = [rank_of(f["V"], d) for d in f["gold"]]
            gold_m = [rank_of(f["M"], d) for d in f["gold"]]
            disp = ",".join(f"{e['doc_id']}({'+'.join(e['routes'])})" for e in f["ctxH"])
            P(f"  | {f['qid']} | {f['len_graph_only']} | {gold_g} | {gold_v} | {gold_m} | {disp} | {m} |")
        # displacement detail for gold docs in graph list but not in context
        P("  displacing chunks for gold docs that are in the graph list but not in the hybrid context (all hop3):")
        for f in hop3:
            ctx_docs = set(docs_of(f["ctxH"]))
            for d in f["gold"]:
                gr = rank_of(f["G"], d)
                if gr is not None and d not in ctx_docs:
                    mr = rank_of(f["M"], d)
                    disp = [f"{e['doc_id']}[{'+'.join(e['routes'])} g{e['graph_rank']} v{e['vector_rank']} {e['rrf_score']:.5f}]" for e in f["ctxH"]]
                    sc = next(e["rrf_score"] for e in f["M"] if fl.norm_doc_id(e["doc_id"]) == d)
                    P(f"    {f['qid']} gold {d}: graph rank {gr}, merged position {mr}, score {sc:.5f}; context = {'; '.join(disp)}")

        # ---- route composition of final contexts
        comp = Counter()
        for f in hop3:
            for e in f["ctxH"]:
                comp["+".join(e["routes"])] += 1
        P(f"  route composition of the 4 hybrid context chunks over {n} hop3 questions: {dict(comp)}")
        compall = Counter()
        for f in feats.values():
            for e in f["ctxH"]:
                compall["+".join(e["routes"])] += 1
        P(f"  same, all answerable questions: {dict(compall)}")
        P(f"  hop3 questions with 0 resolved seeds: {sum(1 for f in hop3 if f['n_seeds']==0)}/{n}; graph list empty: {sum(1 for f in hop3 if not f['G'])}/{n}; "
          f"graph list median length {sorted(len(f['G']) for f in hop3)[n//2]}")

        # ---- counterfactuals
        P(f"\nTABLE 3 (seed {seed}): COUNTERFACTUAL RE-RANKING OF RECORDED LISTS. EXPLORATORY, NOT A FIX, SEED ALREADY USED.")
        P("  Candidate lists are the recorded graph list (<= max_graph_candidates) and vector list (vector_top_k=4); only the selection changes.")
        cf = {}

        def sel(f, name):
            G, V = f["G"], f["V"]
            if name == "baseline RRF top-4 (replica)":
                return docs_of(rrf(G, V)[:4])
            if name == "graph top-4 only (K matched)":
                return docs_of(G[:4])
            if name == "graph top-3 + vector top-1":
                return docs_of(G[:3]) + docs_of(V[:1])
            if name == "graph top-2 + vector top-2":
                return docs_of(G[:2]) + docs_of(V[:2])
            if name.startswith("weighted RRF"):
                w = float(name.split("w_graph=")[1].split()[0])
                return docs_of(rrf(G, V, wg=w)[:4])
            if name == "RRF top-6 (K=6)":
                return docs_of(rrf(G, V)[:6])
            if name == "RRF top-8 (K=8)":
                return docs_of(rrf(G, V)[:8])
            if name == "graph full list as evaluated in graph_only (K variable)":
                return docs_of(f["ctxG"])
            raise KeyError(name)

        names = ["baseline RRF top-4 (replica)", "graph top-4 only (K matched)", "graph top-3 + vector top-1", "graph top-2 + vector top-2",
                 "weighted RRF w_graph=2 (K=4)", "weighted RRF w_graph=3 (K=4)", "weighted RRF w_graph=5 (K=4)", "weighted RRF w_graph=10 (K=4)",
                 "RRF top-6 (K=6)", "RRF top-8 (K=8)", "graph full list as evaluated in graph_only (K variable)"]
        P("| selection rule | " + " | ".join(GROUPS) + " | hop3 gained vs baseline | hop3 lost vs baseline | other-group lost vs baseline |")
        P("|---|" + "---|" * (len(GROUPS) + 3))
        base_ok = {q: fl.all_gold(sel(f, names[0]), f["gold"]) for q, f in feats.items()}
        assert all(base_ok[q] == f["ok_hybrid"] for q, f in feats.items()), "replica mismatch vs recorded hybrid context"
        for name in names:
            ok = {q: fl.all_gold(sel(f, name), f["gold"]) for q, f in feats.items()}
            cells = []
            for grp in GROUPS:
                ids = [q for q, f in feats.items() if f["group"] == grp]
                cells.append(fl.fmt_rate(sum(ok[i] for i in ids), len(ids)))
            h3 = [q for q, f in feats.items() if f["group"] == "hop3"]
            gained = sum(1 for q in h3 if ok[q] and not base_ok[q])
            lost = sum(1 for q in h3 if base_ok[q] and not ok[q])
            oth = [q for q, f in feats.items() if f["group"] != "hop3"]
            lost_o = sum(1 for q in oth if base_ok[q] and not ok[q])
            P(f"| {name} | " + " | ".join(cells) + f" | {gained} | {lost} | {lost_o} |")
            cf[name] = {q: ok[q] for q in ok}
        t1 = {name: {grp: sum(feats[q][key] for q, f in feats.items() if f["group"] == grp) for grp in GROUPS} for name, key in cols}
        cfg = {}
        for name in names:
            cfg[name] = {grp: sum(cf[name][q] for q, f in feats.items() if f["group"] == grp) for grp in GROUPS}
        gl = defaultdict(list)
        for f in feats.values():
            gl[f["group"]].append(f["len_graph_only"])
        summary[seed] = dict(
            table1=t1, counterfactual_by_group=cfg, cf_per_question={n: {q: bool(v) for q, v in cf[n].items()} for n in names}, group_of={q: f['group'] for q, f in feats.items()},
            graph_only_len_gt4={g: sum(1 for x in v if x > 4) for g, v in gl.items()},
            group_n={g: len(v) for g, v in gl.items()},
            structural=dict(merged_top4_equals_context=sum(f["merged_top4_equals_context"] for f in feats.values()),
                            cb_drops=sum(1 for f in feats.values() if f["cb_dropped"]),
                            merge_collisions=sum(1 for f in feats.values() if f["n_unique_inputs"] != f["n_merged"]),
                            n_questions=len(feats), fidelity_same=same, fidelity_total=same + diff),
            hop3=dict(n=n, both=both, graph_only_only=len(g_only), hybrid_only=len(h_only), mechanisms=dict(mech_all), mechanisms_graph_only_wins=dict(mech_g)),
            counterfactual_hop3={k: sum(v[q] for q, f in feats.items() if f["group"] == "hop3" for v in [cf[k]]) for k in cf},
        )
    (DIAG / "part1_tables.txt").write_text("\n".join(OUT_LINES) + "\n", encoding="utf-8")
    (DIAG / "part1_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
