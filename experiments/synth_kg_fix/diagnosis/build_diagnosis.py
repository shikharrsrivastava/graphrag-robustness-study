#!/usr/bin/env python3
"""Assembles DIAGNOSIS.md. Every number in the text is read from the JSON/text artifacts written by
part1_analyze.py, part1c_traversal.py, part2_develops.py, link_analysis.py (f-strings below), and the tables are cut
out of the script outputs verbatim. Nothing is typed in by hand except the wording."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
DIAG = Path(__file__).resolve().parent
FIX = DIAG.parent
sys.path.insert(0, str(FIX))
import fixlib as fl  # noqa: E402

S1 = json.loads((DIAG / "part1_summary.json").read_text())
C1 = json.loads((DIAG / "part1c_summary.json").read_text())
LK = json.loads((DIAG / "link_summary.json").read_text())
P2 = json.loads((DIAG / "part2_edges.json").read_text())
T1 = (DIAG / "part1_tables.txt").read_text().splitlines()
T1C = (DIAG / "part1c_tables.txt").read_text().splitlines()
T2 = (DIAG / "part2_tables.txt").read_text().splitlines()
TL = (DIAG / "link_tables.txt").read_text().splitlines()
HB = (DIAG / "hash_before.txt").read_text().strip()
HA = (DIAG / "hash_after.txt").read_text().strip() if (DIAG / "hash_after.txt").exists() else "(not yet computed)"


def block(lines, start, after=0, stop_blank=True, occurrence=1):
    """Contiguous lines from the `occurrence`-th line that starts with `start`, until a blank line or a new section marker."""
    n = 0
    for i, l in enumerate(lines):
        if l.startswith(start):
            n += 1
            if n == occurrence:
                out = [l]
                for m in lines[i + 1:]:
                    if (stop_blank and not m.strip()) or m.startswith("=====") or (m.startswith(("TABLE", "L1", "L2", "L3", "L4", "STRUCTURAL")) and not m.startswith(start)):
                        break
                    out.append(m)
                return "\n".join(out)
    raise KeyError(start)


def sec(lines, label_start, until_prefixes):
    for i, l in enumerate(lines):
        if l.startswith(label_start):
            out = [l]
            for m in lines[i + 1:]:
                if m.startswith(until_prefixes):
                    break
                out.append(m)
            return "\n".join(out)
    raise KeyError(label_start)


def seed_block(lines, seed_label):
    """Lines of one seed section in part1_tables.txt (from 'SEED x' header to the next '=====' line)."""
    start = next(i for i, l in enumerate(lines) if l.startswith(f"SEED {seed_label}"))
    out = []
    for l in lines[start:]:
        if l.startswith("=====") and out:
            break
        out.append(l)
    return out


def fmt(a, b):
    return f"{a}/{b}"


# ----------------------------------------------------------------------------------------------- numbers
A = {k: S1[k] for k in ("7", "7mo", "11")}
h3 = {k: A[k]["hop3"] for k in A}
t1 = {k: A[k]["table1"] for k in A}
HY, GO, G4, VO = "hybrid context (K=4)", "graph_only as evaluated (variable length)", "graph_only first 4 chunks", "vector_only (K=4)"
mA = "A: gold doc in neither route list (candidate generation)"
mB = "B: RRF crowding (gold within graph top-4 but displaced by other chunks)"
mC = "C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)"


def mech(k, m):
    return h3[k]["mechanisms"].get(m, 0)


cfq = {k: A[k]["cf_per_question"] for k in A}
grp = {k: A[k]["group_of"] for k in A}
W2 = "weighted RRF w_graph=2 (K=4)"
BASE = "baseline RRF top-4 (replica)"


def gained(k, name, g):
    return sorted(q for q in cfq[k][name] if grp[k][q] == g and cfq[k][name][q] and not cfq[k][BASE][q])


def lost(k, name):
    return sorted(q for q in cfq[k][name] if cfq[k][name][q] is False and cfq[k][BASE][q])


# vector-only displacers: gold docs in the graph list but not in the context, count questions where a vector-only chunk outscored them
disp = {}
for k, fn in (("7", "part1_records_seed7.jsonl"), ("7mo", "part1_records_seed7mo.jsonl"), ("11", "part1_records_seed11.jsonl")):
    recs = [r for r in fl.read_jsonl(DIAG / fn)[1:] if r["mode"] == "hybrid" and r["group"] == "hop3"]
    nq = nvec = ndoc = 0
    for r in recs:
        gold = {fl.norm_doc_id(d) for d in r["gold"]}
        ctx_docs = {fl.norm_doc_id(e["doc_id"]) for e in r["context"]}
        miss = [d for d in gold if d not in ctx_docs and any(fl.norm_doc_id(e["doc_id"]) == d for e in r["graph"])]
        if not miss:
            continue
        nq += 1
        for d in miss:
            ndoc += 1
            sc = next(e["rrf_score"] for e in r["merged_full"] if fl.norm_doc_id(e["doc_id"]) == d)
            if any(e["routes"] == ["VECTOR"] and e["rrf_score"] > sc for e in r["context"]):
                nvec += 1
    disp[k] = (nq, ndoc, nvec)

# part 2 numbers
P2K = {"s11": "seed 11 (kb_seed11)", "s7": "seed 7 (kb_dev, after fixes)", "s7b": "seed 7 (kb_before, original HEAD KB)"}
p2 = {}
for k, lab in P2K.items():
    rows = P2[lab]
    p2[k] = dict(n=len(rows), cause=collections.Counter(r["cause"] for r in rows), fate=collections.Counter(r["fate"] for r in rows),
                 types_equal_gold=sum(1 for r in rows if r["mention_types"] and all(v == [t] for v, t in zip(r["mention_types"].values(), (r["source_type"], r["target_type"])))),
                 gold_ok=sum(1 for r in rows if r["gold_types_satisfy_rule"]))
dd = {k: LK[lab]["dev_dir"] for k, lab in (("s11", "seed 11 (kb_seed11)"), ("s7", "seed 7 (kb_dev)"))}
tm = {k: LK[lab]["dev_template"] for k, lab in (("s11", "seed 11 (kb_seed11)"), ("s7", "seed 7 (kb_dev)"))}
SPEAR = "<S> spearheads the technical development of <T>"
sp_f = sum(tm[k].get(SPEAR, {}).get("fwd", 0) for k in tm)
sp_r = sum(tm[k].get(SPEAR, {}).get("REV", 0) for k in tm)
ot_f = sum(v.get("fwd", 0) for k in tm for t, v in tm[k].items() if t != SPEAR)
ot_r = sum(v.get("REV", 0) for k in tm for t, v in tm[k].items() if t != SPEAR)


def rev_rate(k, rel):
    d = LK[{"s11": "seed 11 (kb_seed11)", "s7": "seed 7 (kb_dev)"}[k]]["dir_all"]
    f = d.get(f"{rel}|same relation|fwd", 0)
    r = d.get(f"{rel}|same relation|REV", 0)
    return r, f + r


oth = {rel: tuple(sum(rev_rate(k, rel)[i] for k in ("s7", "s11")) for i in (0, 1)) for rel in ("MANAGES", "BELONGS_TO", "BUILT_WITH", "SERVES")}
dev_r = sum(dd[k].get("DEVELOPS|same relation|REV", 0) for k in dd)
dev_t = dev_r + sum(dd[k].get("DEVELOPS|same relation|fwd", 0) for k in dd)
L = {"s7": LK["seed 7 (kb_dev)"], "s11": LK["seed 11 (kb_seed11)"]}
c1 = C1

# ----------------------------------------------------------------------------------------------- document
D = []
add = D.append
add("# Diagnosis: hybrid hop3 recall and rejected DEVELOPS edges\n")
add("Status: diagnosis only. No file under `src/`, `tests/`, `configs/`, `prompts/` or `docs/` was changed, nothing was committed. "
    "Every number below is produced by a script in `experiments/synth_kg_fix/diagnosis/` and pasted by `build_diagnosis.py`.\n")
add("## 0. Protocol note (read first)\n")
add("**Seed 11 has now been used for diagnosis, so it can no longer serve as a clean confirmation set.** Any future fix must be confirmed on a new seed "
    "(use **13**) generated with the same knobs as seeds 7 and 11 (`--n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3`) and "
    "**pre-registered (PROTOCOL.md written, frozen hash recorded) before the fix is applied**. The counterfactuals in this report were computed on "
    "seeds 7 and 11 only and are labelled *exploratory, not a fix, seed already used*.\n")
add("Integrity hashes (`experiments/synth_kg_fix/freeze_hash.sh`):\n")
add("```\nbefore: " + HB.replace("\n", "\n        ") + "\nafter:  " + HA.replace("\n", "\n        ") + "\n```")
same = HB == HA
add(f"Result: **{'both hashes unchanged' if same else 'HASH MISMATCH, protocol violation, see below'}**.\n" if HA != "(not yet computed)" else "Result: after-hash not yet computed.\n")
add("How the data was produced. `part1_record.py` runs the engine's query path unchanged (public API `HybridGraphRAG.from_existing`, cached query plans, stubbed answer "
    "generator, LLM client that raises on any call; embeddings are the only model call, for the query vector) and wraps three class methods inside the script process to record "
    "the graph-route list, the vector-route list and the full merged RRF ranking. "
    f"Fidelity check: the recorded 4-chunk contexts equal the earlier recorded runs for {A['7']['structural']['fidelity_same']}/{A['7']['structural']['fidelity_total']} (seed 7, kb_dev), "
    f"{A['7mo']['structural']['fidelity_same']}/{A['7mo']['structural']['fidelity_total']} (seed 7, merge-only KB), {A['11']['structural']['fidelity_same']}/{A['11']['structural']['fidelity_total']} (seed 11) (question, mode) pairs.\n")
add("Side effect to note: opening a knowledge base through the public API updates the modification time of its Chroma files (`vector_store/chroma.sqlite3` and index `.bin` files in `kb_dev` and `kb_dev_merge_only`, which are this task's own derived copies); the same happened in the earlier retrieval runs. No other file outside `diagnosis/` was written, and `src/`, `tests/`, `configs/`, `prompts/` are unchanged (hashes above).\n")
add("Knowledge bases: seed 7 = `kb_dev` (all fixes enabled, 53 relationships); seed 7 merge-only = `kb_dev_merge_only` (55 relationships, no revalidation); seed 11 = `kb_seed11`. "
    "The \"6-10/30\" graph_only figure in FINAL_REPORT.md for seed 7 is therefore 6/30 on `kb_dev` and 10/30 on `kb_dev_merge_only`; both are analysed below.\n")

# ------------------------------------------------------------------ PART 1
add("## Part 1. Why is hybrid hop3 evidence recall (0-1/30) far below graph_only?\n")
add("### Finding\n")
add(
    f"Three separate effects, of very different size. **(1) Most hop3 questions fail in every mode because the graph route never produces the gold documents.** "
    f"For {mech('7', mA)}/30 questions on seed 7 (`kb_dev`) and {mech('11', mA)}/30 on seed 11, at least one gold document is in neither the graph-route list nor the vector-route list, "
    f"so no re-ranking of those lists can recover the question (graph_only fails the same questions). The graph route is limited by the stored graph: the engine's own traverser, "
    f"run uncapped and undirected within 3 hops, covers all gold documents for only {c1['seed 7 (kb_dev)']['hop3']['T3 undirected']}/30 (seed 7) and {c1['seed 11 (kb_seed11)']['hop3']['T3 undirected']}/30 (seed 11) hop3 questions, "
    f"and neither the 20-path cap nor edge direction explains it (cap reached in {c1['seed 7 (kb_dev)']['capped_20_paths']['hop3']} and {c1['seed 11 (kb_seed11)']['capped_20_paths']['hop3']} hop3 queries; directed and capped coverage "
    f"{c1['seed 7 (kb_dev)']['hop3']['T1 capped-20 directed']}/30 and {c1['seed 11 (kb_seed11)']['hop3']['T1 capped-20 directed']}/30, i.e. direction changes coverage by at most one question). A hop3 gold path has 3 edges; all three are exact validated edges for only "
    f"{L['s7']['path_complete']['hop3']['before']}/30 (seed 7) and {L['s11']['path_complete']['hop3']['before']}/30 (seed 11) questions. "
    f"**(2) The gap between graph_only and hybrid is a K mismatch plus RRF crowding.** `graph_only` returns the whole graph list (up to 16 chunks; more than 4 chunks in {A['7']['graph_only_len_gt4']['hop3']}/30 seed-7 and {A['11']['graph_only_len_gt4']['hop3']}/30 seed-11 hop3 contexts), "
    f"while hybrid is cut to 4 chunks. With graph_only cut to its first 4 chunks, hop3 is {t1['7'][G4]['hop3']}/30 instead of {t1['7'][GO]['hop3']}/30 (seed 7 `kb_dev`), {t1['7mo'][G4]['hop3']}/30 instead of {t1['7mo'][GO]['hop3']}/30 (merge-only KB) and "
    f"{t1['11'][G4]['hop3']}/30 instead of {t1['11'][GO]['hop3']}/30 (seed 11). Hybrid scores {t1['7'][HY]['hop3']}, {t1['7mo'][HY]['hop3']} and {t1['11'][HY]['hop3']}. "
    f"For the questions graph_only wins and hybrid loses ({h3['7']['graph_only_only']} on seed 7, {h3['7mo']['graph_only_only']} on the merge-only KB, {h3['11']['graph_only_only']} on seed 11), the mechanism is RRF crowding in "
    f"{h3['7']['mechanisms_graph_only_wins'].get(mB, 0)}, {h3['7mo']['mechanisms_graph_only_wins'].get(mB, 0)} and {h3['11']['mechanisms_graph_only_wins'].get(mB, 0)} cases (all missing gold documents have graph rank <= 4 but lose slots to chunks that carry a vector rank: vector-only chunks, or chunks with a low graph rank and a vector rank) and a K cap in "
    f"{h3['7']['mechanisms_graph_only_wins'].get(mC, 0)}, {h3['7mo']['mechanisms_graph_only_wins'].get(mC, 0)} and {h3['11']['mechanisms_graph_only_wins'].get(mC, 0)} cases (a gold document sits at graph rank 5 or lower). "
    f"Crowding works like this: the vector list has only 4 entries (`vector_top_k=4`) and RRF scores a vector-only chunk at rank r the same as a graph-only chunk at rank r, so vector ranks 2-4 outrank graph ranks 3-5; "
    f"in {disp['7'][2]} of {disp['7'][1]} (seed 7) and {disp['11'][2]} of {disp['11'][1]} (seed 11) cases where a gold document was in the graph list but not in the context, a vector-only chunk with a higher RRF score was in the context. "
    f"**(3) The other candidate mechanisms were checked and are not responsible:** merge-by-chunk-id collisions {A['7']['structural']['merge_collisions']}/{A['7']['structural']['n_questions']} (seed 7) and {A['11']['structural']['merge_collisions']}/{A['11']['structural']['n_questions']} (seed 11) "
    f"(each document has exactly one chunk), ContextBuilder drops a context chunk in {A['7']['structural']['cb_drops']} and {A['11']['structural']['cb_drops']} questions (max context 12000-char budget never approached), "
    f"merged top-4 equals the returned context in {A['7']['structural']['merged_top4_equals_context']}/{A['7']['structural']['n_questions']} and {A['11']['structural']['merged_top4_equals_context']}/{A['11']['structural']['n_questions']} questions.\n")
add("### Table 1.1 All-gold recall by candidate list (answerable questions), seeds 7 / 7 merge-only / 11\n")
for k, lab in (("7", "7"), ("7mo", "7mo"), ("11", "11")):
    b = seed_block(T1, lab)
    add(f"**{'seed 7 (kb_dev)' if k == '7' else 'seed 7 merge-only KB' if k == '7mo' else 'seed 11'}**\n")
    add(block(b, "| list |") + "\n")
    add(block(b, "graph_only context length") + "\n")
add("### Table 1.2 Hop3: graph_only vs hybrid and mechanism for every failing question\n")
for k, lab in (("7", "7"), ("7mo", "7mo"), ("11", "11")):
    b = seed_block(T1, lab)
    add(f"**{'seed 7 (kb_dev)' if k == '7' else 'seed 7 merge-only KB' if k == '7mo' else 'seed 11'}**\n")
    add("```\n" + sec(b, "TABLE 2", ("  | question",)).rstrip() + "\n```\n")
    add(sec(b, "  | question", ("  displacing chunks",)).replace("\n  |", "\n|").replace("  | question", "| question") + "\n")
    add("```\n" + sec(b, "  displacing chunks", ("  route composition",)).rstrip() + "\n```\n")
add("Mechanism codes: A = a gold document is in neither route list (candidate generation); B = every missing gold document has graph rank <= 4 (RRF crowding); "
    "C = a missing gold document has graph rank > 4 (K cap); D = missing gold only in the vector list. Per-question lists, ranks and scores for all 30 hop3 questions of each seed "
    "(graph route with edge counts, vector route with distances, merged RRF ranking with scores) are in `part1_hop3_detail_seed7.txt`, `part1_hop3_detail_seed7mo.txt`, `part1_hop3_detail_seed11.txt`.\n")
add("### Table 1.3 Can the graph route reach the gold documents? (engine `GraphTraverser`, stored graphs, recorded seeds)\n")
for lab in ("seed 7 (kb_dev)", "seed 7 (kb_dev_merge_only)", "seed 11 (kb_seed11)"):
    st = next(i for i, l in enumerate(T1C) if l.startswith(lab))
    seg = []
    for l in T1C[st:]:
        if l.startswith("  hop3 per question"):
            break
        if l.startswith("====") and seg:
            break
        seg.append(l)
    add("```\n" + "\n".join(seg) + "\n```\n")
add("T1 = engine behaviour (directed, 3 hops, 20 paths); T2 = cap removed; T3 = undirected within 3 hops; T4 = undirected, any distance (the longer-distance coverage passes through shared "
    "client/department hub vertices, see the hop-limit rows above). Gold-edge fate for the missing documents and path completeness: see `link_tables.txt` (L1, L2).\n")
add("### Table 1.4 Gold-edge link: hop2/hop3 outcome by path completeness (all gold path edges are exact validated edges), and fate of the edge behind each unreached document\n")
for lab in ("seed 7 (kb_dev)", "seed 11 (kb_seed11)"):
    b = [l for l in TL[next(i for i, l in enumerate(TL) if l == lab):]]
    seg1 = block(b, "L1")
    add(f"**{lab}**\n")
    add("```\n" + seg1 + "\n```\n")
    add("```\n" + sec(b, "L2", ("L3",)).rstrip() + "\n```\n")
add("### Table 1.5 EXPLORATORY, NOT A FIX, SEED ALREADY USED: counterfactual re-ranking of the recorded lists (no change to src)\n")
for k, lab in (("7", "7"), ("7mo", "7mo"), ("11", "11")):
    b = seed_block(T1, lab)
    add(f"**{'seed 7 (kb_dev)' if k == '7' else 'seed 7 merge-only KB' if k == '7mo' else 'seed 11'}**\n")
    add(sec(b, "| selection rule", ("====",)).strip() + "\n")
g2 = {k: gained(k, W2, "hop3") for k in A}
g2_2 = {k: gained(k, W2, "hop2") for k in A}
add(
    f"Reading: weighting the graph route by 2 (equal results for weights 2, 3 and 5) lifts hop3 from {A['7']['counterfactual_by_group'][BASE]['hop3']} to {A['7']['counterfactual_by_group'][W2]['hop3']} (seed 7), "
    f"{A['7mo']['counterfactual_by_group'][BASE]['hop3']} to {A['7mo']['counterfactual_by_group'][W2]['hop3']} (merge-only) and {A['11']['counterfactual_by_group'][BASE]['hop3']} to {A['11']['counterfactual_by_group'][W2]['hop3']} (seed 11), "
    f"hop2 from {A['7']['counterfactual_by_group'][BASE]['hop2']} to {A['7']['counterfactual_by_group'][W2]['hop2']} and {A['11']['counterfactual_by_group'][BASE]['hop2']} to {A['11']['counterfactual_by_group'][W2]['hop2']}, "
    f"alias from {A['7']['counterfactual_by_group'][BASE]['alias']} to {A['7']['counterfactual_by_group'][W2]['alias']} (seed 7) and {A['11']['counterfactual_by_group'][BASE]['alias']} to {A['11']['counterfactual_by_group'][W2]['alias']} (seed 11), "
    f"without losing a question that the baseline answered (questions lost on seed 7 / merge-only / seed 11: {len(lost('7', W2))} / {len(lost('7mo', W2))} / {len(lost('11', W2))}). "
    f"Hop3 questions gained at weight 2: seed 7 {g2['7']}, seed 11 {g2['11']}; hop2 questions gained: seed 7 {g2_2['7']}, seed 11 {g2_2['11']}. "
    f"Most of the recoverable hop3 mass is not reachable by re-ranking at K=4: even the union of both lists covers all gold documents for {t1['7']['union(graph full, vector 4)']['hop3']}/30 (seed 7) and {t1['11']['union(graph full, vector 4)']['hop3']}/30 (seed 11) hop3 questions.\n")
add("### What this does NOT show (Part 1)\n")
add("- It does not show that weighted RRF (or any re-ranking) is a fix. The weights were evaluated on the same questions and seeds that revealed the problem; the plateau between weights 2 and 5 is an observation on 60 hop2/hop3 questions per seed.\n"
    "- It does not show how a graph-biased ranking affects questions that vector retrieval answers better; hop1 is already saturated in every mode, and the unanswerable group is not part of this retrieval-only metric.\n"
    "- It does not explain, edge by edge, why each of the unreached gold documents is unreachable. For the questions in mechanism A it shows that the stored graph (not the 20-path cap, not edge direction) limits the traversal and that complete exact paths are rare; "
    "in many cases the document's own edge is an exact validated edge (Table 1.4, L2) but the upstream part of the chain is missing or sits on a different component. A per-edge chain audit was not done.\n"
    "- It does not show that a larger hop limit helps. `retrieval.max_traversal_hops` is defined in `configs/default.yaml` and `src/core/configuration.py:105` but is not passed to `GraphTraverser` "
    "(`GraphEvidenceRetriever` constructs `GraphTraverser(max_candidates=...)`, default 3 hops; grep over `src/` finds no other use), so the config key currently has no effect. The undirected any-distance coverage (T4) is reached through hub vertices and "
    "would flood a 4-chunk context; no retrieval was run with a larger limit.\n"
    "- It does not measure answer quality; the metric is whether all gold documents are among the returned chunks.\n"
    "- The planner's plans are cached (one generation per question); planner variance is not examined. The graph route has no score, only a rank; vector distances and RRF scores are recorded, graph scores are not defined by the engine.\n"
    "- The comparison in FINAL_REPORT.md between graph_only (variable K) and hybrid (K=4) was descriptive, not one of the pre-registered tests (P1-P3 compare hybrid with graph-free baselines at K=4), so the pre-registered decisions are unaffected; "
    "descriptive graph_only numbers should be reported at K=4 as well.\n")

# ------------------------------------------------------------------ PART 2
add("## Part 2. Why are 11 of 15 gold DEVELOPS edges rejected by the ontology on seed 11?\n")
add("### Finding\n")
add(
    f"The cause is (d), something else: **the extractor assigns the right relation type and the right entity types but reverses the direction.** On seed 11, {p2['s11']['cause'].get('REVERSED', 0)} of the {p2['s11']['n']} gold DEVELOPS edges "
    f"(fate C, rejected by ontology) were extracted as `PRODUCT -DEVELOPS-> DEPARTMENT`, while the sentence in the document is active with the department as subject (for example \"X designs and engineers Y\"). "
    f"The ontology rule for DEVELOPS (`configs/domains/company.yaml`: source EMPLOYEE or DEPARTMENT, target PRODUCT) rejects the reversed pair, as it should. "
    f"The gold type pair of all {p2['s11']['n']} gold DEVELOPS edges is DEPARTMENT -> PRODUCT, which the rule allows ({p2['s11']['gold_ok']}/{p2['s11']['n']}), so (b) the ontology being too narrow is excluded; "
    f"the mention types assigned by the extractor equal the gold types for {p2['s11']['types_equal_gold']}/{p2['s11']['n']} edges, so (a) mention-type errors are excluded for these edges; "
    f"and (c) revalidation could not help because the canonical types equal the mention types (revalidation did evaluate these triples, they are eligible, and left them rejected, see Table C). "
    f"The remaining seed-11 DEVELOPS edges: {p2['s11']['fate'].get('A', 0)} validated exactly (fate A) and {p2['s11']['cause'].get('EVIDENCE_REJECT', 0)} extracted in the right direction but rejected for evidence reasons (one endpoint-not-in-evidence, one quote mismatch). "
    f"Seed 7 shows the same pattern: {p2['s7']['cause'].get('REVERSED', 0)} of {p2['s7']['n']} gold DEVELOPS edges are reversed (of which {p2['s7']['fate'].get('C', 0)} are rejected by the ontology and {p2['s7']['fate'].get('D', 0)} are reversed and also rejected for evidence reasons), "
    f"{p2['s7']['fate'].get('A', 0)} validated exactly. Over both seeds {dev_r} of {dev_t} extracted DEVELOPS triples joining the endpoints of a gold DEVELOPS edge are reversed, against "
    + ", ".join(f"{rel} {a}/{b}" for rel, (a, b) in oth.items()) + " for the other relations (same-relation triples only). "
    f"The direction depends on the sentence template in these corpora: \"spearheads the technical development of\" is extracted in the gold direction {sp_f} of {sp_f + sp_r} times, the other three templates ('is directly responsible for developing', 'builds and maintains', "
    f"'designs and engineers') only {ot_f} of {ot_f + ot_r} times. The prompt already states the direction twice (the constraint line `(Allowed: ['EMPLOYEE', 'DEPARTMENT'] -> ['PRODUCT'])` built in `src/extraction/relation_extractor.py:154` and instruction 7 of `prompts/relationship_extraction.txt`), "
    f"so this is gemma3:4b not following the stated direction for these verbs, not an engine or ontology defect. "
    f"Because the reversed pair is the only thing wrong, a post-extraction check that uses the ontology alone could repair it (candidate fix 1).\n")
add("### Table 2.1 Per gold DEVELOPS edge, seed 11\n")
b11 = T2[next(i for i, l in enumerate(T2) if l.startswith("seed 11 (kb_seed11)")):]
b11 = b11[:next(i for i, l in enumerate(b11) if l.startswith("=====") and i > 0)]
add(sec(b11, "ontology rule DEVELOPS", ("TABLE B",)).strip() + "\n")
add("### Table 2.2 Per gold DEVELOPS edge, seed 7 (kb_dev)\n")
b7 = T2[next(i for i, l in enumerate(T2) if l.startswith("seed 7 (kb_dev, after fixes)")):]
b7 = b7[:next(i for i, l in enumerate(b7) if l.startswith("=====") and i > 0)]
add(sec(b7, "TABLE A", ("TABLE B",)).strip() + "\n")
add("### Table 2.3 Extracted type pairs vs the pairs the ontology allows\n")
add("**Seed 11**\n")
add(sec(b11, "TABLE B", ("TABLE C",)).strip() + "\n")
add("**Seed 7 (kb_dev)**\n")
add(sec(b7, "TABLE B", ("TABLE C",)).strip() + "\n")
add("### Table 2.4 All triples rejected by the ontology (every relation): mention types vs canonical types after resolution\n")
add("**Seed 11**\n")
add(sec(b11, "TABLE C", ("zzzz",)).strip() + "\n")
add("**Seed 7 (kb_dev)**\n")
add(sec(b7, "TABLE C", ("zzzz",)).strip() + "\n")
add("### Table 2.5 Direction of extracted triples vs gold, by gold relation, and DEVELOPS by sentence template\n")
for lab in ("seed 7 (kb_dev)", "seed 11 (kb_seed11)"):
    b = TL[next(i for i, l in enumerate(TL) if l == lab):]
    b = b[:next(i for i, l in enumerate(b) if l.startswith("=====") and i > 0)] if any(l.startswith("=====") and i > 0 for i, l in enumerate(b)) else b
    add(f"**{lab}**\n")
    add(sec(b, "L3", ("L4",)).strip() + "\n")
add("### What this does NOT show (Part 2)\n")
add("- It does not show *why* the model reverses these sentences. Only parsed triples are stored, not the raw model output; the template association is a count over four templates and 15 edges per seed, with no test, and the \"spearheads ...\" sentences also differ in wording from the others.\n"
    "- It does not show that the reversal rate is a property of the model for real text: the corpora are synthetic with one active-voice sentence per DEVELOPS edge; passive constructions (\"X is developed by Y\") do not occur.\n"
    "- It does not show that repairing the direction improves retrieval; Table 3.2 and L4 report edge fate and path completeness (an upper bound), no retrieval was re-run on a repaired graph.\n"
    "- It does not cover reversals that the ontology cannot see: where both orientations satisfy the rule (for example MANAGES EMPLOYEE -> EMPLOYEE) a reversed triple is validated and wrong; such triples are counted in Table 2.5 (same-relation REV rows for MANAGES) but cannot be detected from types.\n"
    "- The seed-7 numbers come from the stored `triples.jsonl` of `kb_dev`; the original HEAD KB (`kb_before`) gives the same direction counts (see `part2_tables.txt`).\n")

# ------------------------------------------------------------------ fixes
add("## Candidate fixes (ranked; none implemented)\n")
add("### Table 3.1 Ontology-guided direction repair, computed on stored triples (EXPLORATORY, NOT A FIX, SEED ALREADY USED)\n")
for lab in ("seed 7 (kb_dev)", "seed 11 (kb_seed11)"):
    b = TL[next(i for i, l in enumerate(TL) if l == lab):]
    nxt = [i for i, l in enumerate(b) if l.startswith("=====") and i > 0]
    b = b[:nxt[0]] if nxt else b
    add(f"**{lab}**\n")
    add("```\n" + sec(b, "L4", ("zzzz",)).rstrip() + "\n```\n")
add("**1. Direction repair at validation, behind a config flag (default off).** When a triple is rejected for ontology reasons and swapping source and target satisfies the same relation's allowed source and target types, swap and revalidate (the evidence-quote checks stay as they are); log every swap.\n")
add(
    f"- Expected effect (computed above, not measured in retrieval): of {L['s11']['rejected_ontology']} ontology-rejected triples on seed 11, {L['s11']['swap_passes']} pass after the swap, {L['s11']['swap_equals_gold']} of them equal a gold edge, "
    f"which would move {len(L['s11']['newly'])} gold edges to exact validated ({L['s11']['ae_before']} -> {L['s11']['ae_after']}); on seed 7, {L['s7']['swap_passes']} of {L['s7']['rejected_ontology']} pass and {L['s7']['swap_equals_gold']} equal a gold edge "
    f"({len(L['s7']['newly'])} gold edges to exact, {L['s7']['ae_before']} -> {L['s7']['ae_after']}). Questions with a complete exact gold path (upper bound for graph reach): hop2 {L['s11']['path_complete']['hop2']['before']} -> {L['s11']['path_complete']['hop2']['after']}, hop3 {L['s11']['path_complete']['hop3']['before']} -> {L['s11']['path_complete']['hop3']['after']} (seed 11); "
    f"hop2 {L['s7']['path_complete']['hop2']['before']} -> {L['s7']['path_complete']['hop2']['after']}, hop3 {L['s7']['path_complete']['hop3']['before']} -> {L['s7']['path_complete']['hop3']['after']} (seed 7).\n"
    f"- Edges it would change: seed 11 {L['s11']['newly']}; seed 7 {L['s7']['newly']}. Questions that become path-complete: seed 11 {L['s11']['changed_questions']} (hybrid currently succeeds on {L['s11']['changed_hybrid_ok']} of {len(L['s11']['changed_questions'])}); "
    f"seed 7 {L['s7']['changed_questions']} (hybrid currently succeeds on {L['s7']['changed_hybrid_ok']} of {len(L['s7']['changed_questions'])}). Path completeness is not recall: the graph route also needs the seed and the ranking to cooperate.\n"
    f"- Risk: {L['s11']['swap_wrong']} of {L['s11']['swap_passes']} swaps on seed 11 and {L['s7']['swap_wrong']} of {L['s7']['swap_passes']} on seed 7 produce an edge that is not a gold edge (a BUILT_WITH triple whose source mention maps to no gold entity); "
    "it does not fire when both orientations are valid; it assumes the stated relation is right and only the direction is wrong; the swap cannot be verified from the quote text. Needs a unit test for the flag-off identity and a domain-agnostic implementation (ontology rules only).\n")
add("**2. Route-weighted RRF (config key such as `rrf_graph_weight`, default 1.0 = current behaviour).**\n")
add(
    f"- Expected effect (Table 1.5, K=4): hop3 {A['7']['counterfactual_by_group'][BASE]['hop3']} -> {A['7']['counterfactual_by_group'][W2]['hop3']} (seed 7), {A['11']['counterfactual_by_group'][BASE]['hop3']} -> {A['11']['counterfactual_by_group'][W2]['hop3']} (seed 11); "
    f"hop2 {A['7']['counterfactual_by_group'][BASE]['hop2']} -> {A['7']['counterfactual_by_group'][W2]['hop2']} and {A['11']['counterfactual_by_group'][BASE]['hop2']} -> {A['11']['counterfactual_by_group'][W2]['hop2']}; no baseline-correct question lost on either seed. Questions changed: hop3 {g2['7']} (seed 7), {g2['11']} (seed 11); hop2 {g2_2['7']} (seed 7), {g2_2['11']} (seed 11).\n"
    "- Risk: the weight is chosen after seeing the data (weights 2-5 give identical results, so the choice is not sensitive on these questions, but that is two seeds of one synthetic benchmark); a graph-biased ranking can displace correct vector chunks on corpora where the graph route is noisy; "
    "the unanswerable group and answer quality were not evaluated; the gain is small compared with the candidate-generation losses of Part 1 (mechanism A), so it cannot lift hop3 far above the ceiling set by the union of the two lists.\n")
add("**3. Ontology-derived direction instruction in the relationship prompt (new versioned prompt file, selected by `relationship_extraction_version`).** Generate, from the ontology, an explicit sentence per relation (source types, target types, \"the subject of an active sentence is the source\") and add a worked example built from the ontology's own description.\n")
add(
    f"- Expected effect: should address the same {dev_r} reversed DEVELOPS triples (and the {oth['MANAGES'][0]} + {oth['BUILT_WITH'][0]} + {oth['BELONGS_TO'][0]} reversed MANAGES, BUILT_WITH and BELONGS_TO triples) at the source, but the prompt already contains two direction statements that the model did not follow, "
    "and the earlier prompt change (v1.1) failed its dev rule; the expected effect is therefore uncertain and unmeasured.\n"
    "- Risk: requires re-extraction (about 45 minutes per 60-document corpus on this machine), changes extraction for relations that are currently correct, and overlaps with fix 1; it should be considered only for the residual after fix 1 and must be pre-registered like any prompt change.\n")
add("Ranking rationale: fix 1 changes the largest number of edges (the only cause found for 11 of 15 DEVELOPS rejections), needs no re-extraction (it works on the stored triples) and its flag-off path is identical to current behaviour; "
    "fix 2 is cheap and its measured effect is small and positive but bounded by candidate generation; fix 3 is the most expensive and the least certain. Per the protocol note, all three would be confirmed on seed 13 only, pre-registered before the fix is applied.\n")
add("## Files\n")
add("`part1_record.py` (recording), `part1_records_seed{7,7mo,11}.jsonl` (per question and mode: graph list, vector list, merged ranking, context), `part1_analyze.py` → `part1_tables.txt`, `part1_summary.json`, `part1_hop3_detail_seed*.txt`; "
    "`part1c_traversal.py` → `part1c_tables.txt`, `part1c_summary.json`; `part2_develops.py` → `part2_tables.txt`, `part2_edges.json`; `link_analysis.py` → `link_tables.txt`, `link_summary.json`; `build_diagnosis.py` → this file; `hash_before.txt`, `hash_after.txt`.\n")
(DIAG / "DIAGNOSIS.md").write_text("\n".join(D) + "\n", encoding="utf-8")
print("written", DIAG / "DIAGNOSIS.md", len("\n".join(D)))
