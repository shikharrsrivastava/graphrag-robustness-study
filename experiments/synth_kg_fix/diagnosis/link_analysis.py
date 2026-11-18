#!/usr/bin/env python3
"""Diagnosis: link retrieval outcomes (Part 1) to gold-edge fate and edge direction (Part 2). No engine, no LLM.

Reads part1_records_seed{7,11}.jsonl, the KB files, gold KG, configs/domains/company.yaml.
Writes link_tables.txt.

Section L1  hop2/hop3 questions by path completeness (every gold path edge has fate A/E, i.e. exact validated edge)
            vs hybrid / graph_only / graph top-4 outcome.
Section L2  for each hop3 question where hybrid misses a gold doc that is in NEITHER route list (mechanism A in
            part1_analyze.py): fate of the gold edge(s) supported by that doc.
Section L3  direction of extracted triples vs gold direction, by gold relation, and DEVELOPS by sentence template.
Section L4  EXPLORATORY counterfactual computed on stored triples: swap source/target of triples rejected by the
            ontology when the swapped pair satisfies the same relation's rule. Which gold edges would become exact,
            how many swapped edges would be wrong, how many hop2/hop3 questions would become path-complete.
            Not a fix, seed already used, no retrieval is re-run.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
DIAG = Path(__file__).resolve().parent
FIX = DIAG.parent
ROOT = FIX.parents[1]
sys.path.insert(0, str(FIX))
import fixlib as fl  # noqa: E402
import kb_tables as kt  # noqa: E402

LINES: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


RULES = yaml.safe_load((ROOT / "configs/domains/company.yaml").read_text())["ontology"]["relationship_types"]


def ok(s, r, t):
    x = RULES.get(r)
    return bool(x) and s in x["allowed_source_types"] and t in x["allowed_target_types"]


SETS = [
    ("seed 7 (kb_dev)", 7, FIX / "kb_dev/synth_company_kb", ROOT / "experiments/synth_kg/corpus_final", FIX / "diagnosis/part1_records_seed7.jsonl"),
    ("seed 11 (kb_seed11)", 11, FIX / "kb_seed11/synth_company_kb", FIX / "corpus_seed11", FIX / "diagnosis/part1_records_seed11.jsonl"),
]


def template(quote, sname, tname):
    q = quote.strip().rstrip(".")
    q = q.replace(sname, "<S>").replace(tname, "<T>")
    q = re.sub(r"^(It|They|He|She)\b", "<S>", q)
    return q


def main():
    score = kt.load_score()
    summary = {}
    for label, seed, kbd, corpus, rec_path in SETS:
        gold = json.loads((corpus / "gold_kg.json").read_text())
        kb = kt.KBView(kbd, gold, score)
        edges = {e["id"]: e for e in gold["edges"]}
        fate = {eid: kb.fate(ge)[0] for eid, ge in edges.items()}
        questions = {q["id"]: q for q in fl.read_jsonl(corpus / "questions.jsonl") if q["group"] != "unanswerable"}
        recs = fl.read_jsonl(rec_path)[1:]
        by = collections.defaultdict(dict)
        for r in recs:
            by[r["question_id"]][r["mode"]] = r

        def okq(qid, mode, k=None):
            ctx = by[qid][mode]["context"]
            if k:
                ctx = ctx[:k]
            return fl.all_gold([c["doc_id"] for c in ctx], questions[qid]["gold_supporting_doc_ids"])

        P("=" * 118)
        P(f"{label}")
        # ---- L1
        P(f"\nL1 ({label}): path completeness (all gold path edges have fate A or E) vs outcome")
        P("| group | path-complete? | N | hybrid ok | graph_only ok | graph top-4 ok | vector_only ok |")
        P("|---|---|---|---|---|---|---|")
        for grp in ("hop2", "hop3"):
            for comp in (True, False):
                ids = [q for q, x in questions.items() if x["group"] == grp and (all(fate[e] in "AE" for e in x["gold_path"]) == comp)]
                P(f"| {grp} | {'yes' if comp else 'no'} | {len(ids)} | {sum(okq(i,'hybrid') for i in ids)} | {sum(okq(i,'graph_only') for i in ids)} | "
                  f"{sum(okq(i,'graph_only',4) for i in ids)} | {sum(okq(i,'vector_only') for i in ids)} |")
        # per-hop3 number of path edges and fate multiset
        P(f"  hop3 gold path length distribution: {dict(collections.Counter(len(x['gold_path']) for x in questions.values() if x['group']=='hop3'))}")
        P(f"  hop3 questions by number of path edges with fate A/E: " + str(dict(sorted(collections.Counter(sum(fate[e] in 'AE' for e in x['gold_path']) for x in questions.values() if x['group']=='hop3').items()))) + " (key = edges with exact validated edge)")

        # ---- L2
        P(f"\nL2 ({label}): hop3 questions, hybrid fails, a missing gold doc is in neither route list: fate of the gold edge supported by that doc")
        c = collections.Counter()
        rows = []
        for qid, x in questions.items():
            if x["group"] != "hop3":
                continue
            h = by[qid]["hybrid"]
            gold_docs = [fl.norm_doc_id(d) for d in x["gold_supporting_doc_ids"]]
            ctx = {fl.norm_doc_id(e["doc_id"]) for e in h["context"]}
            lists = {fl.norm_doc_id(e["doc_id"]) for e in h.get("graph", [])} | {fl.norm_doc_id(e["doc_id"]) for e in h.get("vector", [])}
            for d in gold_docs:
                if d not in ctx and d not in lists:
                    es = [e for e in x["gold_path"] if fl.norm_doc_id(edges[e]["supporting_doc_id"]) == d]
                    for e in es:
                        c[(fate[e], edges[e]["relation"])] += 1
                        rows.append((qid, d, e, edges[e]["relation"], fate[e]))
        P("| gold edge fate | relation | count of (question, missing doc) |")
        P("|---|---|---|")
        for (f, r), n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
            P(f"| {f} {kt.FATE_LABEL[f]} | {r} | {n} |")
        P(f"  total (question, missing doc, edge) triples: {sum(c.values())}; with fate A/E: {sum(n for (f,_),n in c.items() if f in 'AE')}")

        # ---- L3 direction table
        P(f"\nL3 ({label}): extracted triples joining the two endpoints of a gold edge (in the supporting doc): direction vs gold")
        mention = {m["mention_id"]: m for m in kb.mentions}
        dirc = collections.Counter()
        tmpl = collections.defaultdict(collections.Counter)
        for eid, ge in edges.items():
            did = fl.norm_doc_id(ge["supporting_doc_id"])
            S, T = ge["source"], ge["target"]
            trips = [t for t in kb.triples if {kb.m2g.get(t["source_entity_id"]), kb.m2g.get(t["target_entity_id"])} == {S, T}
                     and any(fl.norm_doc_id(x) == did for x in t["source_doc_ids"])]
            if not trips:
                dirc[(ge["relation"], "no triple", "-")] += 1
                if ge["relation"] == "DEVELOPS":
                    tmpl[template(ge["evidence_quote"], ge["source_name"], ge["target_name"])]["no triple"] += 1
                continue
            for t in trips:
                same_rel = t["relation_type"] == ge["relation"]
                d = "fwd" if kb.m2g.get(t["source_entity_id"]) == S else "REV"
                st = t.get("mention_level_status") or t["validation_status"]
                dirc[(ge["relation"], f"extracted {t['relation_type']}" if not same_rel else "same relation", d)] += 1
                if ge["relation"] == "DEVELOPS" and same_rel:
                    tmpl[template(ge["evidence_quote"], ge["source_name"], ge["target_name"])][d] += 1
        P("| gold relation | extracted relation | direction | count |")
        P("|---|---|---|---|")
        for k, n in sorted(dirc.items(), key=lambda kv: (kv[0][0], -kv[1])):
            P(f"| {k[0]} | {k[1]} | {k[2]} | {n} |")
        P(f"\n  DEVELOPS by sentence template of the gold evidence quote ({label}); <S>=gold source name or pronoun, <T>=gold target name")
        P("| template | fwd | REV | no triple |")
        P("|---|---|---|---|")
        for t_, cc in sorted(tmpl.items(), key=lambda kv: -sum(kv[1].values())):
            P(f"| {t_} | {cc['fwd']} | {cc['REV']} | {cc['no triple']} |")

        # ---- L4 swap counterfactual
        P(f"\nL4 ({label}) EXPLORATORY, NOT A FIX, SEED ALREADY USED: swap source/target of ontology-rejected triples if the swapped pair satisfies the same relation's rule")
        ctype = {e["entity_id"]: e["entity_type"] for e in kb.entities}
        m2e = {r["mention_id"]: r["canonical_entity_id"] for r in fl.read_jsonl(kbd / "data/processed/mention_to_entity.jsonl")}
        gold_triples = {(e["source"], e["relation"], e["target"]) for e in edges.values()}
        cand = []
        for t in kb.triples:
            if t["validation_status"] != "REJECTED_ONTOLOGY":
                continue
            st, tt = mention[t["source_entity_id"]]["entity_type"], mention[t["target_entity_id"]]["entity_type"]
            swap_ok = ok(tt, t["relation_type"], st)
            gs, gt = kb.m2g.get(t["source_entity_id"]), kb.m2g.get(t["target_entity_id"])
            exact_gold = (gt, t["relation_type"], gs) in gold_triples
            cand.append((t["relationship_id"], t["relation_type"], f"{st}->{tt}", swap_ok, exact_gold, gs, gt))
        n_rej = len(cand)
        sw = [x for x in cand if x[3]]
        P(f"  triples with final status REJECTED_ONTOLOGY: {n_rej}; swap passes the rule: {len(sw)}; of these, swapped triple equals a gold edge (same relation): "
          f"{sum(1 for x in sw if x[4])}; swapped triple matches no gold edge: {sum(1 for x in sw if not x[4])}")
        P("  | relation | mention types (as extracted) | swap allowed | swapped equals a gold edge | count |")
        P("  |---|---|---|---|---|")
        cc = collections.Counter((x[1], x[2], x[3], x[4]) for x in cand)
        for k, n in sorted(cc.items(), key=lambda kv: -kv[1]):
            P("  | " + " | ".join(str(i) for i in k) + f" | {n} |")
        wrong = [x for x in sw if not x[4]]
        for x in wrong:
            P(f"    swap passes but is not a gold edge: {x[0]} {x[1]} {x[2]} gold endpoints {x[5]} -> {x[6]}")
        # edges that would become exact
        newly = set()
        for x in sw:
            if x[4]:
                gs, gt = x[6], x[5]  # after the swap source = old target
                for eid, ge in edges.items():
                    if ge["source"] == gs and ge["target"] == gt and ge["relation"] == x[1] and fate[eid] not in "AE":
                        newly.add(eid)
        fate_after = dict(fate)
        for e in newly:
            fate_after[e] = "A*"
        P(f"  gold edges that would become exact-validated: {len(newly)} {sorted(newly)} (fates before: {dict(collections.Counter(fate[e] for e in newly))})")
        P(f"  exact validated gold edges (A+E) before: {sum(f in 'AE' for f in fate.values())}; after swap (upper bound, ignores the D-fate edges and endpoint splits): "
          f"{sum(f in 'AE' or f == 'A*' for f in fate_after.values())}")
        P("  | group | N | path-complete before | path-complete after swap |")
        P("  |---|---|---|---|")
        for grp in ("hop1", "hop2", "hop3"):
            ids = [q for q, x in questions.items() if x["group"] == grp]
            b = sum(all(fate[e] in "AE" for e in questions[i]["gold_path"]) for i in ids)
            a = sum(all(fate_after[e] in ("A", "E", "A*") for e in questions[i]["gold_path"]) for i in ids)
            P(f"  | {grp} | {len(ids)} | {b} | {a} |")
        changed = [i for i in questions if questions[i]["group"] in ("hop2", "hop3")
                   and not all(fate[e] in "AE" for e in questions[i]["gold_path"]) and all(fate_after[e] in ("A", "E", "A*") for e in questions[i]["gold_path"])]
        P(f"  questions that become path-complete: {sorted(changed)}")
        P(f"  of those, currently succeeding in hybrid: {sum(okq(i,'hybrid') for i in changed)}/{len(changed)}; graph_only: {sum(okq(i,'graph_only') for i in changed)}/{len(changed)}")
        pc = {}
        for grp in ("hop1", "hop2", "hop3"):
            ids = [q for q, x in questions.items() if x["group"] == grp]
            pc[grp] = dict(n=len(ids), before=sum(all(fate[e] in "AE" for e in questions[i]["gold_path"]) for i in ids),
                           after=sum(all(fate_after[e] in ("A", "E", "A*") for e in questions[i]["gold_path"]) for i in ids))
        summary[label] = dict(newly=sorted(newly), changed_questions=sorted(changed), rejected_ontology=n_rej, swap_passes=len(sw),
                              swap_equals_gold=sum(1 for x in sw if x[4]), swap_wrong=len(wrong), path_complete=pc,
                              changed_hybrid_ok=sum(okq(i, "hybrid") for i in changed), changed_graph_only_ok=sum(okq(i, "graph_only") for i in changed),
                              ae_before=sum(f in "AE" for f in fate.values()), ae_after=sum(f in "AE" or f == "A*" for f in fate_after.values()),
                              L2=dict(total=sum(c.values()), ae=sum(n for (f, _), n in c.items() if f in "AE")),
                              dev_template={t_: dict(cc) for t_, cc in tmpl.items()},
                              dev_dir={f"{k[0]}|{k[1]}|{k[2]}": n for k, n in dirc.items() if k[0] == "DEVELOPS"},
                              dir_all={f"{k[0]}|{k[1]}|{k[2]}": n for k, n in dirc.items()})
    (DIAG / "link_tables.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    (DIAG / "link_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
