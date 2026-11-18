#!/usr/bin/env python3
"""Phase 2b-style tables: gold-edge fate, entity splits, alias seeds, retrieval-only evidence recall.

Usage:
  python3 experiments/synth_kg_fix/kb_tables.py --corpus <corpus dir> \
      --run "<label>|<kb dir>|<retrieval results.jsonl>" [--run ...] [--earlier-seed7]

Fate categories used here (mutually exclusive, first match wins) for each gold edge S -R-> T:
  A  validated, exact gold type and direction, both endpoint gold entities map to exactly one canonical entity
  E  validated, exact gold type and direction, but an endpoint gold entity is split over several canonical entities
  B  no exact edge, but a validated edge of another type (or reversed direction) joins the same two gold endpoints
  C  no validated edge between the endpoints; a raw triple between the two endpoint mentions is REJECTED_ONTOLOGY
  D  no validated edge; raw triples between the endpoint mentions exist but were rejected for evidence reasons
     (quote mismatch / endpoint not in evidence / negation / uncertainty)
  F  never extracted: no raw triple joins mentions of the two gold endpoints
Difference from experiments/synth_kg/score.py (score_extraction): score.py reports VALIDATED_CORRECT / REJECTED:<reason> /
EXTRACTED_BUT_DROPPED_IN_RESOLUTION / NEVER_EXTRACTED, matches raw triples by direction or by evidence-quote overlap, and
has no split-endpoint or other-type categories; its counts are printed alongside. Entity-to-gold mapping is the same as
score.py (normalized canonical name or gold alias).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixlib as fl  # noqa: E402

FATES = ["A", "E", "B", "C", "D", "F"]
FATE_LABEL = {
    "A": "validated, exact gold type",
    "E": "validated exact, endpoint on split/duplicate entity",
    "B": "validated under another type / reversed, same endpoints",
    "C": "rejected by ontology",
    "D": "rejected for evidence reasons (not in the requested list)",
    "F": "never extracted",
}


def load_score():
    spec = importlib.util.spec_from_file_location("synth_score", BASE / "experiments/synth_kg/score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KBView:
    def __init__(self, kb_dir: Path, gold: dict, score) -> None:
        self.kb_dir = kb_dir
        self.entities = fl.read_jsonl(kb_dir / "graph/entities.jsonl")
        self.rels = fl.read_jsonl(kb_dir / "graph/relationships.jsonl")
        self.triples = fl.read_jsonl(kb_dir / "data/processed/triples.jsonl")
        self.mentions = fl.read_jsonl(kb_dir / "data/processed/entity_mentions.jsonl")
        self.E = {e["entity_id"]: e for e in self.entities}
        n2g = {}
        for ge in gold["entities"]:
            for n in [ge["canonical_name"]] + ge.get("aliases", []):
                n2g[score.normalize_text(n)] = ge["id"]
        self.c2g = collections.defaultdict(set)
        self.g2c = collections.defaultdict(set)
        for ce in self.entities:
            for n in [ce["name"]] + ce.get("aliases", []):
                gid = n2g.get(score.normalize_text(n))
                if gid:
                    self.c2g[ce["entity_id"]].add(gid)
                    self.g2c[gid].add(ce["entity_id"])
        self.m2g = {}
        for m in self.mentions:
            gid = n2g.get(score.normalize_text(m["surface_name"]))
            if gid:
                self.m2g[m["mention_id"]] = gid
        self.score_ext = score.score_extraction(gold, self.entities, self.rels, self.triples, self.mentions)

    def fate(self, ge: dict):
        S, T, R = ge["source"], ge["target"], ge["relation"]
        exact = other = False
        other_types = []
        for r in self.rels:
            gs, gt = self.c2g.get(r["source_entity_id"], set()), self.c2g.get(r["target_entity_id"], set())
            if S in gs and T in gt and r["relation_type"] == R:
                exact = True
            elif (S in gs and T in gt) or (S in gt and T in gs):
                other = True
                other_types.append(r["relation_type"] + ("" if S in gs else "(reversed)"))
        if exact:
            split = len(self.g2c[S]) > 1 or len(self.g2c[T]) > 1
            return ("E" if split else "A"), other_types
        if other:
            return "B", other_types
        raw = [t for t in self.triples
               if {self.m2g.get(t["source_entity_id"]), self.m2g.get(t["target_entity_id"])} == {S, T}]
        if not raw:
            return "F", []
        statuses = [t["validation_status"] for t in raw]
        if "REJECTED_ONTOLOGY" in statuses:
            return "C", statuses
        return "D", statuses


def recall_table(P, results_path: Path, Q: dict, groups):
    recs = fl.read_jsonl(results_path)
    by_mode = collections.defaultdict(dict)
    for r in recs:
        by_mode[r["mode"]][r["question_id"]] = r
    out = {}
    for mode, d in by_mode.items():
        row, per_q = [], {}
        for g in groups:
            ids = [qid for qid, q in Q.items() if q["group"] == g]
            ok = {qid: fl.all_gold(d[qid]["context_doc_ids"], Q[qid]["gold_supporting_doc_ids"]) for qid in ids}
            err = sum(1 for qid in ids if d[qid].get("error"))
            per_q.update(ok)
            row.append(f"{g} {fl.fmt_rate(sum(ok.values()), len(ids))}" + (f" ERROR={err}" if err else ""))
        out[mode] = per_q
        P(f"   {mode}: " + " | ".join(row))
    return out, by_mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run", action="append", required=True, help="label|kb_dir|results.jsonl")
    ap.add_argument("--earlier-seed7", action="store_true", help="print the recorded seed-7 run numbers for reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    score = load_score()
    corpus = Path(args.corpus)
    gold = json.loads((corpus / "gold_kg.json").read_text())
    G_ENT = {e["id"]: e for e in gold["entities"]}
    G_EDGE = {e["id"]: e for e in gold["edges"]}
    questions = [q for q in fl.read_jsonl(corpus / "questions.jsonl") if q["group"] != "unanswerable"]
    Q = {q["id"]: q for q in questions}
    runs = []
    for spec in args.run:
        label, kb_dir, res = spec.split("|")
        runs.append((label, KBView(Path(kb_dir), gold, score), Path(res) if res else None))

    P("=" * 110)
    P(f"KB TABLES  corpus={corpus}  gold entities={len(gold['entities'])} gold edges={len(gold['edges'])} answerable questions={len(questions)}")
    P("=" * 110)

    # ---------------------------------------------------------------- (i) gold-edge fate
    P("\n(i) GOLD-EDGE FATE (categories defined in the header of experiments/synth_kg_fix/kb_tables.py)")
    fates = {label: {eid: kb.fate(ge) for eid, ge in G_EDGE.items()} for label, kb, _ in runs}
    P("| fate | " + " | ".join(label for label, _, _ in runs) + " |")
    P("|---|" + "---|" * len(runs))
    for f in FATES:
        P(f"| {f} {FATE_LABEL[f]} | " + " | ".join(str(sum(1 for v in fates[label].values() if v[0] == f)) for label, _, _ in runs) + " |")
    P("| total | " + " | ".join(str(len(fates[label])) for label, _, _ in runs) + " |")
    P("| A+E (exact type validated) | " + " | ".join(str(sum(1 for v in fates[label].values() if v[0] in 'AE')) for label, _, _ in runs) + " |")
    by_rel = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for label, _, _ in runs:
        for eid, (f, _) in fates[label].items():
            by_rel[G_EDGE[eid]["relation"]][label][f] += 1
    P("\n   By gold relation type (fate counts A/E/B/C/D/F):")
    for rel in sorted(by_rel):
        P(f"   {rel:11s} " + "  ||  ".join(f"{label}: " + "/".join(str(by_rel[rel][label][f]) for f in FATES) for label, _, _ in runs))
    for label, kb, _ in runs:
        P(f"   score.py fate counts [{label}]: {kb.score_ext['fate_breakdown']['counts']}")
        P(f"   score.py relationship scoring [{label}]: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in kb.score_ext['relationship_scoring'].items()}))
    if len(runs) >= 2:
        a, b = runs[0][0], runs[-1][0]
        trans = collections.Counter((fates[a][e][0], fates[b][e][0]) for e in G_EDGE)
        P(f"\n   Fate transitions {a} -> {b} (only changes): " + ", ".join(f"{x}->{y}: {n}" for (x, y), n in sorted(trans.items()) if x != y))
        for e in sorted(G_EDGE):
            if fates[a][e][0] != fates[b][e][0]:
                ge = G_EDGE[e]
                P(f"      {e} {ge['source_name']} -{ge['relation']}-> {ge['target_name']} [{ge['supporting_doc_id']}]: {fates[a][e][0]} {fates[a][e][1]} -> {fates[b][e][0]} {fates[b][e][1]}")

    # ---------------------------------------------------------------- (ii) entities
    P("\n(ii) CANONICAL ENTITIES")
    P("| quantity | " + " | ".join(label for label, _, _ in runs) + " |")
    P("|---|" + "---|" * len(runs))
    rows = [
        ("canonical entities", lambda kb: len(kb.entities)),
        ("gold entities with >=1 canonical entity", lambda kb: sum(1 for g in G_ENT if kb.g2c[g])),
        ("gold entities split into >1 canonical entity", lambda kb: sum(1 for g in G_ENT if len(kb.g2c[g]) > 1)),
        ("canonical entities mapping to >1 gold entity (false merge)", lambda kb: sum(1 for c in kb.c2g.values() if len(c) > 1)),
        ("extra canonical entities (map to no gold entity)", lambda kb: sum(1 for e in kb.entities if not kb.c2g.get(e["entity_id"]))),
        ("graph relationships (validated, aggregated)", lambda kb: len(kb.rels)),
        ("raw triples VALIDATED", lambda kb: sum(1 for t in kb.triples if t["validation_status"] == "VALIDATED")),
        ("raw triples REJECTED_ONTOLOGY", lambda kb: sum(1 for t in kb.triples if t["validation_status"] == "REJECTED_ONTOLOGY")),
        ("validated graph edges matching a gold edge exactly (score.py)", lambda kb: kb.score_ext["relationship_scoring"]["exact_matched_validated"]),
        ("validated graph edges matching NO gold edge exactly", lambda kb: len(kb.rels) - kb.score_ext["relationship_scoring"]["exact_matched_validated"]),
    ]
    for name, fn in rows:
        P(f"| {name} | " + " | ".join(str(fn(kb)) for _, kb, _ in runs) + " |")
    for label, kb, _ in runs:
        split = {g: sorted(c) for g, c in kb.g2c.items() if len(c) > 1}
        if split:
            P(f"   split gold entities [{label}]: " + "; ".join(f"{G_ENT[g]['canonical_name']} -> " + ",".join(f"{c}:{kb.E[c]['entity_type']}" for c in cs) for g, cs in sorted(split.items())))
        typ_wrong = [(G_ENT[g]["canonical_name"], G_ENT[g]["type"], kb.E[c]["entity_type"]) for g, cs in kb.g2c.items() if len(cs) == 1 for c in cs if kb.E[c]["entity_type"] != G_ENT[g]["type"]]
        P(f"   unsplit gold entities whose canonical type differs from the gold type [{label}]: {len(typ_wrong)} {typ_wrong[:12]}")

    # ---------------------------------------------------------------- (iv) recall
    groups = fl.GROUPS
    P("\n(iv) RETRIEVAL-ONLY ALL-GOLD EVIDENCE RECALL AT K=4 (cached query plans, no answer generation; Wilson 95%)")
    if args.earlier_seed7:
        P("   earlier seed-7 run records (experiments/synth_kg/runs, recomputed in step0_reverify.py): "
          "hybrid hop1 100.0% hop2 56.7% hop3 3.3% alias 40.0%; graph_only hop1 96.7% hop2 53.3% hop3 3.3% alias 0.0%")
    per_run = {}
    for label, kb, res in runs:
        if res is None or not res.exists():
            continue
        P(f"  [{label}]  ({res})")
        per_run[label] = recall_table(P, res, Q, groups)
        plan_err = sorted({r["question_id"] for r in fl.read_jsonl(res) if r.get("plan_error")})
        q_err = sorted({(r["mode"], r["question_id"]) for r in fl.read_jsonl(res) if r.get("error")})
        P(f"   planner failures (cached): {len(plan_err)} {plan_err[:10]} | query ERRORs (counted as not retrieved, listed separately): {len(q_err)} {q_err[:10]}")
    labels = list(per_run)
    if len(labels) >= 2:
        a, b = labels[0], labels[-1]
        for mode in ("hybrid", "graph_only"):
            if mode in per_run[a][0] and mode in per_run[b][0]:
                for g in groups:
                    ids = [qid for qid in Q if Q[qid]["group"] == g]
                    gained = [q for q in ids if per_run[b][0][mode][q] and not per_run[a][0][mode][q]]
                    lost = [q for q in ids if per_run[a][0][mode][q] and not per_run[b][0][mode][q]]
                    P(f"   {mode} {g}: {a} -> {b}: gained {len(gained)} {gained} | lost {len(lost)} {lost} | paired sign test p={fl.sign_test(len(gained), len(lost)):.4f} (descriptive)")

    # ---------------------------------------------------------------- (iii) alias seeds
    P("\n(iii) ALIAS QUESTIONS: SEEDS RESOLVED BY THE GRAPH ROUTE (graph_only mode records)")
    alias_q = [q for q in questions if q["group"] == "alias"]
    summary = {}
    for label, kb, res in runs:
        if label not in per_run or "graph_only" not in per_run[label][1]:
            continue
        d = per_run[label][1]["graph_only"]
        n_any = n_gold = n_gold_only = 0
        P(f"  [{label}]")
        for q in alias_q:
            first = G_EDGE[q["gold_path"][0]]["source"]
            r = d[q["id"]]
            ids = r.get("resolved_seed_ids", [])
            gold_hit = [i for i in ids if first in kb.c2g.get(i, set())]
            n_any += bool(ids)
            n_gold += bool(gold_hit)
            n_gold_only += bool(gold_hit) and len(gold_hit) == len(ids)
            specs = [(s["name"], s.get("expected_type")) for s in r.get("seed_specs", [])]
            P(f"    {q['id']} gold={G_ENT[first]['canonical_name']!r} planner_seeds={specs} resolved={[(s['id'], s['name'], s['type']) for s in r.get('resolved_seeds', [])]} "
              f"gold_entity_resolved={'YES' if gold_hit else 'NO'}" + (f" ERROR={r['error']}" if r.get("error") else ""))
        summary[label] = (n_any, n_gold, n_gold_only)
    P("  | run | questions with >=1 resolved seed | gold entity among resolved seeds | resolved seeds are only the gold entity |")
    P("  |---|---|---|---|")
    for label, (x, y, z) in summary.items():
        P(f"  | {label} | {x}/15 | {y}/15 | {z}/15 |")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
