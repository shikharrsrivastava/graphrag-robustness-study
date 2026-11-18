#!/usr/bin/env python3
"""Diagnosis Part 2: why are gold DEVELOPS edges rejected by the ontology?

Reads (no engine calls, no LLM): the KB files of kb_seed11 / kb_dev (/ kb_before for reference), the gold KG of the
corpus, configs/domains/company.yaml (ontology rules), data/processed/{triples,entity_mentions,mention_to_entity}.jsonl.
Writes part2_tables.txt, part2_edges_seed{11,7}.json.

Per gold DEVELOPS edge it records: gold endpoint types; the entity types the extractor assigned to the mentions of the
two endpoints in the supporting document; the raw triples joining those mentions (relation text, direction, status at
mention level and after revalidation, evidence quote); the ontology rule for the relation actually extracted; and the
result of three counterfactual type checks (gold types, mention types, canonical types) computed with the ontology.

Cause rules (deterministic, applied in this order to each gold edge whose fate is not A/E):
  NOT_EXTRACTED   no raw triple joins mentions of the two gold endpoints                         -> (d)
  WRONG_RELATION  a triple joins them but with a relation type other than DEVELOPS                -> (d) relation choice
  REVERSED        a DEVELOPS triple joins them with source/target swapped w.r.t. gold             -> (d) direction
  EVIDENCE_REJECT every joining DEVELOPS triple is rejected for quote/endpoint/negation reasons    -> (d) evidence
  MENTION_TYPE    a DEVELOPS triple in gold direction rejected by ontology while the gold types
                  satisfy the DEVELOPS rule and the mention types do not                           -> (a)
  ONTOLOGY_NARROW a DEVELOPS triple in gold direction rejected by ontology although the mention
                  types equal the gold types, or the gold types themselves violate the rule        -> (b)
  Revalidation reach (c): among triples rejected by ontology at mention level, whether the CANONICAL endpoint
  types would satisfy the rule, and whether the final status changed.
"""
from __future__ import annotations

import collections
import json
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


ONT = yaml.safe_load((ROOT / "configs/domains/company.yaml").read_text())["ontology"]
RULES = ONT["relationship_types"]


def ok(src_t, rel, tgt_t):
    r = RULES.get(rel)
    if r is None:
        return False
    return src_t in r["allowed_source_types"] and tgt_t in r["allowed_target_types"]


def analyse(label, kb_dir, corpus):
    score = kt.load_score()
    gold = json.loads((corpus / "gold_kg.json").read_text())
    G_ENT = {e["id"]: e for e in gold["entities"]}
    kb = kt.KBView(Path(kb_dir), gold, score)
    m2e = {r["mention_id"]: r["canonical_entity_id"] if "canonical_entity_id" in r else r.get("entity_id") for r in fl.read_jsonl(Path(kb_dir) / "data/processed/mention_to_entity.jsonl")}
    mention = {m["mention_id"]: m for m in kb.mentions}
    ctype = {e["entity_id"]: e["entity_type"] for e in kb.entities}
    dev = [e for e in gold["edges"] if e["relation"] == "DEVELOPS"]
    P("=" * 118)
    P(f"{label}: corpus={corpus.name} KB={kb_dir}")
    P(f"ontology rule DEVELOPS: source in {RULES['DEVELOPS']['allowed_source_types']}, target in {RULES['DEVELOPS']['allowed_target_types']}")
    pairs = collections.Counter((e["source_type"], e["target_type"]) for e in dev)
    P(f"gold DEVELOPS edges: {len(dev)}; gold (source type, target type) pairs: {dict(pairs)}; "
      f"pairs allowed by the DEVELOPS rule: {sum(c for (s, t), c in pairs.items() if ok(s, 'DEVELOPS', t))}/{len(dev)}")
    rows = []
    for ge in dev:
        S, T, d = ge["source"], ge["target"], ge["supporting_doc_id"]
        did = fl.norm_doc_id(d)
        fate, other = kb.fate(ge)
        in_doc = lambda mid: fl.norm_doc_id(mention[mid]["source_doc_id"]) == did
        ments = {S: [mid for mid, g in kb.m2g.items() if g == S and in_doc(mid)], T: [mid for mid, g in kb.m2g.items() if g == T and in_doc(mid)]}
        trip = []
        for t in kb.triples:
            gs, gt = kb.m2g.get(t["source_entity_id"]), kb.m2g.get(t["target_entity_id"])
            if {gs, gt} == {S, T} and any(fl.norm_doc_id(x) == did for x in t["source_doc_ids"]):
                st, tt = mention[t["source_entity_id"]]["entity_type"], mention[t["target_entity_id"]]["entity_type"]
                cs = ctype.get(m2e.get(t["source_entity_id"]))
                ct = ctype.get(m2e.get(t["target_entity_id"]))
                trip.append(dict(
                    relationship_id=t["relationship_id"], relation=t["relation_type"], same_direction=(gs == S),
                    src_mention_type=st, tgt_mention_type=tt, src_canonical_type=cs, tgt_canonical_type=ct,
                    status_mention_level=t.get("mention_level_status") or t["validation_status"], status_final=t["validation_status"],
                    quote=(t["evidence_quotes"] or [""])[0],
                    ontology_ok_mention_types=ok(st, t["relation_type"], tt), ontology_ok_canonical_types=(None if cs is None or ct is None else ok(cs, t["relation_type"], ct)),
                ))
        gold_types_ok = ok(ge["source_type"], "DEVELOPS", ge["target_type"])
        if fate in ("A", "E"):
            cause = "OK (validated exact)"
        elif not trip:
            cause = "NOT_EXTRACTED"
        else:
            dev_trip = [t for t in trip if t["relation"] == "DEVELOPS"]
            if not dev_trip:
                cause = "WRONG_RELATION"
            else:
                fwd = [t for t in dev_trip if t["same_direction"]]
                if not fwd:
                    cause = "REVERSED"
                else:
                    onto = [t for t in fwd if t["status_mention_level"] == "REJECTED_ONTOLOGY"]
                    if not onto:
                        cause = "EVIDENCE_REJECT" if all(t["status_mention_level"] != "VALIDATED" for t in fwd) else "VALIDATED_BUT_NOT_IN_GRAPH"
                    else:
                        t0 = onto[0]
                        if gold_types_ok and (t0["src_mention_type"], t0["tgt_mention_type"]) != (ge["source_type"], ge["target_type"]):
                            cause = "MENTION_TYPE"
                        else:
                            cause = "ONTOLOGY_NARROW"
        rows.append(dict(edge=ge["id"], doc=d, source=ge["source_name"], source_type=ge["source_type"], target=ge["target_name"], target_type=ge["target_type"],
                         fate=fate, other_edge_types=other, cause=cause, gold_types_satisfy_rule=gold_types_ok,
                         mention_types={g: sorted({mention[m]["entity_type"] for m in ms}) for g, ms in ments.items()}, triples=trip,
                         evidence_quote=ge["evidence_quote"]))
    return kb, rows


def report(label, rows):
    P(f"\nTABLE A ({label}): per gold DEVELOPS edge")
    P("| edge | doc | gold source (type) | gold target (type) | fate | mention types source / target | extracted relation (dir, mention types, status ML -> final) | cause |")
    P("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mt = f"{r['mention_types'][list(r['mention_types'])[0]]} / {r['mention_types'][list(r['mention_types'])[1]]}"
        tr = "; ".join(f"{t['relation']} {'fwd' if t['same_direction'] else 'REV'} ({t['src_mention_type']}->{t['tgt_mention_type']}) {t['status_mention_level']}->{t['status_final']}" for t in r["triples"]) or "none"
        P(f"| {r['edge']} | {r['doc']} | {r['source']} ({r['source_type']}) | {r['target']} ({r['target_type']}) | {r['fate']} | {mt} | {tr} | {r['cause']} |")
    P(f"\nCAUSE COUNTS ({label}): " + str(dict(collections.Counter(r["cause"] for r in rows))))
    P(f"FATE COUNTS ({label}): " + str(dict(collections.Counter(r["fate"] for r in rows))))
    return rows


def pair_tables(label, kb, rows):
    # extracted type pairs for triples that join the endpoints of a gold DEVELOPS edge
    P(f"\nTABLE B ({label}): extracted (relation, source mention type -> target mention type) for triples joining the endpoints of a gold DEVELOPS edge")
    c = collections.Counter()
    for r in rows:
        for t in r["triples"]:
            c[(t["relation"], "fwd" if t["same_direction"] else "REV", t["src_mention_type"], t["tgt_mention_type"], t["ontology_ok_mention_types"], t["status_mention_level"], t["status_final"])] += 1
    P("| relation | direction vs gold | source mention type | target mention type | pair allowed by that relation's rule | status at mention level | final status | count |")
    P("|---|---|---|---|---|---|---|---|")
    for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
        P("| " + " | ".join(str(x) for x in k) + f" | {n} |")
    P("allowed pairs per relation in configs/domains/company.yaml: " + "; ".join(
        f"{rel}: {v['allowed_source_types']} -> {v['allowed_target_types']}" for rel, v in RULES.items()))
    # global: all REJECTED_ONTOLOGY triples (mention level) in this KB
    P(f"\nTABLE C ({label}): ALL triples whose mention-level status is REJECTED_ONTOLOGY: relation, mention type pair, canonical type pair, final status")
    cc = collections.Counter()
    mention = {m["mention_id"]: m for m in kb.mentions}
    m2e = {r["mention_id"]: r.get("canonical_entity_id") or r.get("entity_id") for r in fl.read_jsonl(Path(kb.kb_dir) / "data/processed/mention_to_entity.jsonl")}
    ctype = {e["entity_id"]: e["entity_type"] for e in kb.entities}
    for t in kb.triples:
        ml = t.get("mention_level_status") or t["validation_status"]
        if ml != "REJECTED_ONTOLOGY":
            continue
        st, tt = mention[t["source_entity_id"]]["entity_type"], mention[t["target_entity_id"]]["entity_type"]
        cs, ct = ctype.get(m2e.get(t["source_entity_id"])), ctype.get(m2e.get(t["target_entity_id"]))
        cok = None if cs is None or ct is None else ok(cs, t["relation_type"], ct)
        cc[(t["relation_type"], f"{st}->{tt}", f"{cs}->{ct}", f"canonical pair allowed={cok}", t["validation_status"])] += 1
    P("| relation | mention types | canonical types | canonical pair allowed | final status | count |")
    P("|---|---|---|---|---|---|")
    for k, n in sorted(cc.items(), key=lambda kv: -kv[1]):
        P("| " + " | ".join(str(x) for x in k) + f" | {n} |")
    tot = sum(cc.values())
    revived = sum(n for k, n in cc.items() if k[4] == "VALIDATED")
    could = sum(n for k, n in cc.items() if "allowed=True" in k[3])
    P(f"REJECTED_ONTOLOGY at mention level: {tot}; canonical pair would be allowed: {could}; final status VALIDATED (revived by revalidation): {revived}; still rejected: {tot - revived}")
    rep = Path(kb.kb_dir) / "data/processed/resolution_report.json"
    if rep.exists():
        rv = json.loads(rep.read_text()).get("revalidation")
        if rv:
            P(f"resolution_report.json revalidation summary: recomputed={rv['recomputed']} validated_to_rejected={rv['validated_to_rejected_ontology']} rejected_to_validated={rv['rejected_ontology_to_validated']}")


def main():
    out = {}
    for label, kbd, corpus in [("seed 11 (kb_seed11)", FIX / "kb_seed11/synth_company_kb", FIX / "corpus_seed11"),
                               ("seed 7 (kb_dev, after fixes)", FIX / "kb_dev/synth_company_kb", ROOT / "experiments/synth_kg/corpus_final"),
                               ("seed 7 (kb_before, original HEAD KB)", FIX / "kb_before/synth_company_kb", ROOT / "experiments/synth_kg/corpus_final")]:
        try:
            kb, rows = analyse(label, kbd, corpus)
        except Exception as exc:  # report, do not hide
            P(f"{label}: could not analyse: {type(exc).__name__}: {exc}")
            continue
        report(label, rows)
        pair_tables(label, kb, rows)
        out[label] = rows
    (DIAG / "part2_edges.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (DIAG / "part2_tables.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
