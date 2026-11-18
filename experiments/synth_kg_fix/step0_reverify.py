#!/usr/bin/env python3
"""Phase 0 re-verification (read-only on experiments/synth_kg and experiments/synth_kg_audit).

Every number printed here is computed from files on disk. Output: experiments/synth_kg_fix/step0_reverify_data.txt
"""
from __future__ import annotations

import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import yaml

BASE = Path(__file__).resolve().parents[2]
FIX = BASE / "experiments" / "synth_kg_fix"
SK = BASE / "experiments" / "synth_kg"
AUD = BASE / "experiments" / "synth_kg_audit"
KB = SK / "kb" / "synth_company_kb"
sys.path.insert(0, str(FIX))
import fixlib as fl  # noqa: E402

OUT = FIX / "step0_reverify_data.txt"
_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


def load_score_module():
    spec = importlib.util.spec_from_file_location("synth_score", SK / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gold = json.loads((SK / "corpus_final" / "gold_kg.json").read_text())
G_ENT = {e["id"]: e for e in gold["entities"]}
G_EDGE = {e["id"]: e for e in gold["edges"]}
questions = fl.read_jsonl(SK / "corpus_final" / "questions.jsonl")
Q = {q["id"]: q for q in questions}
docs = fl.load_docs(SK / "corpus_final")
triples = fl.read_jsonl(KB / "data/processed/triples.jsonl")
mentions = fl.read_jsonl(KB / "data/processed/entity_mentions.jsonl")
M = {m["mention_id"]: m for m in mentions}
entities = fl.read_jsonl(KB / "graph/entities.jsonl")
E = {e["entity_id"]: e for e in entities}
rels = fl.read_jsonl(KB / "graph/relationships.jsonl")
m2e = {r["mention_id"]: r["canonical_entity_id"] for r in fl.read_jsonl(KB / "data/processed/mention_to_entity.jsonl")}
RUNS = {
    "vector_only": SK / "runs/20261116_035101_vector_only",
    "hybrid": SK / "runs/20261116_045328_hybrid",
    "graph_only": SK / "runs/20261116_062038_graph_only",
    "community_aware": SK / "runs/20261116_074405_community_aware",
}

name2gold = {}
for ge in gold["entities"]:
    for n in [ge["canonical_name"]] + ge.get("aliases", []):
        name2gold[fl.norm_name(n)] = ge["id"]
canon2gold = {fl.norm_name(ge["canonical_name"]): ge["id"] for ge in gold["entities"]}


# ============================================================================= B1
def b1():
    P("=" * 100)
    P("B1. RELATIONSHIP PROMPT ACTUALLY LOADED; COLLAPSING OF RELATION TYPES")
    snap = yaml.safe_load((KB / "config_snapshot.yaml").read_text())
    man = json.loads((KB / "manifest.json").read_text())
    P("KB config_snapshot prompt_versions:", snap.get("prompt_versions"))
    P("KB config_snapshot prompts_dir:", snap.get("prompts_dir"))
    P("KB manifest prompt_versions:", man.get("prompt_versions"))
    for name, path in RUNS.items():
        rs = yaml.safe_load((path / "config_snapshot.yaml").read_text())
        P(f"run {name}: prompt_versions.relationship_extraction={rs['prompt_versions'].get('relationship_extraction')}"
          f" graph_context_mode={rs['retrieval'].get('graph_context_mode')!r}")
    import subprocess
    head_engine = subprocess.run(["git", "show", "HEAD:src/engine.py"], capture_output=True, text=True, cwd=BASE).stdout.splitlines()
    for i, line in enumerate(head_engine, 1):
        if "OllamaRelationshipExtractor(" in line:
            P(f"HEAD src/engine.py:{i}-{i+4}:")
            for j in range(i - 1, i + 4):
                P("    " + head_engine[j])
    head_rx = subprocess.run(["git", "show", "HEAD:src/extraction/relation_extractor.py"], capture_output=True, text=True, cwd=BASE).stdout.splitlines()
    for i, line in enumerate(head_rx, 1):
        if "prompt_template_path" in line:
            P(f"HEAD src/extraction/relation_extractor.py:{i}: {line.strip()}")
    pf = BASE / "prompts" / "relationship_extraction.txt"
    import datetime, hashlib
    P("prompts/relationship_extraction.txt mtime:", datetime.datetime.fromtimestamp(pf.stat().st_mtime).isoformat(timespec="seconds"),
      "sha256:", hashlib.sha256(pf.read_bytes()).hexdigest()[:16],
      "| equals HEAD blob:", subprocess.run(["git", "show", "HEAD:prompts/relationship_extraction.txt"], capture_output=True, cwd=BASE).stdout == pf.read_bytes())
    P("KB triples.jsonl mtime:", datetime.datetime.fromtimestamp((KB / 'data/processed/triples.jsonl').stat().st_mtime).isoformat(timespec="seconds"))
    P("NOTE: the label prompt_versions.relationship_extraction='v1.1' is metadata only at HEAD; no code selects a file from it "
      "(grep for 'prompt_versions' in HEAD src/ shows engine.py passing it to the manifest and reading answer_generation only).")

    P("\nRaw triples in triples.jsonl:", len(triples), dict(collections.Counter(t["validation_status"] for t in triples)))
    by_pair = collections.defaultdict(set)
    by_upair = collections.defaultdict(set)
    for t in triples:
        by_pair[(t["source_entity_id"], t["target_entity_id"])].add(t["relation_type"])
        by_upair[frozenset([t["source_entity_id"], t["target_entity_id"]])].add(t["relation_type"])
    multi_d = {k: v for k, v in by_pair.items() if len(v) > 1}
    multi_u = {k: v for k, v in by_upair.items() if len(v) > 1}
    P(f"Distinct ordered mention pairs: {len(by_pair)}; with >1 relation type: {len(multi_d)}")
    P(f"Distinct unordered mention pairs: {len(by_upair)}; with >1 relation type: {len(multi_u)}")
    for k, v in multi_u.items():
        a, b = sorted(k)
        P(f"   {a} ({M[a]['surface_name']}) <-> {b} ({M[b]['surface_name']}): {sorted(v)}")
        for t in triples:
            if frozenset([t["source_entity_id"], t["target_entity_id"]]) == k:
                P(f"      {t['relationship_id']} {M[t['source_entity_id']]['surface_name']} -{t['relation_type']}-> {M[t['target_entity_id']]['surface_name']} [{t['validation_status']}]")
    dup_same = sum(1 for k in by_pair if sum(1 for t in triples if (t["source_entity_id"], t["target_entity_id"]) == k) > 1)
    P(f"Ordered mention pairs carrying more than one raw triple (any type): {dup_same}")
    P("Graph relationships.jsonl:", len(rels), "| distinct (src,type,tgt):", len({(r['source_entity_id'], r['relation_type'], r['target_entity_id']) for r in rels}),
      "| canonical pairs with >1 type:", sum(1 for v in collections.Counter((r['source_entity_id'], r['target_entity_id']) for r in rels).values() if v > 1))


# ============================================================================= B2
def alias_string_for(q):
    """The surface string in the question that is a gold alias (not the canonical name) of the first gold-path source."""
    first = G_EDGE[q["gold_path"][0]]["source"]
    ge = G_ENT[first]
    hits = [a for a in ge.get("aliases", []) if re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", q["question"])]
    hits.sort(key=len, reverse=True)
    return ge, (hits[0] if hits else None)


def b2():
    P("\n" + "=" * 100)
    P("B2. Q_0091 / Q_0094 AND THE 15 ALIAS STRINGS")
    for qid in ("Q_0091", "Q_0094"):
        q = Q[qid]
        P(f"\n{qid}: {json.dumps(q)}")
        ge, alias = alias_string_for(q)
        P(f"  gold_kg entity of first gold-path source: {json.dumps(ge)}")
        P(f"  alias string used in the question: {alias!r}")
        given, surname = ge["canonical_name"].split()[0], ge["canonical_name"].split()[-1]
        for label, s in [("full name", ge["canonical_name"]), ("given name", given), ("surname", surname), ("alias string", alias)]:
            hit = [d for d, t in docs.items() if s in t]
            P(f"  docs containing {label} {s!r}: {hit if hit else 'NONE'}")
        kb_hits = [(e["entity_id"], e["name"], e["entity_type"]) for e in entities if surname.lower() in e["name"].lower()]
        P(f"  KB canonical entities containing surname: {kb_hits}")
    for wrong in ("Galthor Nebikvane", "Lorik Sarkithtex", "Galthor", "Lorik"):
        in_docs = [d for d, t in docs.items() if wrong in t]
        in_gold = [g["id"] for g in gold["entities"] if wrong in g["canonical_name"] or any(wrong in a for a in g.get("aliases", []))]
        in_q = [q["id"] for q in questions if wrong in json.dumps(q)]
        P(f"  name from docs/NEXT_TASK.md {wrong!r}: docs={in_docs or 'NONE'} gold_kg={in_gold or 'NONE'} questions={in_q or 'NONE'}")
    src = (AUD / "followup/part_c_alias_audit.py").read_text().splitlines()
    for i, line in enumerate(src, 1):
        if "Galthor" in line or "Lorik" in line:
            P(f"  origin of the wrong names: experiments/synth_kg_audit/followup/part_c_alias_audit.py:{i}: {line.strip()}")

    P("\nRecount over the 15 alias questions (alias string = gold alias of the first gold-path source that occurs in the question):")
    none = 0
    for q in [x for x in questions if x["group"] == "alias"]:
        ge, alias = alias_string_for(q)
        hit = [d for d, t in docs.items() if alias and alias in t]
        word_hit = [d for d, t in docs.items() if alias and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", t)]
        none += (len(hit) == 0)
        P(f"  {q['id']} alias={alias!r} gold={ge['canonical_name']!r} ({ge['type']}) docs_with_alias_string={hit or 'NONE'} whole_word={word_hit or 'NONE'}")
    P(f"Alias strings occurring in NO document: {none} of 15")


# ============================================================================= B3
def b3():
    P("\n" + "=" * 100)
    P("B3. B4 DIFFERENCE (audit_part2_baselines.py vs followup/part_f_iterative_baselines.py), hop2")
    emb = np.load(AUD / "dense_embeddings.npz")
    doc_ids, q_ids = list(emb["doc_ids"]), list(emb["q_ids"])
    doc_mat, q_mat = emb["doc_mat"], emb["q_mat"]
    P("doc vector norms min/max:", float(np.linalg.norm(doc_mat, axis=1).min()), float(np.linalg.norm(doc_mat, axis=1).max()),
      "| question vector norms min/max:", float(np.linalg.norm(q_mat, axis=1).min()), float(np.linalg.norm(q_mat, axis=1).max()),
      "(audit script re-normalizes, followup does not; both are unit norm so rankings agree)")
    stripped = {d: t.strip() for d, t in docs.items()}
    P("documents whose text changes under .strip():", sum(1 for d in docs if docs[d] != stripped[d]))
    res = {}
    variants = {
        "audit (no question-name removal)": lambda q, r, s: fl.b4_audit(q, r, s, doc_ids, docs),
        "followup (question names removed)": lambda q, r, s: fl.b4_followup(q, r, s, doc_ids, docs),
    }
    for vname, fn in variants.items():
        ok = {}
        for i, qid in enumerate(q_ids):
            q = Q[qid]
            ranked, sims = fl.dense_ranking(doc_ids, doc_mat, q_mat[i])
            ok[qid] = (fl.all_gold(fn(q["question"], ranked, sims), q["gold_supporting_doc_ids"]), fn(q["question"], ranked, sims))
        res[vname] = ok
        k = sum(1 for qid in q_ids if Q[qid]["group"] == "hop2" and ok[qid][0])
        P(f"  B4 {vname}: hop2 {fl.fmt_rate(k, 30)}")
    a, f = res["audit (no question-name removal)"], res["followup (question names removed)"]
    P("  Per-question differences on hop2:")
    for qid in q_ids:
        if Q[qid]["group"] == "hop2" and a[qid][0] != f[qid][0]:
            q = Q[qid]
            i = q_ids.index(qid)
            ranked, sims = fl.dense_ranking(doc_ids, doc_mat, q_mat[i])
            seeds = ranked[:2]
            names_all = set()
            for s in seeds:
                names_all |= {m for m in fl.NAME_REGEX.findall(docs[s])}
            removed = sorted(n for n in names_all if n.lower() in q["question"].lower())
            P(f"    {qid}: audit={a[qid][0]} docs={a[qid][1]} | followup={f[qid][0]} docs={f[qid][1]} | gold={q['gold_supporting_doc_ids']}")
            P(f"       question: {q['question']}")
            P(f"       seeds={seeds}; names removed by the question filter: {removed}")
    P("  Stated in the two reports: audit_output.txt B4 hop2 and followup/output.txt B4 hop2:")
    for path, pat in [(AUD / "audit_output.txt", "B4_name_expansion_top4"), (AUD / "followup/output.txt", "B4_name_exp_top4")]:
        for line in path.read_text().splitlines():
            if pat in line and "|" in line:
                P(f"    {path.name}: {line.strip()[:200]}")
                break
    P("  Code-level differences (line refs):")
    P("    audit_part2_baselines.py:139-146  names = all capitalized runs of the 2 seed docs, filter len>1, NO question filter")
    P("    part_f_iterative_baselines.py:92-98 names exclude any run that is a case-insensitive substring of the question")
    P("    seed count (2), regex, candidate ranking (cosine to question), backfill from dense ranking: identical")
    P("    tie handling: both use a stable sort on cosine only; no explicit tie rule in either")
    return doc_ids, q_ids, doc_mat, q_mat


# ============================================================================= B4
def b4(score):
    P("\n" + "=" * 100)
    P("B4. 'VALIDATED PATH EXISTS' (experiments/synth_kg/score.py:350-466 compute_path_coverage)")
    P("Definition: undirected, untyped BFS over validated canonical relationships; start = canonical entities whose name/alias")
    P("maps to the gold source of gold_path[0]; target = canonical entities mapping to the gold answer; flag = shortest path <= hop_count.")
    cov = score.compute_path_coverage(questions, entities, rels, gold)
    ext = score.score_extraction(gold, entities, rels, triples, mentions)
    fates = ext["fate_breakdown"]["edge_fates"]
    stored = json.loads((SK / "evaluation_results.json").read_text())

    def find_key(o, key, path=""):
        out = []
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    out.append((path + "/" + k, v))
                out += find_key(v, key, path + "/" + k)
        return out
    stored_cov = find_key(stored, "question_path_coverage")
    P("stored evaluation_results.json question_path_coverage found at:", [p for p, _ in stored_cov])
    cid2gold = collections.defaultdict(set)
    for ce in entities:
        for n in [ce["name"]] + ce.get("aliases", []):
            if score.normalize_text(n) in {score.normalize_text(k) for k in []}:
                pass
    n2g = {}
    for ge in gold["entities"]:
        for n in [ge["canonical_name"]] + ge.get("aliases", []):
            n2g[score.normalize_text(n)] = ge["id"]
    for ce in entities:
        for n in [ce["name"]] + ce.get("aliases", []):
            if score.normalize_text(n) in n2g:
                cid2gold[ce["entity_id"]].add(n2g[score.normalize_text(n)])
    adj = collections.defaultdict(set)
    edge_lookup = collections.defaultdict(list)
    for r in rels:
        adj[r["source_entity_id"]].add(r["target_entity_id"])
        adj[r["target_entity_id"]].add(r["source_entity_id"])
        edge_lookup[frozenset([r["source_entity_id"], r["target_entity_id"]])].append(r)

    def bfs_path(starts, targets):
        queue = collections.deque([(s, [s]) for s in sorted(starts) if s in adj])
        seen = {s for s, _ in queue}
        while queue:
            cur, path = queue.popleft()
            if cur in targets:
                return path
            for nb in sorted(adj[cur]):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    for qid in ("Q_0062", "Q_0063", "Q_0073", "Q_0084"):
        q = Q[qid]
        P(f"\n{qid} ({q['group']}, hop_count={q['hop_count']}): {q['question']}")
        st = [v.get(qid) for _, v in stored_cov]
        P(f"  flag recomputed with score.compute_path_coverage: {cov['question_path_coverage'][qid]} | stored: {st}")
        for eid in q["gold_path"]:
            ge = G_EDGE[eid]
            P(f"  gold {eid}: {ge['source_name']} ({ge['source']},{ge['source_type']}) -{ge['relation']}-> {ge['target_name']} ({ge['target']},{ge['target_type']}) [{ge['supporting_doc_id']}] fate(score.py)={fates[eid]['fate']}")
        first = G_EDGE[q["gold_path"][0]]["source"]
        tg = {n2g[score.normalize_text(n)] for n in [q["gold_answer"]] + q.get("gold_answer_aliases", []) if score.normalize_text(n) in n2g}
        starts = {c for c, g in cid2gold.items() if first in g}
        targets = {c for c, g in cid2gold.items() if g & tg}
        P(f"  start canonical ids: {[(c, E[c]['name'], E[c]['entity_type']) for c in sorted(starts)]}")
        P(f"  target canonical ids: {[(c, E[c]['name'], E[c]['entity_type']) for c in sorted(targets)]}")
        path = bfs_path(starts, targets)
        if path is None:
            P("  no undirected path")
            continue
        P(f"  undirected shortest path ({len(path)-1} edges): " + " -- ".join(f"{c}:{E[c]['name']}" for c in path))
        gold_pairs = {frozenset([G_EDGE[e]["source"], G_EDGE[e]["target"]]): e for e in q["gold_path"]}
        for a, b in zip(path, path[1:]):
            for r in edge_lookup[frozenset([a, b])]:
                ga, gb = cid2gold.get(r["source_entity_id"], set()), cid2gold.get(r["target_entity_id"], set())
                on_gold = [e for pair, e in gold_pairs.items() if (pair & ga) and (pair & gb) and len(pair & (ga | gb)) == 2]
                exact = [e for e in q["gold_path"] if G_EDGE[e]["source"] in ga and G_EDGE[e]["target"] in gb and G_EDGE[e]["relation"] == r["relation_type"]]
                P(f"     stored edge {r['relationship_id']}: {E[r['source_entity_id']]['name']} -{r['relation_type']}-> {E[r['target_entity_id']]['name']} | same endpoints as gold-path edge: {on_gold or 'no'} | exact gold-path edge: {exact or 'no'}")
    P("\nFate of EDGE_0014 and EDGE_0080, with every raw triple touching the gold endpoints (by mention surface name):")
    for eid in ("EDGE_0014", "EDGE_0080"):
        ge = G_EDGE[eid]
        P(f"  {eid}: {ge['source_name']} -{ge['relation']}-> {ge['target_name']} [{ge['supporting_doc_id']}] | score.py fate: {fates[eid]['fate']}")
        P(f"     doc text: {docs[ge['supporting_doc_id']]}")
        sn, tn = fl.norm_name(ge["source_name"]), fl.norm_name(ge["target_name"])
        for t in triples:
            s, o = fl.norm_name(M[t["source_entity_id"]]["surface_name"]), fl.norm_name(M[t["target_entity_id"]]["surface_name"])
            if {s, o} & {sn, tn} and t["source_doc_ids"][0] == "D" + ge["supporting_doc_id"][-3:]:
                P(f"     triple {t['relationship_id']}: {M[t['source_entity_id']]['surface_name']} ({M[t['source_entity_id']]['entity_type']}) -{t['relation_type']}-> {M[t['target_entity_id']]['surface_name']} ({M[t['target_entity_id']]['entity_type']}) [{t['validation_status']}]")
    return ext, cov


# ============================================================================= B5
def b5():
    P("\n" + "=" * 100)
    P("B5. PROVENANCE MARKER IN ANSWERS")
    for name, path in RUNS.items():
        recs = fl.read_jsonl(path / "results.jsonl")
        n_edge = [r["question_id"] for r in recs if "GRAPH_EDGE_PROVENANCE" in (r.get("answer") or "")]
        n_seed = [r["question_id"] for r in recs if "GRAPH_SEED_PROVENANCE" in (r.get("answer") or "")]
        n_any = [r["question_id"] for r in recs if "PROVENANCE" in (r.get("answer") or "")]
        P(f"  {name}: records={len(recs)} answers with 'GRAPH_EDGE_PROVENANCE'={len(n_edge)} 'GRAPH_SEED_PROVENANCE'={len(n_seed)} any 'PROVENANCE'={len(n_any)}")
        if n_edge:
            P(f"     question ids: {n_edge}")
            ex = next(r for r in recs if r["question_id"] == n_edge[0])
            P(f"     example {ex['question_id']}: {ex['answer'][:300]!r}")
    for i, line in enumerate((SK / "run_overnight.py").read_text().splitlines(), 1):
        if "graph_context_mode" in line:
            P(f"  experiments/synth_kg/run_overnight.py:{i}: {line.strip()}")
    for rel_path, pats in [("src/retrieval/graph_retriever.py", ["GRAPH_EDGE_PROVENANCE:", "GRAPH_SEED_PROVENANCE:"]),
                           ("src/retrieval/context_builder.py", ["is_none_mode =", "is_fix_e_mode =", "### Supporting Graph Traversal Paths", "for p in ev.supporting_graph_paths"]),
                           ("src/engine.py", ["graph_context_mode="])]:
        for i, line in enumerate((BASE / rel_path).read_text().splitlines(), 1):
            if any(p in line for p in pats):
                P(f"  {rel_path}:{i}: {line.strip()}")


# ============================================================================= B6
def b6(ext, doc_ids, q_ids, doc_mat, q_mat):
    P("\n" + "=" * 100)
    P("B6. RECOUNTS")
    es = ext["entity_scoring"]
    P("score.py entity scoring on the stored KB:", es)
    g2c = collections.defaultdict(set)
    for e in entities:
        gid = canon2gold.get(fl.norm_name(e["name"]))
        if gid:
            g2c[gid].add(e["entity_id"])
    split = {g: c for g, c in g2c.items() if len(c) > 1}
    P(f"Gold entities whose canonical name maps to >1 KB canonical entity (exact normalized canonical name): {len(split)}")
    for g, cs in sorted(split.items()):
        P(f"   {g} {G_ENT[g]['canonical_name']} ({G_ENT[g]['type']}): " + ", ".join(f"{c}:{E[c]['entity_type']}(mentions={len(E[c]['source_mention_ids'])})" for c in sorted(cs)))
    P("   type-pair pattern of the splits:", dict(collections.Counter(" vs ".join(sorted(E[c]["entity_type"] for c in cs)) for cs in split.values())))
    P(f"KB canonical entities: {len(entities)}; mapping to a gold canonical name: {sum(len(c) for c in g2c.values())}; gold entities with >=1 match: {len(g2c)} of {len(gold['entities'])}; "
      f"extra (no gold canonical-name match): {len(entities) - sum(len(c) for c in g2c.values())}")

    P("\nGold MANAGES edges by endpoint types:", dict(collections.Counter((e["source_type"], e["target_type"]) for e in gold["edges"] if e["relation"] == "MANAGES")))
    cnt = 0
    for ge in gold["edges"]:
        if ge["relation"] != "MANAGES" or ge["target_type"] != "EMPLOYEE":
            continue
        sn, tn = fl.norm_name(ge["source_name"]), fl.norm_name(ge["target_name"])
        hits = [t for t in triples if fl.norm_name(M[t["source_entity_id"]]["surface_name"]) == sn and fl.norm_name(M[t["target_entity_id"]]["surface_name"]) == tn]
        dev = [t for t in hits if t["relation_type"] == "DEVELOPS"]
        if dev:
            cnt += 1
        P(f"   {ge['id']} {ge['source_name']} -MANAGES-> {ge['target_name']} [{ge['supporting_doc_id']}]: raw triples same direction = "
          + (", ".join(f"{t['relation_type']}[{t['validation_status']}] tgt_mention_type={M[t['target_entity_id']]['entity_type']}" for t in hits) or "NONE"))
    P(f"Gold EMPLOYEE->EMPLOYEE MANAGES edges with a raw triple typed DEVELOPS between the same mentions: {cnt}")

    by_doc_pair = collections.defaultdict(list)
    for ge in gold["edges"]:
        by_doc_pair[(ge["supporting_doc_id"], frozenset([ge["source"], ge["target"]]))].append(ge)
    multi = {k: v for k, v in by_doc_pair.items() if len(v) > 1}
    multi_any_doc = collections.defaultdict(list)
    for ge in gold["edges"]:
        multi_any_doc[frozenset([ge["source"], ge["target"]])].append(ge)
    P(f"\nGold (doc, entity pair) groups with >1 gold relation: {len(multi)} in {len({k[0] for k in multi})} documents; "
      f"entity pairs with >1 gold relation across any documents: {sum(1 for v in multi_any_doc.values() if len(v) > 1)}")
    fates = ext["fate_breakdown"]["edge_fates"]
    for (doc, _), ges in sorted(multi.items()):
        P(f"   {doc}: " + " | ".join(f"{g['id']} {g['source_name']} -{g['relation']}-> {g['target_name']} => {fates[g['id']]['fate']}" for g in ges))
    P("score.py fate counts over all 95 gold edges:", ext["fate_breakdown"]["counts"])

    hyb = {r["question_id"]: r for r in fl.read_jsonl(RUNS["hybrid"] / "results.jsonl")}
    P("\nAll-gold evidence recall recomputed from run records (context_doc_ids) and from dense_embeddings.npz:")
    for mode, path in RUNS.items():
        recs = {r["question_id"]: r for r in fl.read_jsonl(path / "results.jsonl")}
        row = []
        for g in fl.GROUPS:
            ids = [q["id"] for q in questions if q["group"] == g]
            k = sum(fl.all_gold(recs[i].get("context_doc_ids", []), Q[i]["gold_supporting_doc_ids"]) for i in ids)
            row.append(f"{g} {fl.fmt_rate(k, len(ids))}")
        P(f"   {mode}: " + " | ".join(row))
    q2 = np.load(AUD / "followup/q2_embeddings.npz")["q2_mat"]
    meth = collections.defaultdict(lambda: collections.defaultdict(int))
    for i, qid in enumerate(q_ids):
        q = Q[qid]
        ranked, sims = fl.dense_ranking(doc_ids, doc_mat, q_mat[i])
        out = {"B1": fl.b1_dense(ranked), "B7": fl.b7_iter_dense(ranked, doc_ids, doc_mat, q2[i]),
               "B9": fl.b9_overlap_top1(q["question"], ranked, sims, doc_ids, docs),
               "B10": fl.b10_two_step(q["question"], ranked, sims, doc_ids, docs)}
        for m, d in out.items():
            meth[m][q["group"]] += fl.all_gold(d, q["gold_supporting_doc_ids"])
    for m in ("B1", "B7", "B9", "B10"):
        P(f"   {m} (fixlib reimplementation, seed 7): " + " | ".join(f"{g} {fl.fmt_rate(meth[m][g], sum(1 for q in questions if q['group']==g))}" for g in fl.GROUPS))
    P("   (B10 on seed 7 is shown for information; it was defined after the seed-7 data were seen.)")


def main():
    score = load_score_module()
    b1()
    b2()
    doc_ids, q_ids, doc_mat, q_mat = b3()
    ext, cov = b4(score)
    b5()
    b6(ext, doc_ids, q_ids, doc_mat, q_mat)
    OUT.write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
