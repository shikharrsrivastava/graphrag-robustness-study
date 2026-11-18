#!/usr/bin/env python3
"""Phase 3c: pre-registered seed-11 evaluation (retrieval only). See experiments/synth_kg_fix/PROTOCOL.md.

Inputs (all on disk): corpus_seed11/, embeddings_seed11.npz (embed_corpus.py), retrieval_seed11.jsonl
(retrieval_eval.py: hybrid and vector_only through the public API with cached plans).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
FIX = Path(__file__).resolve().parent
sys.path.insert(0, str(FIX))
import fixlib as fl  # noqa: E402

CORPUS = FIX / "corpus_seed11"
OUT = FIX / "seed11_eval_output.txt"


def main() -> None:
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    questions = [q for q in fl.read_jsonl(CORPUS / "questions.jsonl") if q["group"] != "unanswerable"]
    Q = {q["id"]: q for q in questions}
    docs = fl.load_docs(CORPUS)
    emb = np.load(FIX / "embeddings_seed11.npz")
    doc_ids, q_ids = [str(x) for x in emb["doc_ids"]], [str(x) for x in emb["q_ids"]]
    assert q_ids == [q["id"] for q in questions] and sorted(doc_ids) == sorted(docs)
    doc_mat, q_mat, q2_mat = emb["doc_mat"], emb["q_mat"], emb["q2_mat"]

    ok = {m: {} for m in ["hybrid", "vector_only", "B1_dense_top4", "B7_iter_dense", "B9_overlap_top1", "B10_two_step"]}
    ndocs = {m: [] for m in ok}
    errors = {m: [] for m in ok}
    recs = fl.read_jsonl(FIX / "retrieval_seed11_all.jsonl")
    for r in recs:
        if r["mode"] in ("hybrid", "vector_only"):
            ok[r["mode"]][r["question_id"]] = fl.all_gold(r["context_doc_ids"], Q[r["question_id"]]["gold_supporting_doc_ids"])
            ndocs[r["mode"]].append(len({fl.norm_doc_id(d) for d in r["context_doc_ids"]}))
            if r.get("error"):
                errors[r["mode"]].append(r["question_id"])
    plan_errors = sorted({r["question_id"] for r in recs if r.get("plan_error")})
    for i, qid in enumerate(q_ids):
        q = Q[qid]
        ranked, sims = fl.dense_ranking(doc_ids, doc_mat, q_mat[i])
        out = {
            "B1_dense_top4": fl.b1_dense(ranked),
            "B7_iter_dense": fl.b7_iter_dense(ranked, doc_ids, doc_mat, q2_mat[i]),
            "B9_overlap_top1": fl.b9_overlap_top1(q["question"], ranked, sims, doc_ids, docs),
            "B10_two_step": fl.b10_two_step(q["question"], ranked, sims, doc_ids, docs),
        }
        for m, d in out.items():
            ok[m][qid] = fl.all_gold(d, q["gold_supporting_doc_ids"])
            ndocs[m].append(len(set(d)))
    for m in ok:
        assert set(ok[m]) == set(Q), (m, len(ok[m]))

    P("=" * 110)
    P("PHASE 3c - SEED 11 CONFIRMATION: ALL-GOLD EVIDENCE RECALL AT K=4 (retrieval only), Wilson 95% intervals")
    P(f"answerable questions: {len(questions)} " + str({g: sum(1 for q in questions if q['group'] == g) for g in fl.GROUPS}))
    P("| method | " + " | ".join(fl.GROUPS) + " | max distinct docs returned | ERROR |")
    P("|---|" + "---|" * (len(fl.GROUPS) + 2))
    for m in ok:
        cells = []
        for g in fl.GROUPS:
            ids = [qid for qid in Q if Q[qid]["group"] == g]
            cells.append(fl.fmt_rate(sum(ok[m][i] for i in ids), len(ids)))
        P(f"| {m} | " + " | ".join(cells) + f" | {max(ndocs[m])} | {len(errors[m])} |")
    P(f"planner failures (cached plans): {len(plan_errors)} {plan_errors}; query ERRORs hybrid={errors['hybrid']} vector_only={errors['vector_only']} "
      "(an ERROR counts as not retrieved and is listed here separately)")

    P("\nPRIMARY OUTCOMES (paired exact two-sided sign test on per-question success; Holm over the 3 tests; alpha 0.05)")
    prim = [("P1", "hop2", "B9_overlap_top1"), ("P2", "hop3", "B10_two_step"), ("P3", "alias", "B9_overlap_top1")]
    raw_p, rows = [], []
    for name, g, base in prim:
        ids = [qid for qid in Q if Q[qid]["group"] == g]
        h, b = sum(ok["hybrid"][i] for i in ids), sum(ok[base][i] for i in ids)
        n11 = sum(1 for i in ids if ok["hybrid"][i] and ok[base][i])
        n10 = sum(1 for i in ids if ok["hybrid"][i] and not ok[base][i])
        n01 = sum(1 for i in ids if not ok["hybrid"][i] and ok[base][i])
        n00 = len(ids) - n11 - n10 - n01
        p = fl.sign_test(n10, n01)
        raw_p.append(p)
        rows.append((name, g, base, len(ids), h, b, n11, n10, n01, n00, p))
    adj = fl.holm(raw_p)
    P("| outcome | group | baseline | N | hybrid | baseline | diff (points) | both | hybrid only | baseline only | neither | p (sign) | p (Holm) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    decisions = []
    for (name, g, base, n, h, b, n11, n10, n01, n00, p), pa in zip(rows, adj):
        diff = 100.0 * (h - b) / n
        P(f"| {name} | {g} | {base} | {n} | {h}/{n} ({100*h/n:.1f}%) | {b}/{n} ({100*b/n:.1f}%) | {diff:+.1f} | {n11} | {n10} | {n01} | {n00} | {p:.4f} | {pa:.4f} |")
        if diff >= 10.0 and pa < 0.05:
            d = "graph advantage demonstrated"
        elif (100.0 * b / n) >= (100.0 * h / n) - 5.0:
            d = "graph-free baseline matches or exceeds hybrid"
        else:
            d = "inconclusive"
        decisions.append((name, g, base, d))
    P("\nDECISION STATEMENTS (pre-registered rule)")
    for name, g, base, d in decisions:
        P(f"  {name} (hybrid vs {base} on {g}): {d}")

    P("\nDESCRIPTIVE (no decision): hybrid vs vector_only per group")
    for g in fl.GROUPS:
        ids = [qid for qid in Q if Q[qid]["group"] == g]
        n10 = sum(1 for i in ids if ok["hybrid"][i] and not ok["vector_only"][i])
        n01 = sum(1 for i in ids if not ok["hybrid"][i] and ok["vector_only"][i])
        P(f"  {g}: hybrid {fl.fmt_rate(sum(ok['hybrid'][i] for i in ids), len(ids))} | vector_only {fl.fmt_rate(sum(ok['vector_only'][i] for i in ids), len(ids))} | hybrid only {n10}, vector only {n01}, sign p={fl.sign_test(n10, n01):.4f} (unadjusted)")
    (FIX / "seed11_per_question.json").write_text(json.dumps(ok, indent=1), encoding="utf-8")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
