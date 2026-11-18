#!/usr/bin/env python3
"""Shared helpers for experiments/synth_kg_fix: statistics, loaders and graph-free baselines.

Baseline code (B7, B9, B4-followup) is copied from
experiments/synth_kg_audit/followup/part_f_iterative_baselines.py (read-only original) and
B4-audit from experiments/synth_kg_audit/audit_part2_baselines.py. B1 and B10 are written here.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

NAME_REGEX = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
GROUPS = ["hop1", "hop2", "hop3", "alias"]


# ----------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.95996) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    pct = (100.0 * k / n) if n else 0.0
    return f"{k}/{n} = {pct:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def sign_test(n10: int, n01: int) -> float:
    """Exact two-sided binomial sign test on discordant pairs."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


# ----------------------------------------------------------------------------- loaders
def read_jsonl(path) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def norm_doc_id(did: str) -> str:
    digits = re.findall(r"\d+", did)
    return f"DOC_{int(digits[0]):04d}" if digits else did.strip().upper()


def load_docs(corpus_dir) -> Dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((Path(corpus_dir) / "docs").glob("DOC_*.txt"))}


def norm_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


# ----------------------------------------------------------------------------- baselines
def dense_ranking(doc_ids: List[str], doc_mat: np.ndarray, q_vec: np.ndarray) -> Tuple[List[str], np.ndarray]:
    sims = np.dot(doc_mat, q_vec)
    return [doc_ids[i] for i in np.argsort(-sims)], sims


def _names(doc_text: str, exclude_texts: Iterable[str]) -> Set[str]:
    """Capitalized-name runs in doc_text that do not occur (case-insensitive substring) in any exclude text."""
    lowered = [t.lower() for t in exclude_texts]
    return {m for m in NAME_REGEX.findall(doc_text) if not any(m.lower() in t for t in lowered)}


def _overlap_candidates(names: Set[str], skip: Sequence[str], doc_ids, docs, sims) -> List[str]:
    cands = []
    for d in doc_ids:
        if d in skip:
            continue
        contained = [n for n in names if n in docs[d]]
        if contained:
            cands.append((len(contained), sims[doc_ids.index(d)], d))
    cands.sort(key=lambda x: (-x[0], -x[1]))
    return [c[2] for c in cands]


def _backfill(selected: List[str], ranked: List[str], k: int = 4) -> List[str]:
    out = list(selected)
    for d in ranked:
        if len(out) >= k:
            break
        if d not in out:
            out.append(d)
    return out[:k]


def b1_dense(ranked: List[str]) -> List[str]:
    return ranked[:4]


def b7_iter_dense(ranked: List[str], doc_ids, doc_mat, q2_vec) -> List[str]:
    d1 = ranked[0]
    q2_ranked, _ = dense_ranking(doc_ids, doc_mat, q2_vec)
    return [d1] + [d for d in q2_ranked if d != d1][:3]


def b9_overlap_top1(q_text: str, ranked, sims, doc_ids, docs) -> List[str]:
    s1 = ranked[0]
    names = _names(docs[s1], [q_text])
    cands = _overlap_candidates(names, [s1], doc_ids, docs, sims)
    return _backfill([s1] + cands[:3], ranked)


def b10_two_step(q_text: str, ranked, sims, doc_ids, docs) -> List[str]:
    """Seed = dense top-1; step 1 = B9 expansion keeping only the top-1 candidate; step 2 repeats the
    name-overlap expansion from that candidate (names in the question and in the seed excluded) and adds
    the top 2 candidates. Backfill from the dense ranking if fewer than 4 documents result."""
    s1 = ranked[0]
    names1 = _names(docs[s1], [q_text])
    cands1 = _overlap_candidates(names1, [s1], doc_ids, docs, sims)
    selected = [s1]
    if cands1:
        c1 = cands1[0]
        selected.append(c1)
        names2 = _names(docs[c1], [q_text, docs[s1]])
        cands2 = _overlap_candidates(names2, [s1, c1], doc_ids, docs, sims)
        selected.extend(cands2[:2])
    return _backfill(selected, ranked)


def b4_followup(q_text: str, ranked, sims, doc_ids, docs) -> List[str]:
    """B4 as implemented in followup/part_f_iterative_baselines.py (question names removed)."""
    seeds = ranked[:2]
    names: Set[str] = set()
    for s in seeds:
        names |= _names(docs[s], [q_text])
    cands = [(sims[doc_ids.index(d)], d) for d in doc_ids if d not in seeds and any(n in docs[d] for n in names)]
    cands.sort(key=lambda x: -x[0])
    return _backfill(list(seeds) + [c[1] for c in cands[:2]], ranked)


def b4_audit(q_text: str, ranked, sims, doc_ids, docs, strip_docs: bool = True) -> List[str]:
    """B4 as implemented in audit_part2_baselines.py (question names NOT removed; len>1 filter; stripped docs)."""
    seeds = ranked[:2]
    names: Set[str] = set()
    for s in seeds:
        text = docs[s].strip() if strip_docs else docs[s]
        for m in NAME_REGEX.findall(text):
            if len(m.strip()) > 1:
                names.add(m.strip())
    cands = []
    for d in doc_ids:
        if d in seeds:
            continue
        if any(n in docs[d] for n in names):
            cands.append((d, sims[doc_ids.index(d)]))
    cands.sort(key=lambda x: -x[1])
    return _backfill(list(seeds) + [c[0] for c in cands], ranked)


def all_gold(retrieved: Iterable[str], gold: Iterable[str]) -> bool:
    return {norm_doc_id(g) for g in gold}.issubset({norm_doc_id(r) for r in retrieved})
