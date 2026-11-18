#!/usr/bin/env python3
"""Offline scoring engine for Synthetic KG Benchmark.

Computes:
1. Answer correctness and failure categorization per group and mode.
2. Evidence recall (gold supporting docs present in retrieved context).
3. Exact two-sided sign test and Wilson 95% confidence intervals.
4. Extraction scoring against gold_kg.json (precision, recall, splits, merges, fate breakdown).
5. Path coverage analysis on the validated graph.

All metrics are computed strictly from files on disk.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def normalize_doc_id(doc_id: Optional[str]) -> str:
    """Normalize document identifiers (e.g. 'D001' or 'DOC_0001' -> 'DOC_0001')."""
    if not doc_id:
        return ""
    m = re.search(r"\d+", str(doc_id))
    if m:
        return f"DOC_{int(m.group()):04d}"
    return str(doc_id).strip().upper()


def normalize_text(text: Optional[str]) -> str:
    """Normalize text: lower-case, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    # Replace non-alphanumeric characters with space
    s = re.sub(r"[^\w\s]", " ", text.lower())
    # Collapse multiple whitespaces and strip
    return " ".join(s.split())


def is_whole_word_match(needle: str, haystack: str) -> bool:
    """Check if needle appears in haystack as a whole-word phrase."""
    norm_n = normalize_text(needle)
    norm_h = normalize_text(haystack)
    if not norm_n or not norm_h:
        return False
    # Whole-word regex match on normalized strings
    pattern = r"\b" + re.escape(norm_n) + r"\b"
    return bool(re.search(pattern, norm_h))


def wilson_ci(k: int, n: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculate Wilson 95% score confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def exact_sign_test(n10: int, n01: int) -> float:
    """Exact two-sided sign test (exact binomial test with p=0.5 on discordant pairs)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p_lower = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_lower)


def evaluate_single_answer(
    question_item: Dict[str, Any],
    record: Dict[str, Any],
    gold_entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[bool, Optional[str]]:
    """Evaluate correctness and failure category of a single question prediction.

    Returns:
        (is_correct, failure_class)
        failure_class is None when correct; otherwise one of:
        "ERROR", "ABSTAINED", "WRONG_ENTITY", "OTHER"
    """
    group = question_item.get("group", "")
    is_unanswerable = (group == "unanswerable")

    # Check for generation error
    gen_err = bool(record.get("generation_error", False))
    if gen_err:
        return False, "ERROR"

    abstained = bool(record.get("abstained", False))

    if is_unanswerable:
        # Correct if abstained is true and generation_error is false
        if abstained:
            return True, None
        return False, "OTHER"

    # Answerable groups: hop1, hop2, hop3, alias
    if abstained:
        return False, "ABSTAINED"

    answer_text = str(record.get("answer", ""))
    gold_answer = question_item.get("gold_answer")
    gold_aliases = question_item.get("gold_answer_aliases", [])

    candidates: List[str] = []
    if gold_answer:
        candidates.append(str(gold_answer))
    for alias in gold_aliases:
        if alias:
            candidates.append(str(alias))

    # Check if answer contains any gold candidate as a whole-word phrase
    for cand in candidates:
        if is_whole_word_match(cand, answer_text):
            return True, None

    # Incorrect: determine whether WRONG_ENTITY or OTHER
    if gold_entities_by_type:
        gold_type = question_item.get("target_type") or question_item.get("gold_answer_type")
        # If target_type not in question, search gold answer entity type
        if not gold_type and gold_entities_by_type:
            for etype, elist in gold_entities_by_type.items():
                for e in elist:
                    if e.get("canonical_name") == gold_answer:
                        gold_type = etype
                        break
                if gold_type:
                    break

        if gold_type and gold_type in gold_entities_by_type:
            # Check other entities of the same type
            cand_norms = {normalize_text(c) for c in candidates}
            for other_e in gold_entities_by_type[gold_type]:
                other_names = [other_e.get("canonical_name", "")] + other_e.get("aliases", [])
                for oname in other_names:
                    if oname and normalize_text(oname) not in cand_norms:
                        if is_whole_word_match(oname, answer_text):
                            return False, "WRONG_ENTITY"

    return False, "OTHER"


def score_extraction(
    gold_kg: Dict[str, Any],
    canonical_entities: List[Dict[str, Any]],
    validated_relationships: List[Dict[str, Any]],
    all_extracted_triples: Optional[List[Dict[str, Any]]] = None,
    entity_mentions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score knowledge base extraction and validation against gold_kg.json.

    Computes:
    - Entity recall, entity precision, splits, false merges
    - Exact and relaxed relationship precision and recall
    - Gold edge fate breakdown: VALIDATED_CORRECT, REJECTED (<reason>), NEVER_EXTRACTED
    """
    gold_entities = gold_kg.get("entities", [])
    gold_edges = gold_kg.get("edges", [])

    # Map gold entity ID to canonical names and aliases
    gold_entity_map: Dict[str, Dict[str, Any]] = {}
    name_to_gold_id: Dict[str, str] = {}
    for ge in gold_entities:
        gid = ge["id"]
        gold_entity_map[gid] = ge
        all_names = [ge["canonical_name"]] + ge.get("aliases", [])
        for n in all_names:
            norm = normalize_text(n)
            if norm:
                name_to_gold_id[norm] = gid

    # 1. Map extracted canonical entities to gold entities
    extracted_to_gold: Dict[str, Set[str]] = collections.defaultdict(set)
    gold_to_extracted: Dict[str, Set[str]] = collections.defaultdict(set)

    for ce in canonical_entities:
        cid = ce["entity_id"]
        ce_names = [ce["name"]] + ce.get("aliases", [])
        for n in ce_names:
            norm = normalize_text(n)
            if norm in name_to_gold_id:
                gid = name_to_gold_id[norm]
                extracted_to_gold[cid].add(gid)
                gold_to_extracted[gid].add(cid)

    total_gold_entities = len(gold_entities)
    total_extracted_entities = len(canonical_entities)

    recalled_gold_entities = sum(1 for gid in gold_entity_map if len(gold_to_extracted[gid]) >= 1)
    entity_recall = (recalled_gold_entities / total_gold_entities) if total_gold_entities > 0 else 0.0

    valid_extracted_entities = sum(1 for cid in extracted_to_gold if len(extracted_to_gold[cid]) >= 1)
    entity_precision = (valid_extracted_entities / total_extracted_entities) if total_extracted_entities > 0 else 0.0

    splits = sum(1 for gid, cids in gold_to_extracted.items() if len(cids) > 1)
    false_merges = sum(1 for cid, gids in extracted_to_gold.items() if len(gids) > 1)

    # 2. Relationship scoring over validated edges
    # Build gold edge sets: exact (u, v, rel) and relaxed (frozenset({u, v}))
    gold_edges_exact: Dict[Tuple[str, str, str], str] = {}
    gold_edges_relaxed: Dict[frozenset, str] = {}
    for ge in gold_edges:
        gold_edges_exact[(ge["source"], ge["target"], ge["relation"])] = ge["id"]
        gold_edges_relaxed[frozenset([ge["source"], ge["target"]])] = ge["id"]

    total_validated_edges = len(validated_relationships)
    exact_matched_validated = 0
    relaxed_matched_validated = 0

    recalled_gold_edges_exact: Set[str] = set()
    recalled_gold_edges_relaxed: Set[str] = set()

    for rel in validated_relationships:
        s_cids = extracted_to_gold.get(rel["source_entity_id"], set())
        t_cids = extracted_to_gold.get(rel["target_entity_id"], set())
        rtype = rel["relation_type"].strip().upper().replace(" ", "_")

        matched_exact = False
        matched_relaxed = False

        for sg in s_cids:
            for tg in t_cids:
                if (sg, tg, rtype) in gold_edges_exact:
                    matched_exact = True
                    recalled_gold_edges_exact.add(gold_edges_exact[(sg, tg, rtype)])
                if frozenset([sg, tg]) in gold_edges_relaxed:
                    matched_relaxed = True
                    recalled_gold_edges_relaxed.add(gold_edges_relaxed[frozenset([sg, tg])])

        if matched_exact:
            exact_matched_validated += 1
        if matched_relaxed:
            relaxed_matched_validated += 1

    rel_exact_prec = (exact_matched_validated / total_validated_edges) if total_validated_edges > 0 else 0.0
    rel_exact_rec = (len(recalled_gold_edges_exact) / len(gold_edges)) if gold_edges else 0.0

    rel_relaxed_prec = (relaxed_matched_validated / total_validated_edges) if total_validated_edges > 0 else 0.0
    rel_relaxed_rec = (len(recalled_gold_edges_relaxed) / len(gold_edges)) if gold_edges else 0.0

    # 3. Gold edge fate breakdown
    # Build lookup of extracted triples by matching mention text or quotes
    mention_to_gold: Dict[str, Set[str]] = collections.defaultdict(set)
    if entity_mentions:
        for m in entity_mentions:
            mid = m["mention_id"]
            mnorm = normalize_text(m["surface_name"])
            if mnorm in name_to_gold_id:
                mention_to_gold[mid].add(name_to_gold_id[mnorm])

    edge_fates: Dict[str, Dict[str, Any]] = {}
    fate_counts: collections.Counter = collections.Counter()

    # Pre-index all extracted triples by candidate (source_gold, target_gold) or quote
    gold_edge_to_triples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

    if all_extracted_triples:
        for tr in all_extracted_triples:
            s_mid = tr.get("source_entity_id", "")
            t_mid = tr.get("target_entity_id", "")
            s_gids = mention_to_gold.get(s_mid, extracted_to_gold.get(s_mid, set()))
            t_gids = mention_to_gold.get(t_mid, extracted_to_gold.get(t_mid, set()))
            quotes = tr.get("evidence_quotes", [])

            # Match triple to gold edge
            for ge in gold_edges:
                matched = False
                if ge["source"] in s_gids and ge["target"] in t_gids:
                    matched = True
                elif any(is_whole_word_match(ge["evidence_quote"], q) or is_whole_word_match(q, ge["evidence_quote"]) for q in quotes):
                    matched = True

                if matched:
                    gold_edge_to_triples[ge["id"]].append(tr)

    for ge in gold_edges:
        eid = ge["id"]
        if eid in recalled_gold_edges_exact:
            fate = "VALIDATED_CORRECT"
        else:
            # Check if extracted and rejected
            triples = gold_edge_to_triples.get(eid, [])
            if triples:
                # Find rejected status
                reasons = [
                    t.get("validation_status", "REJECTED_UNKNOWN")
                    for t in triples
                    if str(t.get("validation_status", "")).startswith("REJECTED")
                ]
                if reasons:
                    fate = f"REJECTED: {reasons[0]}"
                else:
                    # Maybe validated with wrong type or remapping dropped it
                    fate = "EXTRACTED_BUT_DROPPED_IN_RESOLUTION"
            else:
                fate = "NEVER_EXTRACTED"

        edge_fates[eid] = {
            "edge_id": eid,
            "source": ge.get("source_name", ge.get("source", "")),
            "relation": ge.get("relation", ""),
            "target": ge.get("target_name", ge.get("target", "")),
            "doc_id": ge.get("supporting_doc_id", ""),
            "fate": fate,
        }
        fate_counts[fate] += 1

    return {
        "entity_scoring": {
            "gold_entities_total": total_gold_entities,
            "extracted_entities_total": total_extracted_entities,
            "recalled_gold_entities": recalled_gold_entities,
            "valid_extracted_entities": valid_extracted_entities,
            "precision": entity_precision,
            "recall": entity_recall,
            "splits": splits,
            "false_merges": false_merges,
        },
        "relationship_scoring": {
            "gold_edges_total": len(gold_edges),
            "validated_edges_total": total_validated_edges,
            "exact_matched_validated": exact_matched_validated,
            "exact_recalled_gold": len(recalled_gold_edges_exact),
            "exact_precision": rel_exact_prec,
            "exact_recall": rel_exact_rec,
            "relaxed_matched_validated": relaxed_matched_validated,
            "relaxed_recalled_gold": len(recalled_gold_edges_relaxed),
            "relaxed_precision": rel_relaxed_prec,
            "relaxed_recall": rel_relaxed_rec,
        },
        "fate_breakdown": {
            "counts": dict(fate_counts),
            "edge_fates": edge_fates,
        },
    }


def compute_path_coverage(
    questions: List[Dict[str, Any]],
    canonical_entities: List[Dict[str, Any]],
    validated_relationships: List[Dict[str, Any]],
    gold_kg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute graph path coverage for multi-hop questions on the validated graph.

    Determines if an undirected path of length <= h exists between the entity
    matched to the first entity and the entity matched to the gold answer.
    """
    # Build name -> gold ID map
    name_to_gold: Dict[str, str] = {}
    for ge in gold_kg.get("entities", []):
        for n in [ge["canonical_name"]] + ge.get("aliases", []):
            norm = normalize_text(n)
            if norm:
                name_to_gold[norm] = ge["id"]

    # Map canonical entity ID -> set of gold IDs
    cid_to_gold: Dict[str, Set[str]] = collections.defaultdict(set)
    for ce in canonical_entities:
        cid = ce["entity_id"]
        for n in [ce["name"]] + ce.get("aliases", []):
            norm = normalize_text(n)
            if norm in name_to_gold:
                cid_to_gold[cid].add(name_to_gold[norm])

    # Build undirected adjacency graph between canonical entities
    adj: Dict[str, Set[str]] = collections.defaultdict(set)
    for rel in validated_relationships:
        u = rel["source_entity_id"]
        v = rel["target_entity_id"]
        adj[u].add(v)
        adj[v].add(u)

    def shortest_path_len(start_cids: Set[str], target_cids: Set[str]) -> Optional[int]:
        if not start_cids or not target_cids:
            return None
        if start_cids.intersection(target_cids):
            return 0
        queue = collections.deque([(cid, 0) for cid in start_cids if cid in adj])
        visited = set(cid for cid, _ in queue)
        while queue:
            curr, dist = queue.popleft()
            if curr in target_cids:
                return dist
            for neighbor in adj.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    # Evaluate each question
    question_path_coverage: Dict[str, bool] = {}
    group_stats: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"total": 0, "path_exists": 0})

    # Find starting entity and target entity for questions
    for q in questions:
        qid = q["id"]
        group = q.get("group", "")
        h = q.get("hop_count", 1)
        if group not in ("hop1", "hop2", "hop3", "alias"):
            continue

        # Target entity candidates
        gold_ans = q.get("gold_answer")
        gold_aliases = q.get("gold_answer_aliases", [])
        target_names = [gold_ans] + gold_aliases
        target_gold_ids = {name_to_gold[normalize_text(n)] for n in target_names if normalize_text(n) in name_to_gold}

        # First entity: from question text or gold_path
        # If question has gold_path, find the source of the first edge
        first_gold_id = None
        gold_edges = {e["id"]: e for e in gold_kg.get("edges", [])}
        gold_path = q.get("gold_path", [])
        if gold_path and gold_path[0] in gold_edges:
            first_gold_id = gold_edges[gold_path[0]]["source"]

        # If not, try matching question text against gold entity names
        if not first_gold_id:
            for n, gid in name_to_gold.items():
                if gid not in target_gold_ids and is_whole_word_match(n, q["question"]):
                    first_gold_id = gid
                    break

        if not first_gold_id or not target_gold_ids:
            question_path_coverage[qid] = False
            group_stats[group]["total"] += 1
            continue

        # Get canonical IDs matching first_gold_id and target_gold_ids
        start_cids = {cid for cid, gids in cid_to_gold.items() if first_gold_id in gids}
        target_cids = {cid for cid, gids in cid_to_gold.items() if any(tg in gids for tg in target_gold_ids)}

        path_len = shortest_path_len(start_cids, target_cids)
        has_path = (path_len is not None and path_len <= h)
        question_path_coverage[qid] = has_path

        group_stats[group]["total"] += 1
        if has_path:
            group_stats[group]["path_exists"] += 1

    group_fractions = {}
    for g, data in group_stats.items():
        tot = data["total"]
        pe = data["path_exists"]
        group_fractions[g] = {
            "total": tot,
            "path_exists": pe,
            "fraction": (pe / tot) if tot > 0 else 0.0,
        }

    return {
        "group_path_coverage": group_fractions,
        "question_path_coverage": question_path_coverage,
    }


def score_runs_against_disk(
    corpus_final_dir: Path,
    kb_dir: Path,
    runs_dir: Path,
) -> Dict[str, Any]:
    """Load all runs from disk and compute complete benchmark scoring."""
    gold_kg_path = corpus_final_dir / "gold_kg.json"
    questions_path = corpus_final_dir / "questions.jsonl"

    with open(gold_kg_path, "r", encoding="utf-8") as f:
        gold_kg = json.load(f)

    questions: List[Dict[str, Any]] = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Pre-organize gold entities by type
    gold_entities_by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for ge in gold_kg.get("entities", []):
        gold_entities_by_type[ge["type"]].append(ge)

    # Load KB artifacts
    canonical_entities_path = kb_dir / "graph" / "entities.jsonl"
    validated_rel_path = kb_dir / "graph" / "relationships.jsonl"
    all_triples_path = kb_dir / "data" / "processed" / "triples.jsonl"
    mentions_path = kb_dir / "data" / "processed" / "entity_mentions.jsonl"

    canonical_entities: List[Dict[str, Any]] = []
    if canonical_entities_path.exists():
        with open(canonical_entities_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    canonical_entities.append(json.loads(line))

    validated_relationships: List[Dict[str, Any]] = []
    if validated_rel_path.exists():
        with open(validated_rel_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    validated_relationships.append(json.loads(line))

    all_triples: List[Dict[str, Any]] = []
    if all_triples_path.exists():
        with open(all_triples_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_triples.append(json.loads(line))

    entity_mentions: List[Dict[str, Any]] = []
    if mentions_path.exists():
        with open(mentions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entity_mentions.append(json.loads(line))

    # Compute extraction scoring
    extraction_results = score_extraction(
        gold_kg=gold_kg,
        canonical_entities=canonical_entities,
        validated_relationships=validated_relationships,
        all_extracted_triples=all_triples,
        entity_mentions=entity_mentions,
    )

    # Compute path coverage
    path_coverage_results = compute_path_coverage(
        questions=questions,
        canonical_entities=canonical_entities,
        validated_relationships=validated_relationships,
        gold_kg=gold_kg,
    )

    # Find run directories
    modes = ["vector_only", "hybrid", "graph_only", "community_aware"]
    mode_runs: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        # Find matching run directories
        matching_dirs = sorted(runs_dir.glob(f"*_{mode}"))
        if not matching_dirs:
            continue
        # Use latest
        run_dir = matching_dirs[-1]
        results_file = run_dir / "results.jsonl"
        records_by_qid: Dict[str, Dict[str, Any]] = {}
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        records_by_qid[rec.get("question_id") or rec.get("id")] = rec

        mode_runs[mode] = {
            "run_dir": str(run_dir),
            "records": records_by_qid,
        }

    # Evaluate each question per mode
    per_question_eval: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    mode_group_stats: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: collections.defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "error": 0,
        "abstained": 0,
        "wrong_entity": 0,
        "other": 0,
        "evidence_recalled": 0,
        "latencies": [],
    }))

    for q in questions:
        qid = q["id"]
        group = q.get("group", "")
        gold_docs = {normalize_doc_id(d) for d in q.get("gold_supporting_doc_ids", []) if d}

        for mode, rdata in mode_runs.items():
            rec = rdata["records"].get(qid)
            if not rec:
                continue

            is_corr, fail_class = evaluate_single_answer(
                question_item=q,
                record=rec,
                gold_entities_by_type=gold_entities_by_type,
            )

            # Check evidence recall
            ctx_docs: Set[str] = set()
            for c in rec.get("context_chunks", []):
                if isinstance(c, dict) and "doc_id" in c:
                    ctx_docs.add(normalize_doc_id(c["doc_id"]))
            for doc_id in rec.get("context_doc_ids", []):
                ctx_docs.add(normalize_doc_id(doc_id))

            ev_recalled = (gold_docs.issubset(ctx_docs)) if gold_docs else False

            lat = float(rec.get("latency_seconds", 0.0) or rec.get("total_latency_seconds", 0.0))

            per_question_eval[qid][mode] = {
                "correct": is_corr,
                "failure_class": fail_class,
                "evidence_recalled": ev_recalled,
                "latency": lat,
                "prediction": rec.get("answer", ""),
            }

            stats = mode_group_stats[mode][group]
            stats["total"] += 1
            if is_corr:
                stats["correct"] += 1
            if fail_class == "ERROR":
                stats["error"] += 1
            elif fail_class == "ABSTAINED":
                stats["abstained"] += 1
            elif fail_class == "WRONG_ENTITY":
                stats["wrong_entity"] += 1
            elif fail_class == "OTHER":
                stats["other"] += 1

            if ev_recalled:
                stats["evidence_recalled"] += 1
            if lat > 0:
                stats["latencies"].append(lat)

    # Compute overall per mode and per group metrics
    mode_summaries: Dict[str, Any] = {}
    for mode in modes:
        if mode not in mode_runs:
            continue
        g_summaries = {}
        tot_all = 0
        corr_all = 0
        tot_ans = 0
        corr_ans = 0

        for group, s in mode_group_stats[mode].items():
            tot = s["total"]
            corr = s["correct"]
            err = s["error"]
            tot_all += tot
            corr_all += corr
            if group != "unanswerable":
                tot_ans += tot
                corr_ans += corr

            acc = (corr / tot) if tot > 0 else 0.0
            ci_low, ci_high = wilson_ci(corr, tot)
            ev_rec = (s["evidence_recalled"] / tot) if tot > 0 else 0.0

            g_summaries[group] = {
                "total": tot,
                "correct": corr,
                "accuracy": acc,
                "ci_95": [ci_low, ci_high],
                "error": err,
                "abstained": s["abstained"],
                "wrong_entity": s["wrong_entity"],
                "other": s["other"],
                "evidence_recall": ev_rec,
                "mean_latency": (sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else 0.0,
            }

        ans_acc = (corr_ans / tot_ans) if tot_ans > 0 else 0.0
        ans_low, ans_high = wilson_ci(corr_ans, tot_ans)
        all_acc = (corr_all / tot_all) if tot_all > 0 else 0.0
        all_low, all_high = wilson_ci(corr_all, tot_all)

        mode_summaries[mode] = {
            "answerable_accuracy": ans_acc,
            "answerable_ci_95": [ans_low, ans_high],
            "overall_accuracy": all_acc,
            "overall_ci_95": [all_low, all_high],
            "by_group": g_summaries,
        }

    # Primary comparisons: paired sign test against vector_only
    paired_comparisons: Dict[str, Any] = {}
    if "vector_only" in mode_runs:
        vec_q = {qid: per_question_eval[qid].get("vector_only", {}) for qid in per_question_eval}
        for cand_mode in ["hybrid", "graph_only", "community_aware"]:
            if cand_mode not in mode_runs:
                continue
            cand_q = {qid: per_question_eval[qid].get(cand_mode, {}) for qid in per_question_eval}

            # Compare overall answerable and per hop group
            comp_by_group = {}
            for target_group in ["all_answerable", "hop1", "hop2", "hop3", "alias"]:
                n11 = n10 = n01 = n00 = 0
                for q in questions:
                    qid = q["id"]
                    qgrp = q.get("group", "")
                    if target_group == "all_answerable" and qgrp == "unanswerable":
                        continue
                    if target_group != "all_answerable" and qgrp != target_group:
                        continue

                    cand_corr = cand_q.get(qid, {}).get("correct", False)
                    vec_corr = vec_q.get(qid, {}).get("correct", False)

                    if cand_corr and vec_corr:
                        n11 += 1
                    elif cand_corr and not vec_corr:
                        n10 += 1
                    elif not cand_corr and vec_corr:
                        n01 += 1
                    else:
                        n00 += 1

                p_val = exact_sign_test(n10, n01)
                detected = (p_val < 0.05)
                comp_by_group[target_group] = {
                    "cand_mode": cand_mode,
                    "ref_mode": "vector_only",
                    "both_correct": n11,
                    "only_cand_correct": n10,
                    "only_ref_correct": n01,
                    "neither_correct": n00,
                    "discordant": n10 + n01,
                    "p_value": p_val,
                    "difference_detected": detected,
                }
            paired_comparisons[f"{cand_mode}_vs_vector_only"] = comp_by_group

    # Conditional accuracy based on path coverage
    path_cov_map = path_coverage_results.get("question_path_coverage", {})
    conditional_accuracies: Dict[str, Any] = {}
    for mode in ["graph_only", "hybrid", "vector_only"]:
        if mode not in mode_runs:
            continue
        for group in ["hop2", "hop3"]:
            pe_corr = pe_tot = 0
            npe_corr = npe_tot = 0
            for q in questions:
                if q.get("group") != group:
                    continue
                qid = q["id"]
                has_path = path_cov_map.get(qid, False)
                is_corr = per_question_eval.get(qid, {}).get(mode, {}).get("correct", False)
                if has_path:
                    pe_tot += 1
                    if is_corr:
                        pe_corr += 1
                else:
                    npe_tot += 1
                    if is_corr:
                        npe_corr += 1

            conditional_accuracies[f"{mode}_{group}"] = {
                "path_exists": {
                    "total": pe_tot,
                    "correct": pe_corr,
                    "accuracy": (pe_corr / pe_tot) if pe_tot > 0 else 0.0,
                },
                "path_missing": {
                    "total": npe_tot,
                    "correct": npe_corr,
                    "accuracy": (npe_corr / npe_tot) if npe_tot > 0 else 0.0,
                },
            }

    return {
        "extraction_results": extraction_results,
        "path_coverage_results": path_coverage_results,
        "mode_summaries": mode_summaries,
        "paired_comparisons": paired_comparisons,
        "conditional_accuracies": conditional_accuracies,
        "per_question_eval": per_question_eval,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline scoring for Synthetic KG Benchmark")
    parser.add_argument("--corpus_dir", default="experiments/synth_kg/corpus_final")
    parser.add_argument("--kb_dir", default="experiments/synth_kg/kb/synth_company_kb")
    parser.add_argument("--runs_dir", default="experiments/synth_kg/runs")
    parser.add_argument("--out", default="experiments/synth_kg/evaluation_results.json")
    args = parser.parse_args()

    results = score_runs_against_disk(
        corpus_final_dir=Path(args.corpus_dir),
        kb_dir=Path(args.kb_dir),
        runs_dir=Path(args.runs_dir),
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Scoring complete. Output written to {args.out}")
