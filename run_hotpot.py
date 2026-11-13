"""Question-isolated HotpotQA evaluation and route diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from src.core.configuration import EngineConfig
from src.engine import HybridGraphRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("hotpot_eval")
ROUTES = ("vector_only", "graph_only", "hybrid", "community_aware")


def select_contexts(sample: dict[str, Any], mode: str) -> tuple[list[tuple[str, list[str]]], str | None]:
    contexts = sample.get("context") or []
    if mode == "full":
        return contexts, None
    support = sample.get("supporting_facts")
    if support is None:
        return contexts[:2], "supporting_facts metadata missing; used first two contexts"
    titles = {fact[0] for fact in support if isinstance(fact, (list, tuple)) and fact}
    if not titles:
        return contexts[:2], "supporting_facts metadata empty; used first two contexts"
    selected = [ctx for ctx in contexts if ctx and ctx[0] in titles]
    return selected, None


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)[:48] or hashlib.sha1(value.encode()).hexdigest()[:12]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((__import__("collections").Counter(p) & __import__("collections").Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def extract_canonical_candidate(answer: str, canonical_answer: str | None = None, question: str = "") -> str:
    """Extract or prioritize short canonical answer for benchmark evaluation."""
    if canonical_answer and canonical_answer.strip():
        return canonical_answer.strip()
    cleaned = re.sub(r"\[[^\]]+\]", "", answer).strip()
    # Check leading yes/no
    m = re.match(r"^(yes|no)\b", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Check boolean question conclusion
    q_lower = question.lower().strip()
    if q_lower.startswith(("were ", "was ", "is ", "are ", "did ", "do ", "does ", "can ", "could ")):
        if re.search(r"\b(were|was|are|is)\s+not\b", cleaned, re.IGNORECASE) or re.search(r"\bneither\b", cleaned, re.IGNORECASE):
            return "no"
        if re.search(r"\btherefore,\s*they were\b", cleaned, re.IGNORECASE) or re.search(r"\bthey were of the same\b", cleaned, re.IGNORECASE):
            return "yes"
        if re.search(r"\byes\b", cleaned, re.IGNORECASE) and not re.search(r"\bno\b", cleaned, re.IGNORECASE):
            return "yes"
        if re.search(r"\bno\b", cleaned, re.IGNORECASE) and not re.search(r"\byes\b", cleaned, re.IGNORECASE):
            return "no"
    return cleaned


def supporting_metrics(evidence, sample, contexts, k: int) -> dict[str, Any]:
    gold = {(str(t), int(s)) for t, s in sample.get("supporting_facts", [])}
    ctx_by_title = {str(title): sentences for title, sentences in contexts}
    retrieved = evidence[:k]
    covered = set()
    for ev in retrieved:
        for title, sentences in ctx_by_title.items():
            if safe_id(title) not in ev.filename:
                continue
            ev_text = normalize_answer(ev.text)
            for sent_id, sentence in enumerate(sentences):
                if normalize_answer(sentence) and normalize_answer(sentence) in ev_text:
                    covered.add((title, sent_id))
    docs = {title for title, _ in gold}
    covered_docs = {title for title, _ in covered}
    return {
        "supporting_fact_recall_at_k": (len(gold & covered) / len(gold)) if gold else None,
        "supporting_document_recall_at_k": (len(docs & covered_docs) / len(docs)) if docs else None,
        "complete_supporting_evidence_at_k": bool(gold) and gold <= covered,
        "supporting_document_coverage": sorted(docs & covered_docs),
        "supporting_fact_coverage": sorted([list(x) for x in gold & covered]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    base = yaml.safe_load(config_path.read_text())
    samples = json.loads(Path(args.dataset).read_text())[: args.limit]
    out: dict[str, Any] = {"mode": args.context_mode, "dataset": str(args.dataset), "metrics_scope": "question-isolated", "questions": []}

    for i, sample in enumerate(samples, 1):
        qid = str(sample.get("_id", f"row_{i}"))
        question = sample["question"]
        contexts, fallback = select_contexts(sample, args.context_mode)
        kb_id = f"hotpot_{args.context_mode}_{safe_id(qid)}"
        raw_dir = Path(args.input_root) / args.context_mode / safe_id(qid)
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Clear only generated context files in this question-specific input directory.
        for old in raw_dir.glob("ctx_*.txt"):
            old.unlink()
        title_to_filename = {}
        for idx, (title, sentences) in enumerate(contexts):
            filename = f"ctx_{idx:03d}_{safe_id(str(title))}.txt"
            title_to_filename[str(title)] = filename
            (raw_dir / filename).write_text(f"Title: {title}\n" + " ".join(sentences), encoding="utf-8")

        row: dict[str, Any] = {"id": qid, "question": question, "gold_answer": sample.get("answer", ""),
                               "context_mode": args.context_mode, "context_count": len(contexts),
                               "fallback": fallback, "kb_id": kb_id, "stages": {}, "routes": {}, "errors": []}
        try:
            cfg_data = dict(base, kb_id=kb_id, storage_root=args.storage_root)
            config = EngineConfig(**cfg_data)
            engine = HybridGraphRAGEngine(config=config, base_dir=args.base_dir)
            t0 = time.perf_counter()
            stats = engine.index_directory(raw_dir, rebuild=True)
            row["indexing"] = stats
            row["stages"]["total_indexing_seconds"] = time.perf_counter() - t0
            print(f"[{i}/{len(samples)}] {qid}: docs={stats['documents']} pages={stats['pages']} chunks={stats['chunks']} entities={stats['entities']} relations={stats['relationships']} communities={stats['communities']}")
            for route in ROUTES:
                t1 = time.perf_counter()
                llm_before = dict(engine.llm_client.stats)
                try:
                    answer, result = engine.query(question, mode=route)
                    elapsed = time.perf_counter() - t1
                    llm_after = dict(engine.llm_client.stats)
                    gold_ans = str(sample.get("answer", ""))
                    canonical_pred = extract_canonical_candidate(answer, getattr(result, "canonical_answer", None), question)
                    cleaned_ans = re.sub(r"\[[^\]]+\]", "", answer).strip()
                    em = (normalize_answer(canonical_pred) == normalize_answer(gold_ans)) or (normalize_answer(cleaned_ans) == normalize_answer(gold_ans))
                    f1 = max(
                        token_f1(canonical_pred, gold_ans),
                        token_f1(cleaned_ans, gold_ans),
                    )

                    row["routes"][route] = {
                        "answer": answer,
                        "canonical_answer": getattr(result, "canonical_answer", None) or canonical_pred,
                        "abstained": result.abstained,
                        "evidence": [e.model_dump(mode="json") for e in result.evidence_list],
                        "diagnostics": result.route_diagnostics,
                        "query_plan": result.query_plan.model_dump(mode="json"),
                        "query_seconds": elapsed,
                        "llm_requests": llm_after["requests"] - llm_before["requests"],
                        "llm_failed_requests": llm_after["failed_requests"] - llm_before["failed_requests"],
                        "llm_failed_generations": llm_after["failed_generations"] - llm_before["failed_generations"],
                        "evidence_provenance_valid": all(
                            engine.doc_store.get_chunk(e.chunk_id) is not None
                            and engine.doc_store.get_chunk(e.chunk_id).doc_id == e.doc_id
                            and engine.doc_store.get_chunk(e.chunk_id).text == e.text
                            for e in result.evidence_list
                        ),
                        "invalid_plan": bool(result.query_plan.invalid_relationship_types),
                        "retrieval_metrics": {
                            "recall_at_2": supporting_metrics(result.evidence_list, sample, contexts, 2),
                            "recall_at_5": supporting_metrics(result.evidence_list, sample, contexts, 5),
                        },
                        "answer_metrics": {
                            "exact_match": em,
                            "token_f1": f1,
                            "abstained": result.abstained,
                            "citation_coverage": len(result.cited_chunk_ids) / max(1, len(result.evidence_list)),
                        },
                    }
                    row["stages"][f"{route}_query_seconds"] = elapsed
                except Exception as exc:
                    row["errors"].append({"stage": f"query:{route}", "category": type(exc).__name__, "message": str(exc)})
                    LOG.exception("Question %s route %s failed", qid, route)
        except Exception as exc:
            row["errors"].append({"stage": "indexing", "category": type(exc).__name__, "message": str(exc)})
            LOG.exception("Question %s indexing failed", qid)
        route_chunks = {
            route: {item["chunk_id"] for item in data.get("evidence", [])}
            for route, data in row["routes"].items()
        }
        vector_chunks = route_chunks.get("vector_only", set())
        graph_chunks = route_chunks.get("graph_only", set())
        hybrid_chunks = route_chunks.get("hybrid", set())
        community_chunks = route_chunks.get("community_aware", set())
        row["route_contributions"] = {
            "vector_chunks": sorted(vector_chunks), "graph_chunks": sorted(graph_chunks),
            "community_aware_chunks": sorted(community_chunks),
            "graph_unique_vs_vector": sorted(graph_chunks - vector_chunks),
            "hybrid_unique_vs_vector": sorted(hybrid_chunks - vector_chunks),
            "community_unique_vs_hybrid": sorted(community_chunks - hybrid_chunks),
            "shared_graph_vector": sorted(graph_chunks & vector_chunks),
            "final_hybrid_chunks": sorted(hybrid_chunks),
            "question_type": sample.get("type", "unknown"),
        }
        out["questions"].append(row)

    # Macro averages are computed only over successfully completed route records.
    aggregate = {}
    for route in ROUTES:
        records = [q["routes"][route] for q in out["questions"] if route in q.get("routes", {})]
        def mean(values): return sum(values) / len(values) if values else None
        aggregate[route] = {
            "completed": len(records),
            "failures": sum(1 for q in out["questions"] if any(e["stage"] == f"query:{route}" for e in q["errors"])),
            "mean_query_seconds": mean([r["query_seconds"] for r in records]),
            "median_query_seconds": statistics.median([r["query_seconds"] for r in records]) if records else None,
            "mean_answer_exact_match": mean([float(r["answer_metrics"]["exact_match"]) for r in records]),
            "mean_answer_token_f1": mean([r["answer_metrics"]["token_f1"] for r in records]),
            "abstention_rate": mean([float(r["answer_metrics"]["abstained"]) for r in records]),
            "invalid_graph_plan_rate": mean([float(r["invalid_plan"]) for r in records]),
            "mean_recall_at_2": mean([r["retrieval_metrics"]["recall_at_2"]["supporting_fact_recall_at_k"] or 0 for r in records]),
            "mean_recall_at_5": mean([r["retrieval_metrics"]["recall_at_5"]["supporting_fact_recall_at_k"] or 0 for r in records]),
            "complete_support_at_2": mean([float(r["retrieval_metrics"]["recall_at_2"]["complete_supporting_evidence_at_k"]) for r in records]),
            "complete_support_at_5": mean([float(r["retrieval_metrics"]["recall_at_5"]["complete_supporting_evidence_at_k"]) for r in records]),
        }
    out["aggregate"] = aggregate
    successful_index = [q.get("stages", {}).get("total_indexing_seconds") for q in out["questions"]
                        if q.get("stages", {}).get("total_indexing_seconds") is not None]
    out["aggregate"]["indexing_latency"] = {
        "mean_seconds": sum(successful_index) / len(successful_index) if successful_index else None,
        "median_seconds": statistics.median(successful_index) if successful_index else None,
    }
    by_type = {}
    for question_type in sorted({q.get("route_contributions", {}).get("question_type", "unknown") for q in out["questions"]}):
        typed = [q for q in out["questions"] if q.get("route_contributions", {}).get("question_type") == question_type]
        by_type[question_type] = {route: aggregate[route]["mean_recall_at_5"] for route in ROUTES}
    out["retrieval_recall_at_5_by_question_type"] = by_type

    out_path = Path(args.output) if args.output else Path("experiments/hotpotqa") / f"{args.context_mode}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {len(out['questions'])} question records to {out_path}")
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="adapters/datasets/hotpotqa/hotpot_smoke_20.json")
    p.add_argument("--config", default="configs/domains/academic.yaml")
    p.add_argument("--context-mode", choices=("oracle", "full"), default="full")
    p.add_argument("--limit", type=int, default=5, help="Five-question smoke test by default")
    p.add_argument("--input-root", default="experiments/hotpotqa/inputs")
    p.add_argument("--storage-root", default="knowledge_bases/hotpotqa_eval")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
