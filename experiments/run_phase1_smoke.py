"""Live Local Ollama Smoke Run Script executing 19-stage GraphRAG pipeline on real models."""

from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict
from src.core.configuration import load_engine_config
from src.engine import HybridGraphRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase1_smoke")


def setup_smoke_corpus() -> Dict[str, Path]:
    """Create sample smoke corpus for academic_kb and company_kb."""
    base_smoke_dir = Path("experiments/smoke_data")
    acad_dir = base_smoke_dir / "academic"
    comp_dir = base_smoke_dir / "company"

    acad_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    (acad_dir / "doc1.txt").write_text(
        "Rahul studies at IIIT Naya Raipur. Rahul works under Professor Sharma.",
        encoding="utf-8",
    )
    (acad_dir / "doc2.txt").write_text(
        "Professor Sharma supervises the Medical AI project.",
        encoding="utf-8",
    )
    (acad_dir / "doc3.txt").write_text(
        "Medical AI uses machine learning to detect diseases.",
        encoding="utf-8",
    )

    (comp_dir / "comp_doc1.txt").write_text(
        "Priya manages the CloudSec product in the Platform Department.",
        encoding="utf-8",
    )
    (comp_dir / "comp_doc2.txt").write_text(
        "CloudSec is built with Rust and serves Acme Corp.",
        encoding="utf-8",
    )

    return {"academic": acad_dir, "company": comp_dir}


def run_smoke_test() -> None:
    logger.info("Setting up smoke corpora...")
    corpus = setup_smoke_corpus()

    results_payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "llm": "gemma3:4b",
            "embedding": "embeddinggemma:latest",
        },
        "knowledge_bases": {},
        "isolation_checks": {},
    }

    # =========================================================================
    # 1. Academic Knowledge Base
    # =========================================================================
    logger.info("Initializing academic_kb engine with live Ollama...")
    acad_config = load_engine_config(
        domain_config_path=Path("configs/domains/academic.yaml"),
        kb_id_override="academic_kb",
    )
    acad_engine = HybridGraphRAGEngine(config=acad_config)

    logger.info("Starting live indexing of academic_kb...")
    t0 = time.time()
    acad_index_stats = acad_engine.index_directory(corpus["academic"])
    acad_index_time = time.time() - t0
    logger.info("academic_kb indexing completed in %.2fs: %s", acad_index_time, acad_index_stats)

    acad_queries = [
        {"type": "single_hop", "q": "Who supervises the Medical AI project?", "mode": "hybrid"},
        {"type": "multi_hop_3hop", "q": "What technology does the project supervised by Rahul's mentor use?", "mode": "hybrid"},
        {"type": "global_theme", "q": "Summarize the primary research themes across the projects.", "mode": "community_aware"},
        {"type": "alias_query", "q": "Who works under Prof. Sharma?", "mode": "hybrid"},
        {"type": "unanswerable", "q": "What is the capital of France?", "mode": "hybrid"},
    ]

    acad_query_results = []
    for item in acad_queries:
        logger.info("Running query [%s]: %s", item["type"], item["q"])
        t_q = time.time()
        ans, res = acad_engine.query(question=item["q"], mode=item["mode"])
        q_time = time.time() - t_q
        acad_query_results.append(
            {
                "query_type": item["type"],
                "question": item["q"],
                "mode": item["mode"],
                "answer": ans,
                "abstained": res.abstained,
                "abstention_reason": res.abstention_reason,
                "evidence_count": len(res.evidence_list),
                "evidence_chunks": [
                    {
                        "chunk_id": ev.chunk_id,
                        "doc_id": ev.doc_id,
                        "routes": [r.value for r in ev.routes],
                        "rrf_score": ev.rrf_score,
                    }
                    for ev in res.evidence_list
                ],
                "community_reports_count": len(res.community_reports),
                "duration_seconds": round(q_time, 2),
            }
        )

    results_payload["knowledge_bases"]["academic_kb"] = {
        "index_stats": acad_index_stats,
        "indexing_duration_seconds": round(acad_index_time, 2),
        "queries": acad_query_results,
    }

    # =========================================================================
    # 2. Company Knowledge Base
    # =========================================================================
    logger.info("Initializing company_kb engine with live Ollama...")
    comp_config = load_engine_config(
        domain_config_path=Path("configs/domains/company.yaml"),
        kb_id_override="company_kb",
    )
    comp_engine = HybridGraphRAGEngine(config=comp_config)

    logger.info("Starting live indexing of company_kb...")
    t0 = time.time()
    comp_index_stats = comp_engine.index_directory(corpus["company"])
    comp_index_time = time.time() - t0
    logger.info("company_kb indexing completed in %.2fs: %s", comp_index_time, comp_index_stats)

    comp_queries = [
        {"type": "domain_query", "q": "What is CloudSec built with?", "mode": "hybrid"},
        {"type": "manager_query", "q": "Who manages the CloudSec product?", "mode": "hybrid"},
    ]

    comp_query_results = []
    for item in comp_queries:
        logger.info("Running company query [%s]: %s", item["type"], item["q"])
        t_q = time.time()
        ans, res = comp_engine.query(question=item["q"], mode=item["mode"])
        q_time = time.time() - t_q
        comp_query_results.append(
            {
                "query_type": item["type"],
                "question": item["q"],
                "mode": item["mode"],
                "answer": ans,
                "abstained": res.abstained,
                "evidence_count": len(res.evidence_list),
                "evidence_chunks": [
                    {
                        "chunk_id": ev.chunk_id,
                        "doc_id": ev.doc_id,
                        "routes": [r.value for r in ev.routes],
                        "rrf_score": ev.rrf_score,
                    }
                    for ev in res.evidence_list
                ],
                "duration_seconds": round(q_time, 2),
            }
        )

    results_payload["knowledge_bases"]["company_kb"] = {
        "index_stats": comp_index_stats,
        "indexing_duration_seconds": round(comp_index_time, 2),
        "queries": comp_query_results,
    }

    # =========================================================================
    # 3. Cross-Domain Isolation Verification
    # =========================================================================
    logger.info("Verifying cross-domain isolation...")
    # Query academic_kb for company term "Priya"
    _, leak_1 = acad_engine.query("Who is Priya?", mode="vector_only")
    acad_leakage_detected = any("company" in ev.doc_id or "CloudSec" in ev.text for ev in leak_1.evidence_list)

    # Query company_kb for academic term "Rahul"
    _, leak_2 = comp_engine.query("Where does Rahul study?", mode="vector_only")
    comp_leakage_detected = any("academic" in ev.doc_id or "IIIT" in ev.text for ev in leak_2.evidence_list)

    results_payload["isolation_checks"] = {
        "academic_kb_isolated": not acad_leakage_detected,
        "company_kb_isolated": not comp_leakage_detected,
        "zero_cross_kb_leakage_confirmed": (not acad_leakage_detected) and (not comp_leakage_detected),
    }

    out_file = Path("experiments/self_generated/phase1_smoke_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Phase 1 smoke test results saved to %s", out_file)
    print("\n=== Phase 1 Live Smoke Test Completed Successfully ===")
    print(f"Results written to: {out_file}")
    print(f"Zero Cross-KB Leakage Confirmed: {results_payload['isolation_checks']['zero_cross_kb_leakage_confirmed']}")
    print("========================================================\n")


if __name__ == "__main__":
    run_smoke_test()
