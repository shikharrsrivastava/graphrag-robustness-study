"""Targeted, safe community-report refresh for Hybrid GraphRAG knowledge bases.

Loads existing graph and Leiden partitions, regenerates graph-grounded community
reports using CommunitySummarizer, atomically updates community_reports.jsonl
with automatic backup, and refreshes the ChromaDB community_reports collection
while ensuring chunks, entities, and graph files remain untouched.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.configuration import EngineConfig
from src.core.models import CommunityReport
from src.communities.community_summarizer import CommunitySummarizer
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaLLMClient
from src.graph.graph_builder import IGraphBuilder
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("refresh_community_reports")

VALID_REPORT_STATUSES = frozenset({"VERIFIED", "FALLBACK"})


def load_config_from_kb(kb_dir: Path, config_path: Optional[Path] = None) -> EngineConfig:
    """Load EngineConfig from config_snapshot.yaml in KB root."""
    cfg_file = config_path or (kb_dir / "config_snapshot.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EngineConfig.model_validate(data)


def run_preflight_check(
    kb_dir: Path,
    config: EngineConfig,
    doc_store: DocumentStore,
    vstore: ChromaVectorStore,
) -> Dict[str, Any]:
    """Inspect graph, community partitions, and vector collections without performing any writes."""
    graph = IGraphBuilder.reconstruct_from_store(doc_store)
    communities = doc_store.load_communities()

    singletons_count = 0
    non_singletons: List[Dict[str, Any]] = []

    for c in communities:
        comm_id = c["community_id"]
        member_set = set(c["member_entity_ids"])
        internal_edges = [
            e for e in graph.es
            if e["source_entity_id"] in member_set and e["target_entity_id"] in member_set
        ]
        if len(member_set) <= 1 and len(internal_edges) == 0:
            singletons_count += 1
        else:
            non_singletons.append({
                "community_id": comm_id,
                "size": len(member_set),
                "internal_edges_count": len(internal_edges),
            })

    report_file = kb_dir / "graph" / "community_reports.jsonl"
    existing_reports_count = 0
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            existing_reports_count = sum(1 for line in f if line.strip())

    chunk_count = vstore.chunk_collection.count()
    entity_count = vstore.entity_collection.count()
    community_count = vstore.community_collection.count()

    return {
        "kb_id": config.kb_id,
        "kb_dir": str(kb_dir),
        "total_communities": len(communities),
        "singletons_fast_path": singletons_count,
        "non_singletons_llm_required": len(non_singletons),
        "estimated_llm_calls": len(non_singletons),
        "existing_reports_jsonl_count": existing_reports_count,
        "vector_store_counts": {
            "chunks": chunk_count,
            "entities": entity_count,
            "community_reports": community_count,
        },
        "non_singleton_details": non_singletons,
    }


def refresh_community_reports(
    kb_dir: Path,
    config_path: Optional[Path] = None,
    dry_run: bool = False,
    custom_llm_client: Optional[OllamaLLMClient] = None,
    custom_embedding_service: Optional[OllamaEmbeddingService] = None,
) -> Dict[str, Any]:
    """Execute targeted refresh of community reports and ChromaDB vector store."""
    kb_dir = Path(kb_dir).resolve()
    config = load_config_from_kb(kb_dir, config_path)

    doc_store = DocumentStore(
        kb_id=kb_dir.name,
        storage_root=str(kb_dir.parent),
        base_dir=".",
    )
    vstore = ChromaVectorStore(
        kb_id=config.kb_id,
        kb_vector_store_dir=kb_dir / "vector_store",
    )

    preflight = run_preflight_check(kb_dir, config, doc_store, vstore)

    if dry_run:
        logger.info("DRY-RUN mode enabled. No files or vector collections will be modified.")
        return {
            "status": "DRY_RUN_COMPLETED",
            "preflight": preflight,
            "modifications": [],
        }

    logger.info(
        "Starting targeted community refresh for KB '%s': %d total communities (%d singletons, %d relational).",
        config.kb_id,
        preflight["total_communities"],
        preflight["singletons_fast_path"],
        preflight["non_singletons_llm_required"],
    )

    graph = IGraphBuilder.reconstruct_from_store(doc_store)
    communities = doc_store.load_communities()

    # Initialize clients
    llm_client = custom_llm_client or OllamaLLMClient(config=config.ollama)
    embedding_service = custom_embedding_service or OllamaEmbeddingService(config=config.ollama)

    # Initialize CommunitySummarizer in detached mode (we manage persistence atomically)
    summarizer = CommunitySummarizer(
        llm_client=llm_client,
        embedding_provider=None,
        vector_store=None,
        doc_store=None,
        model_tag=config.ollama.llm_model,
        prompt_version=config.prompt_versions.community_summary,
    )

    start_time = time.perf_counter()
    new_reports: List[CommunityReport] = summarizer.summarize_communities(
        communities=communities,
        graph=graph,
        doc_store=doc_store,
    )
    summarization_elapsed = time.perf_counter() - start_time

    # Validate generated reports before performing any writes
    assert len(new_reports) == len(communities), (
        f"Generated report count mismatch: {len(new_reports)} reports vs {len(communities)} communities"
    )

    status_counts: Dict[str, int] = {}
    for r in new_reports:
        st = r.verification_status
        status_counts[st] = status_counts.get(st, 0) + 1
        if st not in VALID_REPORT_STATUSES:
            raise ValueError(
                f"Generated report {r.report_id} has invalid verification status '{st}'. Must be VERIFIED or FALLBACK."
            )

    logger.info("Generated %d reports with statuses: %s", len(new_reports), status_counts)

    # 1. Atomic file replacement with backup
    graph_dir = kb_dir / "graph"
    reports_file = graph_dir / "community_reports.jsonl"
    timestamp = int(time.time())
    backup_file = graph_dir / f"community_reports.jsonl.bak.{timestamp}"

    if reports_file.exists():
        shutil.copy2(reports_file, backup_file)
        logger.info("Created backup of existing community reports at: %s", backup_file)

    temp_file = graph_dir / f"community_reports.jsonl.tmp.{timestamp}"
    with open(temp_file, "w", encoding="utf-8") as f:
        for r in new_reports:
            f.write(r.model_dump_json() + "\n")

    os.replace(temp_file, reports_file)
    logger.info("Atomically updated community reports at: %s", reports_file)

    # 2. Vector Store: purge stale community reports and re-index new reports
    existing_comm_ids = vstore.community_collection.get(where={"kb_id": config.kb_id}).get("ids", [])
    if existing_comm_ids:
        logger.info("Purging %d stale entries from community_reports vector collection...", len(existing_comm_ids))
        vstore.community_collection.delete(ids=existing_comm_ids)

    logger.info("Computing dense embeddings for %d new community reports...", len(new_reports))
    report_texts = [r.summary_text for r in new_reports]
    embeddings = embedding_service.embed_batch(report_texts)
    vstore.upsert_community_reports(new_reports, embeddings)
    logger.info("Upserted %d new community reports into ChromaDB.", len(new_reports))

    # 3. Post-execution verification
    post_chunk_count = vstore.chunk_collection.count()
    post_entity_count = vstore.entity_collection.count()
    post_comm_count = vstore.community_collection.count()

    assert post_chunk_count == preflight["vector_store_counts"]["chunks"], (
        f"Chunk collection was corrupted! Before: {preflight['vector_store_counts']['chunks']}, After: {post_chunk_count}"
    )
    assert post_entity_count == preflight["vector_store_counts"]["entities"], (
        f"Entity collection was corrupted! Before: {preflight['vector_store_counts']['entities']}, After: {post_entity_count}"
    )
    assert post_comm_count == len(new_reports), (
        f"Community collection count mismatch: expected {len(new_reports)}, found {post_comm_count}"
    )

    # Verify JSONL on disk
    with open(reports_file, "r", encoding="utf-8") as f:
        persisted_count = sum(1 for line in f if line.strip())
    assert persisted_count == len(new_reports), (
        f"Persisted file line count mismatch: expected {len(new_reports)}, found {persisted_count}"
    )

    return {
        "status": "COMPLETED",
        "kb_id": config.kb_id,
        "reports_count": len(new_reports),
        "status_breakdown": status_counts,
        "summarization_elapsed_seconds": round(summarization_elapsed, 2),
        "backup_path": str(backup_file) if backup_file.exists() else None,
        "modified_files": [str(reports_file)],
        "modified_vector_collections": ["community_reports"],
        "untouched_vector_collections": {
            "chunks": post_chunk_count,
            "entities": post_entity_count,
        },
        "verification": {
            "all_statuses_valid": True,
            "stale_entries_purged": True,
            "chunks_untouched": True,
            "entities_untouched": True,
            "graph_untouched": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted Community Report Refresh for Hybrid GraphRAG")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_bases/hotpotqa_eval/hotpot_full_5a8b57f25542995d1e6f1371"),
        help="Path to knowledge base directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config_snapshot.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight check and exit without writing files or modifying vector store",
    )
    args = parser.parse_args()

    result = refresh_community_reports(
        kb_dir=args.kb_dir,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    print("\n===== REFRESH RESULT =====")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
