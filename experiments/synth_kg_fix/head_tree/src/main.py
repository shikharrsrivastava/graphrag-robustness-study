"""CLI entrypoint for the Hybrid GraphRAG System."""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from src.core.configuration import load_engine_config
from src.engine import HybridGraphRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("graphrag_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid GraphRAG System: 19-Stage Multi-Domain Knowledge Base Engine"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the domain YAML configuration file (defaults to matching domain or configs/domains/academic.yaml)",
    )
    parser.add_argument(
        "--index-dir",
        "-i",
        type=str,
        help="Path to directory containing raw documents (.pdf, .txt, .md) to index into the knowledge base",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Natural language question to query the knowledge base",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["hybrid", "vector_only", "graph_only", "community_aware"],
        default="hybrid",
        help="Retrieval mode: hybrid (default), vector_only, graph_only, or community_aware",
    )
    parser.add_argument(
        "--scope-id",
        type=str,
        default=None,
        help="Optional scope filter to isolate chunks and traversal paths",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    elif args.index_dir and "company" in args.index_dir.lower():
        config_path = Path("configs/domains/company.yaml")
    elif args.index_dir and "legal" in args.index_dir.lower():
        config_path = Path("configs/domains/legal.yaml")
    elif args.index_dir and "medical" in args.index_dir.lower():
        config_path = Path("configs/domains/medical_research.yaml")
    else:
        config_path = Path("configs/domains/academic.yaml")

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    logger.info("Loading engine configuration from %s...", config_path)
    config = load_engine_config(config_path)

    engine = HybridGraphRAGEngine(config=config)

    # 1. Execute indexing if --index-dir provided
    if args.index_dir:
        input_dir = Path(args.index_dir)
        logger.info("Starting indexing for directory %s into KB '%s'...", input_dir, config.kb_id)
        index_stats = engine.index_directory(input_dir)
        print("\n=== Indexing Summary ===")
        print(json.dumps(index_stats, indent=2))
        print("========================\n")

    # 2. Execute query if --query provided
    if args.query:
        logger.info("Executing query in '%s' mode: %s", args.mode, args.query)
        answer, result = engine.query(question=args.query, mode=args.mode, scope_id=args.scope_id)
        print("\n=== Query Result ===")
        print(f"Question: {result.question}")
        print(f"Abstained: {result.abstained}")
        if result.abstained:
            print(f"Reason: {result.abstention_reason}")
        print(f"\nAnswer:\n{answer}")
        print(f"\nEvidence Count: {len(result.evidence_list)}")
        for i, ev in enumerate(result.evidence_list[:5], 1):
            routes_str = "+".join(r.value for r in ev.routes)
            print(f"  [{i}] Chunk {ev.chunk_id} (Doc: {ev.doc_id}) | Routes: {routes_str} | RRF Score: {ev.rrf_score:.5f}")
        print("====================\n")

    if not args.index_dir and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
