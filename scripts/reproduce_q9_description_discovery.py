#!/usr/bin/env python3
"""Part 4: Real-model reproduction script for Q9 Animorphs description discovery.

Loads the persisted HotpotQA pilot KB strictly READ-ONLY.
Runs the Q9 query with entity_description_discovery_enabled=False and then =True.
Measures and prints:
- Resolved seeds
- Matching method & similarity scores
- Distractor candidates with similarity
- Resulting evidence list
- Gold-chunk recall

Example Invocation:
    python scripts/reproduce_q9_description_discovery.py \\
        --kb-id 5a85ea095542994775f606a8 \\
        --kb-root knowledge_bases/hotpotqa_pilot \\
        --base-url http://localhost:11434 \\
        --embed-model embeddinggemma:latest
"""

import argparse
import sys
from pathlib import Path
import yaml

from src.core.configuration import EngineConfig
from src.core.models import QueryIntent, QueryPlan, SeedEntitySpec
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.graph.graph_builder import IGraphBuilder
from src.graph.traversal import GraphTraverser
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-model reproduction script for Q9 Animorphs description discovery."
    )
    parser.add_argument(
        "--kb-id",
        type=str,
        required=True,
        help="Target Knowledge Base ID (example: 5a85ea095542994775f606a8)",
    )
    parser.add_argument(
        "--kb-root",
        type=str,
        required=True,
        help="Path to storage root containing the KB (example: knowledge_bases/hotpotqa_pilot)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Ollama base URL (example: http://localhost:11434; defaults to config_snapshot value if omitted)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Ollama embedding model (example: embeddinggemma:latest; defaults to config_snapshot value if omitted)",
    )
    return parser.parse_args()


def run_reproduction():
    args = parse_args()
    kb_id = args.kb_id
    kb_root = args.kb_root
    kb_dir = Path(kb_root) / kb_id
    config_path = kb_dir / "config_snapshot.yaml"

    if not kb_dir.exists():
        print(f"ERROR: Persisted KB not found at {kb_dir}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if args.base_url:
        config_data.setdefault("ollama", {})["base_url"] = args.base_url
    if args.embed_model:
        config_data.setdefault("ollama", {})["embedding_model"] = args.embed_model

    cfg = EngineConfig.model_validate(config_data)

    print("=" * 80)
    print("PART 4: REAL-MODEL REPRODUCTION (READ-ONLY)")
    print(f"KB ID: {kb_id}")
    print(f"KB Root: {kb_root}")
    print(f"Ollama base_url: {cfg.ollama.base_url}")
    print(f"Embedding model: {cfg.ollama.embedding_model}")
    print("=" * 80)

    # 1. Initialize stores strictly read-only
    doc_store = DocumentStore(
        kb_id=kb_id,
        storage_root=kb_root,
        base_dir=".",
    )
    vector_store = ChromaVectorStore(
        kb_id=kb_id,
        storage_root=kb_root,
        base_dir=".",
    )
    embedding_service = OllamaEmbeddingService(cfg.ollama)

    # Reconstruct graph from doc_store (read-only)
    graph = IGraphBuilder.reconstruct_from_store(doc_store)
    traverser = GraphTraverser()

    entities = doc_store.load_canonical_entities()
    chunks = doc_store.load_chunks()
    print(f"Loaded canonical entities: {len(entities)}")
    print(f"Loaded text chunks: {len(chunks)}")
    print(f"Graph vertices: {graph.vcount()}, edges: {graph.ecount()}")

    # 2. Query definition for HotpotQA Q9
    question = (
        "What science fantasy young adult series, told in first person, has a set "
        "of companion books narrating the stories of enslaved worlds and alien species?"
    )
    # The gold supporting documents for Q9:
    # - Animorphs (D009, chunk D009_C0000)
    # - The Hork-Bajir Chronicles (D003, chunk D003_C0000)
    gold_chunk_ids = {"D009_C0000", "D003_C0000"}

    # Seed specification produced by QueryPlanner for Q9
    seed_specs = [
        SeedEntitySpec(name="science fantasy young adult series", expected_type="BOOK")
    ]
    plan = QueryPlan(
        question=question,
        intent=QueryIntent.LOCAL,
        seed_entities=seed_specs,
        relationship_path=[],
    )

    results = {}

    for flag_enabled in [False, True]:
        flag_label = "ENABLED" if flag_enabled else "DISABLED"
        print("\n" + "-" * 80)
        print(f"RUNNING WITH entity_description_discovery_enabled={flag_enabled} ({flag_label})")
        print("-" * 80)

        seed_retriever = SeedEntityRetriever(
            doc_store=doc_store,
            vector_store=vector_store,
            embedding_provider=embedding_service,
            similarity_threshold=0.82,
            entity_description_discovery_enabled=flag_enabled,
            entity_description_similarity_threshold=0.65,
            max_description_candidates_per_seed=3,
        )

        graph_retriever = GraphEvidenceRetriever(
            doc_store=doc_store,
            seed_retriever=seed_retriever,
            graph=graph,
            traverser=traverser,
            entity_description_discovery_enabled=flag_enabled,
        )

        # Execute seed resolution
        seed_res = seed_retriever.resolve_seeds(plan.seed_entities)

        # Execute retrieval
        evidence = graph_retriever.retrieve(
            question=question,
            plan=plan,
            kb_id=kb_id,
        )

        retrieved_chunk_ids = [ev.chunk_id for ev in evidence]
        gold_recovered = [cid for cid in retrieved_chunk_ids if cid in gold_chunk_ids]
        recall = len(gold_recovered) / len(gold_chunk_ids)

        print(f"Resolved Seed IDs: {seed_res.resolved_entity_ids}")
        print(f"Unresolved Seeds: {seed_res.unresolved_seeds}")
        print(f"Ambiguous Seeds: {seed_res.ambiguous_seeds}")
        print(f"Description Candidates Count: {len(seed_res.description_candidates)}")

        print("\nResolution Diagnostics:")
        distractors = []
        for d in seed_res.resolution_diagnostics:
            cand_info = f"ID={d.candidate_id}, Name='{d.candidate_name}', Type={d.candidate_type}"
            sim_str = f"sim={d.similarity_score}" if d.similarity_score is not None else "sim=None"
            conf_str = f"conf={d.confidence_score}"
            print(f"  [{d.status}] {d.matching_method or 'N/A'}: {cand_info} | {sim_str} | {conf_str} | Reason: {d.reason}")
            if d.candidate_id and d.candidate_id != "E019":
                distractors.append((d.candidate_id, d.candidate_name, d.candidate_type, d.similarity_score, d.status))

        print(f"\nDistractor Candidates (non-Animorphs):")
        for cid, cname, ctype, sim, status in distractors[:5]:
            print(f"  - {cid}: '{cname}' ({ctype}) | sim={sim} | status={status}")

        print(f"\nResulting Evidence Chunks ({len(evidence)}):")
        for ev in evidence:
            prov_types = [p.value for p in ev.graph_provenance_types]
            print(f"  - Chunk ID: {ev.chunk_id} | Doc ID: {ev.doc_id} | Provenance: {prov_types}")
            if ev.entity_description_details:
                for prov in ev.entity_description_details:
                    print(f"    * Description Prov: seed='{prov.seed_name}' -> candidate='{prov.candidate_name}' ({prov.candidate_id}), sim={prov.similarity_score}")

        print(f"\nGold Chunks Target: {sorted(list(gold_chunk_ids))}")
        print(f"Gold Chunks Recovered: {sorted(gold_recovered)}")
        print(f"Gold-Chunk Recall: {recall * 100:.1f}% ({len(gold_recovered)}/{len(gold_chunk_ids)})")

        results[flag_enabled] = {
            "resolved_seeds": list(seed_res.resolved_entity_ids),
            "diagnostics": [d.model_dump() for d in seed_res.resolution_diagnostics],
            "distractors": distractors,
            "evidence_chunk_ids": retrieved_chunk_ids,
            "gold_recovered": gold_recovered,
            "gold_recall": recall,
        }

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    r_off = results[False]
    r_on = results[True]
    print(f"Flag OFF: resolved_seeds={r_off['resolved_seeds']}, gold_recall={r_off['gold_recall']*100:.1f}%, evidence_chunks={len(r_off['evidence_chunk_ids'])}")
    print(f"Flag ON:  resolved_seeds={r_on['resolved_seeds']}, gold_recall={r_on['gold_recall']*100:.1f}%, evidence_chunks={len(r_on['evidence_chunk_ids'])}")

    return results


if __name__ == "__main__":
    run_reproduction()
