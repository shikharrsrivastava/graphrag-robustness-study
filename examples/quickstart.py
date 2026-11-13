"""Quickstart example demonstrating the public Python API for Hybrid GraphRAG.

This example illustrates:
1. Creating or opening a knowledge base using `HybridGraphRAG`.
2. Ingesting text documents into an isolated knowledge base.
3. Querying the engine with hybrid retrieval.
4. Inspecting the structured answer, route metadata, and verified citations.
5. Reopening an existing persisted knowledge base without re-ingesting documents.
"""

from pathlib import Path
import tempfile

from hybrid_graphrag import HybridGraphRAG, QueryResult


def run_quickstart():
    # 1. Prepare temporary directory and sample documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "sample_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        doc1 = docs_dir / "quantum_computing.txt"
        doc1.write_text(
            "Quantum computers leverage quantum mechanical phenomena like superposition and entanglement "
            "to perform computations exponentially faster for specific problems than classical machines. "
            "Google Quantum AI and IBM Quantum are leading research institutions in superconducting qubits.",
            encoding="utf-8",
        )

        kb_storage = temp_path / "kbs"
        kb_storage.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "storage_root": str(kb_storage),
            "prompts_dir": "prompts",
            "chunking": {"chunk_size": 500, "chunk_overlap": 50},
        }

        print("=== Step 1: Initializing Knowledge Base ===")
        # Use context manager to guarantee automatic resource cleanup
        with HybridGraphRAG.from_config(config_dict, kb_id="quantum_research") as engine:
            print(f"Engine initialized with KB ID: {engine.kb_id}")

            # 2. Ingest sample documents
            print("\n=== Step 2: Ingesting Documents ===")
            stats = engine.ingest(docs_dir)
            print(f"Ingestion complete: {stats}")

            # 3. Query the knowledge base
            print("\n=== Step 3: Querying the Engine ===")
            query_str = "What organizations lead superconducting qubit research?"
            print(f"Question: {query_str}")

            try:
                result: QueryResult = engine.query(query_str, route="hybrid")
                print(f"Answer: {result.answer}")
                print(f"Route: {result.route}")
                print(f"Abstained: {result.abstained}")
                print(f"Latency: {result.latency_seconds:.3f}s")
                print("\nCitations:")
                for citation in result.citations:
                    print(f"  - [{citation.citation_id}] Doc: {citation.doc_id}, Chunk: {citation.chunk_id}")
                    print(f"    Text: {citation.text[:80]}...")
            except Exception as e:
                print(f"Query note (live LLM might not be running): {e}")

            kb_info = engine.get_kb_stats()
            print(f"\nKB Stats: {kb_info}")

        # 4. Reopening the persisted knowledge base without re-indexing
        print("\n=== Step 4: Reopening Persisted Knowledge Base ===")
        persisted_kb_path = kb_storage / "quantum_research"
        with HybridGraphRAG.from_existing(persisted_kb_path) as restored_engine:
            print(f"Successfully reopened KB at {restored_engine.kb_dir}")
            restored_stats = restored_engine.get_kb_stats()
            print(f"Restored KB Document count: {restored_stats.get('document_count')}")


if __name__ == "__main__":
    run_quickstart()
