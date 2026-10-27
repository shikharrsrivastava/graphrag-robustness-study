"""Compare full re-ingestion with basic incremental re-ingestion.

Workflow:
1. Run PREPARE while the PDF is still in its original state.
2. Add a paragraph to the same PDF and save it.
3. Run COMPARE to time full and incremental re-ingestion separately.

This script uses two isolated ChromaDB collections, so the full-ingestion run
cannot affect the baseline used by the incremental run. It does not modify the
production `pdf_chunks` collection used by rag.py.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import chromadb
import ollama

from rag import (
    DB_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    batches,
    embed_texts,
    read_pdfs,
)

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "basic_incremental_benchmark.json"
FULL_COLLECTION_NAME = "pdf_chunks_full_benchmark"
INCREMENTAL_COLLECTION_NAME = "pdf_chunks_incremental_benchmark"
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def content_hash(text: str) -> str:
    """Create a repeatable fingerprint of a chunk's actual text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_clean_collection(chroma_client, name: str):
    """Delete and recreate only this experiment's collection."""
    try:
        chroma_client.delete_collection(name=name)
    except Exception:
        # The collection may not exist yet. Continue by creating it.
        pass
    return chroma_client.create_collection(
        name=name,
        metadata=COLLECTION_METADATA,
    )


def get_existing_records(collection) -> dict[str, Any]:
    """Fetch IDs, texts, metadata and vectors currently stored in a collection."""
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    records: dict[str, Any] = {}
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")

    for index, record_id in enumerate(ids):
        records[record_id] = {
            "document": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "embedding": embeddings[index].tolist()
            if embeddings is not None and hasattr(embeddings[index], "tolist")
            else (list(embeddings[index]) if embeddings is not None else None),
        }
    return records


def prepare_baseline(ollama_client, chroma_client) -> None:
    """Index the unmodified PDF(s), measure original ingestion, seed both test DBs."""
    print("\nPREPARE: do this BEFORE adding the new paragraph.")
    print("This resets only the two benchmark collections, not rag.py's pdf_chunks collection.")
    full_collection = get_clean_collection(chroma_client, FULL_COLLECTION_NAME)
    incremental_collection = get_clean_collection(chroma_client, INCREMENTAL_COLLECTION_NAME)

    start = time.perf_counter()
    texts, ids, metadatas = read_pdfs()
    vectors = embed_texts(ollama_client, texts)
    full_collection.upsert(
        ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors
    )
    original_seconds = time.perf_counter() - start

    # Seed the incremental benchmark from exactly the same pre-edit snapshot.
    incremental_collection.upsert(
        ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors
    )

    state = {
        "baseline_ready": True,
        "original_ingestion_seconds": original_seconds,
        "original_chunk_count": len(texts),
        "pdf_names": sorted({metadata["source"] for metadata in metadatas}),
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"Original ingestion time: {original_seconds:.3f} seconds")
    print(f"Original chunks: {len(texts)}")
    print("Baseline saved to both benchmark collections.")
    print("NEXT: add a new paragraph to the same PDF, save it, then choose COMPARE.")


def remove_stale_ids(collection, current_ids: set[str], old_ids: set[str]) -> int:
    """Delete records that were stored previously but are absent from current PDFs."""
    stale_ids = sorted(old_ids - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(stale_ids)


def full_reingest(ollama_client, collection) -> dict[str, Any]:
    """Reprocess and re-embed every current chunk; remove stale records afterwards."""
    start = time.perf_counter()
    old_ids = set(collection.get(include=["metadatas"]).get("ids") or [])
    texts, ids, metadatas = read_pdfs()
    vectors = embed_texts(ollama_client, texts)

    collection.upsert(
        ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors
    )
    stale_deleted = remove_stale_ids(collection, set(ids), old_ids)
    elapsed = time.perf_counter() - start

    return {
        "elapsed_seconds": elapsed,
        "current_chunks": len(texts),
        "embeddings_generated": len(texts),
        "embeddings_reused": 0,
        "stale_chunks_deleted": stale_deleted,
    }


def incremental_reingest(ollama_client, collection) -> dict[str, Any]:
    """Re-chunk current PDFs, but reuse vectors for unchanged chunk text."""
    start = time.perf_counter()
    old_records = get_existing_records(collection)
    texts, ids, metadatas = read_pdfs()
    current_ids = set(ids)

    # Index old vectors by content hash. This can reuse a vector even if the
    # same text received a different chunk ID after nearby content shifted.
    old_vectors_by_hash: dict[str, list[float]] = {}
    for record in old_records.values():
        old_text = record["document"]
        old_vector = record["embedding"]
        if old_text and old_vector is not None:
            old_vectors_by_hash.setdefault(content_hash(old_text), old_vector)

    to_upsert_ids: list[str] = []
    to_upsert_texts: list[str] = []
    to_upsert_metadatas: list[dict[str, str | int]] = []
    to_upsert_vectors: list[list[float] | None] = []
    reused = 0
    need_embedding_indices: list[int] = []

    for index, (text, record_id, metadata) in enumerate(zip(texts, ids, metadatas)):
        existing = old_records.get(record_id)
        if existing is not None and existing["document"] == text:
            # Same ID and same content: no embedding or database update needed.
            reused += 1
            continue

        # Content may be unchanged but the ID may have shifted. If so, reuse
        # its old vector and upsert under the current ID.
        vector = old_vectors_by_hash.get(content_hash(text))
        to_upsert_ids.append(record_id)
        to_upsert_texts.append(text)
        to_upsert_metadatas.append(metadata)
        to_upsert_vectors.append(vector)
        if vector is None:
            need_embedding_indices.append(len(to_upsert_ids) - 1)

    # Generate vectors only for new/changed text for which no old matching
    # content vector was found. Batch embedding remains consistent with rag.py.
    new_texts = [to_upsert_texts[i] for i in need_embedding_indices]
    new_vectors = embed_texts(ollama_client, new_texts) if new_texts else []
    for target_index, vector in zip(need_embedding_indices, new_vectors):
        to_upsert_vectors[target_index] = vector

    if to_upsert_ids:
        # Every vector is filled either from the old content or from new embedding.
        collection.upsert(
            ids=to_upsert_ids,
            documents=to_upsert_texts,
            metadatas=to_upsert_metadatas,
            embeddings=[vector for vector in to_upsert_vectors if vector is not None],
        )

    stale_deleted = remove_stale_ids(collection, current_ids, set(old_records))
    elapsed = time.perf_counter() - start
    return {
        "elapsed_seconds": elapsed,
        "current_chunks": len(texts),
        "embeddings_generated": len(new_texts),
        "embeddings_reused": reused + (len(to_upsert_ids) - len(new_texts)),
        "stale_chunks_deleted": stale_deleted,
    }


def compare_strategies(ollama_client, chroma_client) -> None:
    if not STATE_FILE.exists():
        print("No baseline found. Choose PREPARE before editing the PDF.")
        return

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not state.get("baseline_ready"):
        print("Baseline is not ready. Choose PREPARE first.")
        return

    full_collection = chroma_client.get_collection(FULL_COLLECTION_NAME)
    incremental_collection = chroma_client.get_collection(INCREMENTAL_COLLECTION_NAME)

    print("\nComparing updated PDFs against the same pre-edit baseline...\n")
    full = full_reingest(ollama_client, full_collection)
    incremental = incremental_reingest(ollama_client, incremental_collection)
    original_seconds = float(state["original_ingestion_seconds"])

    print("=== TIMING RESULTS ===")
    print(f"Original ingestion (before edit): {original_seconds:.3f} s")
    print(f"Full re-ingestion (after edit):   {full['elapsed_seconds']:.3f} s")
    print(f"Basic incremental (after edit):   {incremental['elapsed_seconds']:.3f} s")
    print()
    print("=== WORK PERFORMED ===")
    print(
        "Full:        "
        f"chunks={full['current_chunks']}, "
        f"embeddings generated={full['embeddings_generated']}, "
        f"stale deleted={full['stale_chunks_deleted']}"
    )
    print(
        "Incremental: "
        f"chunks inspected={incremental['current_chunks']}, "
        f"embeddings generated={incremental['embeddings_generated']}, "
        f"embeddings reused={incremental['embeddings_reused']}, "
        f"stale deleted={incremental['stale_chunks_deleted']}"
    )

    full_time = full["elapsed_seconds"]
    incremental_time = incremental["elapsed_seconds"]
    if full_time > 0:
        saved = full_time - incremental_time
        reduction = saved / full_time * 100
        print(f"\nIncremental vs full re-ingestion time saved: {saved:.3f} s ({reduction:.1f}%)")
    print("\nNote: times are measurements from this run, not guaranteed in advance.")
    print("For reliable results, repeat the comparison several times with the same edit and setup.")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ollama_client = ollama.Client()
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    print("Basic incremental-ingestion experiment")
    print(f"PDF folder: {DATA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding batch size: {EMBED_BATCH_SIZE}")
    print("Production collection 'pdf_chunks' is not modified by this experiment.")

    while True:
        print("\nChoose: [P] Prepare baseline (before editing PDF)  [C] Compare (after editing PDF)  [X] Exit")
        choice = input("> ").strip().lower()
        if choice == "p":
            try:
                prepare_baseline(ollama_client, chroma_client)
            except Exception as exc:
                print(f"Could not prepare baseline: {exc}")
                print("Check that PDFs are in data/, Ollama is running, and embeddinggemma is available.")
        elif choice == "c":
            try:
                compare_strategies(ollama_client, chroma_client)
            except Exception as exc:
                print(f"Could not compare strategies: {exc}")
                print("If an error occurred during comparison, run PREPARE again before retrying.")
        elif choice == "x":
            break
        else:
            print("Please enter P, C, or X.")


if __name__ == "__main__":
    main()
