#!/usr/bin/env python3
"""Compare exact FAISS search with IVF and HNSW using Ollama EmbeddingGemma.

Input file: faiss_data.json
{
  "documents": ["passage 1", "passage 2", ...],
  "queries": ["question 1", "question 2", ...]
}

All three indexes use normalized vectors + inner product, which is cosine
similarity. Flat search is the exact-neighbour ground truth for recall@k.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import ollama

MODEL = "embeddinggemma"
DATA_FILE = Path("faiss_data.json")
TOP_K = 5
EMBED_BATCH_SIZE = 32


def load_data(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Create it with 'documents' and 'queries' lists."
        )

    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object with 'documents' and 'queries'.")

    documents = payload.get("documents")
    queries = payload.get("queries")

    if not isinstance(documents, list) or not documents:
        raise ValueError("'documents' must be a non-empty list of strings.")
    if not isinstance(queries, list) or not queries:
        raise ValueError("'queries' must be a non-empty list of strings.")
    if not all(isinstance(x, str) and x.strip() for x in documents + queries):
        raise ValueError("Every document and query must be a non-empty string.")

    return documents, queries


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed in batches to avoid sending a very large request to Ollama."""
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = ollama.embed(model=MODEL, input=batch)
        batch_embeddings = response.get("embeddings")
        if batch_embeddings is None or len(batch_embeddings) != len(batch):
            raise RuntimeError("Ollama returned an unexpected embedding response.")
        all_embeddings.extend(batch_embeddings)
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} texts")

    vectors = np.asarray(all_embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise RuntimeError(f"Expected a 2D embedding matrix; got shape {vectors.shape}")
    return vectors


def build_indexes(x: np.ndarray) -> tuple[dict[str, Any], dict[str, str]]:
    n, dimension = x.shape
    indexes: dict[str, Any] = {}
    notes: dict[str, str] = {}

    # Exact baseline: checks every stored vector.
    flat = faiss.IndexFlatIP(dimension)
    flat.add(x)
    indexes["Flat exact"] = flat
    notes["Flat exact"] = "Exact search; used as the recall@k reference."

    # IVF: train a coarse clustering model, then add the vectors to its lists.
    # Keep roughly 39+ training vectors per centroid where corpus size permits.
    nlist = max(1, min(8, n // 39))
    quantizer = faiss.IndexFlatIP(dimension)
    ivf = faiss.IndexIVFFlat(
        quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
    )
    ivf.train(x)
    ivf.add(x)
    ivf.nprobe = max(1, min(nlist, max(1, nlist // 4)))
    indexes[f"IVF (nlist={nlist}, nprobe={ivf.nprobe})"] = ivf
    notes[f"IVF (nlist={nlist}, nprobe={ivf.nprobe})"] = (
        "Trained centroid partitions; searches only the selected inverted lists."
    )

    # HNSW: constructs a navigable neighbour graph while vectors are added.
    hnsw = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.hnsw.efSearch = min(64, max(16, n))
    hnsw.add(x)
    indexes[f"HNSW (M=32, efSearch={hnsw.hnsw.efSearch})"] = hnsw
    notes[f"HNSW (M=32, efSearch={hnsw.hnsw.efSearch})"] = (
        "Navigates a graph of vector neighbours; efSearch controls search effort."
    )

    return indexes, notes


def search_one(index: Any, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    scores, ids = index.search(q.reshape(1, -1), k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scores[0], ids[0], elapsed_ms


def main() -> int:
    try:
        documents, queries = load_data(DATA_FILE)
    except Exception as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nDocuments: {len(documents)} | Queries: {len(queries)}")
    print(f"Embedding model: {MODEL}")

    try:
        document_vectors = embed_texts(documents)
        query_vectors = embed_texts(queries)
    except Exception as exc:
        print(
            f"EMBEDDING ERROR: {exc}\n"
            "Check that Ollama is running and that `ollama pull embeddinggemma` succeeded.",
            file=sys.stderr,
        )
        return 1

    if document_vectors.shape[1] != query_vectors.shape[1]:
        print("Document and query embedding dimensions do not match.", file=sys.stderr)
        return 1

    # Normalize both sides so inner product is cosine similarity.
    faiss.normalize_L2(document_vectors)
    faiss.normalize_L2(query_vectors)

    # Save embeddings so later runs can reuse/inspect the exact vectors.
    np.save("faiss_document_vectors.npy", document_vectors)
    np.save("faiss_query_vectors.npy", query_vectors)

    indexes, notes = build_indexes(document_vectors)
    k = min(TOP_K, len(documents))

    print(f"\nVector dimension: {document_vectors.shape[1]}")
    print(f"Top-k: {k}")
    print("\nIndex setup:")
    for name, note in notes.items():
        print(f"- {name}: {note}")

    # Exact Flat results are the reference nearest neighbours.
    exact_ids: list[set[int]] = []
    for q in query_vectors:
        _, ids, _ = search_one(indexes["Flat exact"], q, k)
        exact_ids.append({int(i) for i in ids if i >= 0})

    summary: list[tuple[str, float, float]] = []
    all_results: dict[str, list[dict[str, Any]]] = {}

    print("\n" + "=" * 78)
    print("SEARCH RESULTS")
    print("=" * 78)

    for name, index in indexes.items():
        per_query_ms: list[float] = []
        per_query_recall: list[float] = []
        method_results: list[dict[str, Any]] = []

        for qi, (query, q_vector) in enumerate(zip(queries, query_vectors)):
            scores, ids, elapsed_ms = search_one(index, q_vector, k)
            valid = [(int(i), float(s)) for i, s in zip(ids, scores) if i >= 0]
            found_ids = {doc_id for doc_id, _ in valid}
            denominator = max(1, len(exact_ids[qi]))
            recall = len(found_ids.intersection(exact_ids[qi])) / denominator

            per_query_ms.append(elapsed_ms)
            per_query_recall.append(recall)
            method_results.append(
                {
                    "query": query,
                    "results": [
                        {"document_id": doc_id, "score": score,
                         "text_preview": documents[doc_id][:220]}
                        for doc_id, score in valid
                    ],
                    "recall_at_k_vs_flat": recall,
                    "latency_ms": elapsed_ms,
                }
            )

        avg_recall = statistics.mean(per_query_recall)
        avg_latency = statistics.mean(per_query_ms)
        summary.append((name, avg_recall, avg_latency))
        all_results[name] = method_results

    print(f"{'Index':43} {'Recall@k':>10} {'Avg ms/query':>14}")
    print("-" * 70)
    for name, recall, latency in summary:
        print(f"{name:43} {recall:10.3f} {latency:14.3f}")

    print("\nExample retrievals for the first query:")
    print(f"QUERY: {queries[0]}\n")
    for name, method_results in all_results.items():
        print(f"[{name}]")
        for item in method_results[0]["results"]:
            print(
                f"  doc_id={item['document_id']:>4}  "
                f"score={item['score']:.4f}  {item['text_preview']}"
            )
        print()

    output = {
        "model": MODEL,
        "document_count": len(documents),
        "query_count": len(queries),
        "dimension": int(document_vectors.shape[1]),
        "top_k": k,
        "metric": "cosine similarity (normalized vectors + inner product)",
        "summary": [
            {"index": name, "mean_recall_at_k_vs_flat": recall,
             "mean_latency_ms_per_query": latency}
            for name, recall, latency in summary
        ],
        "results": all_results,
    }
    with open("faiss_ivf_hnsw_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Saved: faiss_ivf_hnsw_results.json")
    print("Saved: faiss_document_vectors.npy and faiss_query_vectors.npy")

    if len(documents) < 200:
        print(
            "\nCAUTION: This is a small corpus. Treat the run as a mechanics/demo "
            "experiment, not evidence of large-scale speed superiority."
        )
    if len(documents) < 1000:
        print(
            "For more stable latency comparisons, use a larger corpus and repeat "
            "queries; small datasets are often too fast for meaningful timing."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
