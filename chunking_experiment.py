
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
QUESTIONS_FILE = ROOT / "chunking_questions.json"

EMBEDDING_MODEL = "embeddinggemma"
EMBED_BATCH_SIZE = 16
TOP_K_VALUES = [5, 10]
MAX_TOP_K = max(TOP_K_VALUES)

# Six experimental configurations
EXPERIMENTS = [
    {"name": "A", "chunk_size": 500, "overlap": 0},
    {"name": "B", "chunk_size": 500, "overlap": 100},
    {"name": "C", "chunk_size": 800, "overlap": 100},
    {"name": "D", "chunk_size": 800, "overlap": 200},
    {"name": "E", "chunk_size": 1100, "overlap": 180},
    {"name": "F", "chunk_size": 1100, "overlap": 300},
]

COLLECTION_METADATA = {"hnsw:space": "cosine"}


# ==================================================
# 2. SMALL UTILITIES
# ==================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and capitalization for quote matching."""
    return " ".join(text.split()).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def file_metadata(path: Path, page_count: int) -> dict[str, Any]:
    stat = path.stat()

    return {
        "filename": path.name,
        "page_count": page_count,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


# ==================================================
# 3. LOAD PDF PAGES
# ==================================================

def load_pdf_pages() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {DATA_DIR}"
        )

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}"
        )

    pages = []
    documents = []

    for pdf_path in pdf_files:

        reader = PdfReader(str(pdf_path))

        documents.append(
            file_metadata(pdf_path, len(reader.pages))
        )

        # Physical PDF page numbers start at 1.
        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                })

    return pages, documents


# ==================================================
# 4. LOAD AND VALIDATE GROUND-TRUTH EVIDENCE
# ==================================================

def load_and_resolve_questions(
    pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Question file not found: {QUESTIONS_FILE}"
        )

    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "chunking_questions.json must contain a non-empty JSON list."
        )

    seen_ids = set()
    resolved_questions = []

    for question in questions:

        question_id = str(question.get("id", "")).strip()
        question_text = str(question.get("question", "")).strip()

        evidence_items = question.get("relevant_evidence", [])

        if not question_id or question_id in seen_ids:
            raise ValueError(
                f"Question IDs must be present and unique: {question_id!r}"
            )

        seen_ids.add(question_id)

        if not question_text:
            raise ValueError(
                f"Question {question_id} has an empty question."
            )

        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError(
                f"Question {question_id} needs relevant_evidence entries."
            )

        resolved_evidence = []

        for evidence_index, item in enumerate(evidence_items, start=1):

            if not isinstance(item, dict):
                raise ValueError(
                    f"{question_id}: evidence item must be an object."
                )

            try:
                target_page = int(item["page"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"{question_id}: evidence item needs an integer page."
                )

            quote = str(item.get("quote", "")).strip()
            normalized_quote = normalize_text(quote)

            if not normalized_quote:
                raise ValueError(
                    f"{question_id}: evidence quote cannot be empty."
                )

            requested_source = item.get("source")

            candidates = [
                page for page in pages
                if page["page"] == target_page
                and (
                    requested_source is None
                    or page["source"] == requested_source
                )
                and normalized_quote in normalize_text(page["text"])
            ]

            if not candidates:
                raise ValueError(
                    f"Ground-truth validation failed for {question_id}, "
                    f"page {target_page}.\n"
                    f"Quote not found in PDF:\n{quote}"
                )

            if len(candidates) > 1:
                filenames = sorted({
                    candidate["source"]
                    for candidate in candidates
                })

                raise ValueError(
                    f"Quote occurs in multiple PDFs: {filenames}. "
                    f"Add a source field to the question JSON."
                )

            resolved_evidence.append({
                "source": candidates[0]["source"],
                "page": target_page,
                "quote": quote,
                "normalized_quote": normalized_quote,
                "evidence_id": f"{question_id}_E{evidence_index}",
            })

        resolved_questions.append({
            "id": question_id,
            "question": question_text,
            "relevant_evidence": resolved_evidence,
        })

    return resolved_questions


# ==================================================
# 5. RECURSIVE CHUNKING
# ==================================================

def recursive_chunk(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for page in pages:

        documents = splitter.create_documents(
            [page["text"]],
            metadatas=[{
                "source": page["source"],
                "page": page["page"],
            }],
        )

        for index, document in enumerate(documents):

            start_index = document.metadata.get("start_index", -1)

            chunks.append({
                "id": f"{page['source']}_{page['page']}_{index}",
                "text": document.page_content,
                "source": page["source"],
                "page": page["page"],
                "start_index": (
                    int(start_index)
                    if start_index is not None
                    else -1
                ),
            })

    return chunks


# ==================================================
# 6. GENERATE EMBEDDINGS ONCE
# ==================================================

def embed_texts(
    client: ollama.Client,
    texts: list[str],
) -> list[list[float]]:

    vectors = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):

        batch = texts[start:start + EMBED_BATCH_SIZE]

        response = client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        vectors.extend(response.embeddings)

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, "
            f"received {len(vectors)}."
        )

    if vectors and not vectors[0]:
        raise RuntimeError(
            "Ollama returned an empty embedding vector."
        )

    return vectors


def generate_embeddings_once(
    client: ollama.Client,
    texts: list[str],
) -> tuple[list[list[float]], dict[str, Any]]:

    # Exactly ONE embedding pass per configuration.
    start = time.perf_counter()

    vectors = embed_texts(client, texts)

    elapsed = time.perf_counter() - start

    print(
        f"    Embedding completed: {elapsed:.3f}s "
        "(single pass)"
    )

    timing = {
        "elapsed_seconds": elapsed,
        "measurement_count": 1,
        "scope": (
            "One embedding pass. Includes Ollama embedding API calls "
            "and batching. Excludes extraction, chunking, database "
            "writes, query embedding and answer generation."
        ),
    }

    return vectors, timing


# ==================================================
# 7. EVALUATE ONE QUESTION
# ==================================================

def evaluate_one_question(
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:

    gold_pages = {
        (item["source"], item["page"])
        for item in question["relevant_evidence"]
    }

    found_pages = {
        (chunk["source"], chunk["page"])
        for chunk in retrieved_chunks
    }

    retrieved_relevant_pages = gold_pages.intersection(found_pages)

    page_recall = (
        len(retrieved_relevant_pages) / len(gold_pages)
        if gold_pages
        else None
    )

    evidence_diagnostics = []

    for evidence in question["relevant_evidence"]:

        matching_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk["source"] == evidence["source"]
            and chunk["page"] == evidence["page"]
            and evidence["normalized_quote"]
            in normalize_text(chunk["text"])
        ]

        evidence_diagnostics.append({
            "evidence_id": evidence["evidence_id"],
            "source": evidence["source"],
            "page": evidence["page"],
            "quote": evidence["quote"],
            "retrieved_as_complete_quote": bool(matching_chunks),
            "matching_chunk_ranks": [
                chunk["rank"] for chunk in matching_chunks
            ],
        })

    evidence_hits = sum(
        1
        for item in evidence_diagnostics
        if item["retrieved_as_complete_quote"]
    )

    evidence_recall = (
        evidence_hits / len(evidence_diagnostics)
        if evidence_diagnostics
        else None
    )

    return {
        "question_id": question["id"],
        "question": question["question"],

        "gold_pages": [
            {"source": source, "page": page}
            for source, page in sorted(gold_pages)
        ],

        "retrieved_pages": [
            {"source": source, "page": page}
            for source, page in sorted(found_pages)
        ],

        "retrieved_relevant_pages": [
            {"source": source, "page": page}
            for source, page in sorted(retrieved_relevant_pages)
        ],

        "page_recall": page_recall,
        "gold_evidence_count": len(evidence_diagnostics),
        "retrieved_complete_evidence_count": evidence_hits,
        "evidence_recall": evidence_recall,
        "evidence_details": evidence_diagnostics,
        "retrieved_chunks": retrieved_chunks,
    }


# ==================================================
# 8. RETRIEVAL EVALUATION
# ==================================================

def evaluate_retrieval(
    collection: Any,
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    summaries = {}
    all_diagnostics = []

    for question, query_vector in zip(questions, query_vectors):

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(MAX_TOP_K, collection.count()),
            include=["metadatas", "documents", "distances"],
        )

        ids = result["ids"][0]
        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]

        ranked_chunks = []

        for rank, (chunk_id, metadata, document, distance) in enumerate(
            zip(ids, metadatas, documents, distances),
            start=1,
        ):

            ranked_chunks.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source": metadata["source"],
                "page": int(metadata["page"]),
                "start_index": int(metadata.get("start_index", -1)),
                "cosine_distance": float(distance),
                "text": document,
            })

        for k in TOP_K_VALUES:

            diagnostic = evaluate_one_question(
                question,
                ranked_chunks[:k],
            )

            diagnostic["k"] = k
            all_diagnostics.append(diagnostic)

    for k in TOP_K_VALUES:

        subset = [
            item for item in all_diagnostics
            if item["k"] == k
        ]

        page_values = [
            item["page_recall"]
            for item in subset
            if item["page_recall"] is not None
        ]

        evidence_values = [
            item["evidence_recall"]
            for item in subset
            if item["evidence_recall"] is not None
        ]

        total_gold_evidence = sum(
            item["gold_evidence_count"]
            for item in subset
        )

        total_retrieved_evidence = sum(
            item["retrieved_complete_evidence_count"]
            for item in subset
        )

        summaries[f"at_{k}"] = {
            "page_recall_macro_mean": (
                statistics.mean(page_values)
                if page_values else None
            ),

            "evidence_recall_macro_mean": (
                statistics.mean(evidence_values)
                if evidence_values else None
            ),

            "evidence_recall_micro": (
                total_retrieved_evidence / total_gold_evidence
                if total_gold_evidence else None
            ),

            "evaluated_questions": len(subset),
            "gold_evidence_items": total_gold_evidence,
            "retrieved_complete_evidence_items": (
                total_retrieved_evidence
            ),
        }

    return summaries, all_diagnostics


# ==================================================
# 9. RUN ONE EXPERIMENT
# ==================================================

def run_experiment(
    config: dict[str, Any],
    pages: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]],
    chroma_client: Any,
    ollama_client: ollama.Client,
) -> dict[str, Any]:

    name = config["name"]
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]

    print("\n" + "=" * 58)
    print(f"EXPERIMENT {name}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Nominal stride: {chunk_size - overlap}")
    print("=" * 58)

    # Step 1: Generate chunks.
    chunks = recursive_chunk(
        pages,
        chunk_size,
        overlap,
    )

    if not chunks:
        raise RuntimeError(
            f"Experiment {name} generated no chunks."
        )

    texts = [chunk["text"] for chunk in chunks]

    average_length = statistics.mean(
        len(chunk["text"])
        for chunk in chunks
    )

    print(f"Chunks generated: {len(chunks)}")
    print(f"Average chunk length: {average_length:.2f} characters")

    # Step 2: Generate embeddings ONCE.
    vectors, timing = generate_embeddings_once(
        ollama_client,
        texts,
    )

    # Step 3: Create an isolated ChromaDB collection.
    collection_name = f"chunking_exp_{name.lower()}"

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata=COLLECTION_METADATA,
    )

    try:

        # Step 4: Store the single set of vectors.
        collection.add(
            ids=[
                chunk["id"]
                for chunk in chunks
            ],

            documents=texts,

            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "start_index": chunk["start_index"],
                }
                for chunk in chunks
            ],

            embeddings=vectors,
        )

        # Step 5: Evaluate retrieval.
        recall_summary, diagnostics = evaluate_retrieval(
            collection,
            questions,
            query_vectors,
        )

        for k in TOP_K_VALUES:

            metric = recall_summary[f"at_{k}"]

            print(
                f"Page Recall@{k}: "
                f"{metric['page_recall_macro_mean']:.4f} | "
                f"Evidence Recall@{k}: "
                f"{metric['evidence_recall_macro_mean']:.4f}"
            )

    finally:
        # Remove temporary experiment collection.
        chroma_client.delete_collection(collection_name)

    return {
        "experiment": name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stride_nominal": chunk_size - overlap,
        "overlap_ratio_percent": round(
            overlap / chunk_size * 100,
            2,
        ),
        "chunk_count": len(chunks),
        "average_chunk_length_characters": round(
            average_length,
            2,
        ),
        "embedding_time": timing,
        "retrieval_summary": recall_summary,
        "per_question_diagnostics": diagnostics,
    }


# ==================================================
# 10. SAVE RESULTS
# ==================================================

def save_results(
    all_results: list[dict[str, Any]],
    document_metadata: list[dict[str, Any]],
    question_count: int,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment_metadata": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_batch_size": EMBED_BATCH_SIZE,
            "embedding_timing_measurements_per_configuration": 1,
            "top_k_values": TOP_K_VALUES,
            "splitter": "LangChain RecursiveCharacterTextSplitter",
            "length_unit": "characters",
            "separators": ["\\n\\n", "\\n", " ", ""],
            "distance_metric": "cosine",
            "question_count": question_count,

            "documents": document_metadata,

            "metrics": {
                "page_recall": (
                    "Fraction of unique gold source-page pairs "
                    "represented by at least one retrieved chunk."
                ),

                "evidence_recall": (
                    "Fraction of manually specified gold quote snippets "
                    "found completely inside at least one retrieved chunk."
                ),

                "macro_mean": (
                    "Mean of the per-question recall values."
                ),

                "micro": (
                    "Total complete gold evidence snippets retrieved "
                    "divided by total gold evidence snippets."
                ),
            },

            "limitations": [
                (
                    "The evidence metric requires the complete "
                    "normalized gold quote to occur inside one retrieved "
                    "chunk; a quote split across chunks is not counted "
                    "as a complete hit."
                ),
                (
                    "Page recall is a coarse measure: a chunk from a "
                    "relevant page may not contain the needed evidence."
                ),
                (
                    "The question set is a manually constructed pilot "
                    "set, not a statistically representative benchmark."
                ),
                (
                    "The single embedding-time measurement covers Ollama "
                    "document embedding calls only, not extraction, "
                    "chunking, ChromaDB writes, query embedding, or "
                    "answer generation."
                ),
            ],
        },

        "experiments": all_results,
    }

    # Save JSON.
    json_path = RESULTS_DIR / "chunking_results.json"

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Save experiment summary CSV.
    summary_path = RESULTS_DIR / "chunking_results.csv"

    summary_fields = [
        "experiment",
        "chunk_size",
        "overlap",
        "stride_nominal",
        "overlap_ratio_percent",
        "chunk_count",
        "average_chunk_length_characters",
        "embedding_time_seconds",
        "page_recall_at_5",
        "evidence_recall_at_5",
        "page_recall_at_10",
        "evidence_recall_at_10",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        for result in all_results:

            row = {
                "experiment": result["experiment"],
                "chunk_size": result["chunk_size"],
                "overlap": result["overlap"],
                "stride_nominal": result["stride_nominal"],
                "overlap_ratio_percent": result["overlap_ratio_percent"],
                "chunk_count": result["chunk_count"],
                "average_chunk_length_characters": (
                    result["average_chunk_length_characters"]
                ),
                "embedding_time_seconds": round(
                    result["embedding_time"]["elapsed_seconds"],
                    4,
                ),
            }

            for k in TOP_K_VALUES:

                metrics = result["retrieval_summary"][f"at_{k}"]

                row[f"page_recall_at_{k}"] = (
                    metrics["page_recall_macro_mean"]
                )

                row[f"evidence_recall_at_{k}"] = (
                    metrics["evidence_recall_macro_mean"]
                )

            writer.writerow(row)

    # Save per-question diagnostic CSV.
    diagnostics_path = (
        RESULTS_DIR / "chunking_question_diagnostics.csv"
    )

    diagnostic_fields = [
        "experiment",
        "question_id",
        "question",
        "k",
        "gold_pages",
        "retrieved_pages",
        "retrieved_relevant_pages",
        "page_recall",
        "gold_evidence_count",
        "retrieved_complete_evidence_count",
        "evidence_recall",
        "evidence_details",
        "retrieved_chunks",
    ]

    with diagnostics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=diagnostic_fields,
        )

        writer.writeheader()

        for result in all_results:

            for diagnostic in result["per_question_diagnostics"]:

                writer.writerow({
                    "experiment": result["experiment"],
                    "question_id": diagnostic["question_id"],
                    "question": diagnostic["question"],
                    "k": diagnostic["k"],

                    "gold_pages": json.dumps(
                        diagnostic["gold_pages"],
                        ensure_ascii=False,
                    ),

                    "retrieved_pages": json.dumps(
                        diagnostic["retrieved_pages"],
                        ensure_ascii=False,
                    ),

                    "retrieved_relevant_pages": json.dumps(
                        diagnostic["retrieved_relevant_pages"],
                        ensure_ascii=False,
                    ),

                    "page_recall": diagnostic["page_recall"],
                    "gold_evidence_count": (
                        diagnostic["gold_evidence_count"]
                    ),

                    "retrieved_complete_evidence_count": (
                        diagnostic["retrieved_complete_evidence_count"]
                    ),

                    "evidence_recall": diagnostic["evidence_recall"],

                    "evidence_details": json.dumps(
                        diagnostic["evidence_details"],
                        ensure_ascii=False,
                    ),

                    "retrieved_chunks": json.dumps(
                        diagnostic["retrieved_chunks"],
                        ensure_ascii=False,
                    ),
                })

    print("\nSaved results:")
    print(f"  JSON: {json_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Diagnostics CSV: {diagnostics_path}")


# ==================================================
# 11. MAIN
# ==================================================

def main() -> None:

    print("Loading PDFs...")

    pages, document_metadata = load_pdf_pages()

    print(f"PDF files: {len(document_metadata)}")
    print(f"Text-bearing pages loaded: {len(pages)}")

    for document in document_metadata:

        print(
            f"  {document['filename']}: "
            f"{document['page_count']} physical pages, "
            f"SHA-256 {document['sha256'][:12]}..."
        )

    # Load questions and verify evidence against the PDF.
    questions = load_and_resolve_questions(pages)

    total_evidence = sum(
        len(question["relevant_evidence"])
        for question in questions
    )

    print(
        f"\nQuestions loaded and ground truth verified: "
        f"{len(questions)}"
    )

    print(
        f"Gold evidence snippets verified against PDF text: "
        f"{total_evidence}"
    )

    ollama_client = ollama.Client()
    chroma_client = chromadb.EphemeralClient()

    # Query embeddings are generated ONCE.
    query_texts = [
        question["question"]
        for question in questions
    ]

    print(
        f"\nEmbedding {len(query_texts)} questions once "
        f"using {EMBEDDING_MODEL}..."
    )

    query_vectors = embed_texts(
        ollama_client,
        query_texts,
    )

    # Run all six experiments.
    all_results = []

    for config in EXPERIMENTS:

        result = run_experiment(
            config=config,
            pages=pages,
            questions=questions,
            query_vectors=query_vectors,
            chroma_client=chroma_client,
            ollama_client=ollama_client,
        )

        all_results.append(result)

    # Save all results.
    save_results(
        all_results,
        document_metadata,
        len(questions),
    )

    print("\nALL EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    main()
