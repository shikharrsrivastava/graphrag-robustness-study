from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import chromadb
import ollama
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "gemma3:4b"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 16
TOP_K = 2


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks, preferring paragraph breaks."""
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_pdfs() -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}. Add at least one PDF to the data/ folder."
        )

    for pdf_path in pdf_files:
        file_hash = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:10]
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(chunk_text(page_text), start=1):
                texts.append(chunk)
                ids.append(f"{file_hash}-p{page_number}-c{chunk_number}")
                metadatas.append({"source": pdf_path.name, "page": page_number})

    if not texts:
        raise ValueError(
            "PDFs were found, but no extractable text was found. If this is a scanned PDF, it may need OCR."
        )
    return texts, ids, metadatas


def batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def embed_texts(client: ollama.Client, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batches(texts, EMBED_BATCH_SIZE):
        response = client.embed(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(response.embeddings)
    return vectors


def ingest_documents(ollama_client: ollama.Client, collection) -> None:
    texts, ids, metadatas = read_pdfs()
    print(f"Found {len(texts)} text chunks. Creating embeddings with {EMBEDDING_MODEL}...")
    embeddings = embed_texts(ollama_client, texts)
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(texts)} chunks from PDF(s) in: {DATA_DIR}")


def answer_question(ollama_client: ollama.Client, collection, question: str) -> None:
    if collection.count() == 0:
        print("The vector database is empty. Choose I to ingest your PDF(s) first.")
        return

    query_response = ollama_client.embed(model=EMBEDDING_MODEL, input=question)
    query_embedding = query_response.embeddings[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not documents:
        print("No relevant chunks were retrieved.")
        return

    context_parts: list[str] = []
    print("\n--- Retrieved evidence ---")
    for index, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        print(f"[{index}] {source}, page {page} (distance {distance:.3f})")
        print(document[:350].replace("\n", " ") + ("..." if len(document) > 350 else ""))
        print()
        context_parts.append(f"Source: {source}, page {page}\n{document}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        "Answer the user's question using only the evidence in the context below. "
        "If the context does not contain enough information, say that you cannot find the answer in the provided document(s). "
        "Do not make up facts. Briefly mention the source filename and page when useful.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = ollama_client.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print("--- Answer ---")
    print(response.message.content)
    print()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print("Local PDF RAG starter")
    print(f"PDF folder: {DATA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Answer model: {GENERATION_MODEL}")

    ollama_client = ollama.Client()
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    while True:
        print("\nChoose an action: [I] Ingest/re-index PDFs  [Q] Ask a question  [X] Exit")
        choice = input("> ").strip().lower()
        if choice == "i":
            try:
                ingest_documents(ollama_client, collection)
            except Exception as exc:
                print(f"\nCould not ingest PDFs: {exc}")
                print("Check that Ollama is running, both models are downloaded, and your PDF is text-readable.")
        elif choice == "q":
            question = input("Your question: ").strip()
            if question:
                try:
                    answer_question(ollama_client, collection, question)
                except Exception as exc:
                    print(f"\nCould not answer: {exc}")
                    print("Check that Ollama is running and the models are downloaded.")
        elif choice == "x":
            print("Goodbye!")
            break
        else:
            print("Please enter I, Q, or X.")


if __name__ == "__main__":
    main()
