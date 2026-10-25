from typing import Literal

import chromadb
import ollama
from chromadb.config import Settings

from config import (
    COLLECTION_NAME,
    DB_PATH,
    EMBED_DOC_PREFIX,
    EMBED_FALLBACK,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)

client = chromadb.PersistentClient(
    path=str(DB_PATH),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

_embed_model: str | None = None


def _resolve_embed_model() -> str:
    global _embed_model
    if _embed_model is None:
        try:
            ollama.embed(model=EMBED_MODEL, input=["warm-up"])
            _embed_model = EMBED_MODEL
        except Exception:
            _embed_model = EMBED_FALLBACK
            print(
                f"[store] '{EMBED_MODEL}' does not support embeddings, "
                f"falling back to '{EMBED_FALLBACK}'."
            )
    return _embed_model


def embed(
    texts: list[str], kind: Literal["document", "query"] = "document"
) -> list[list[float]]:
    """Batch-embed texts in a single fast ollama call.

    Uses the EmbeddingGemma task prefix (document or query) when the
    resolved model is EmbeddingGemma, per Google's model card.
    """
    model = _resolve_embed_model()
    prefix = ""
    if "embeddinggemma" in model.lower():
        prefix = EMBED_QUERY_PREFIX if kind == "query" else EMBED_DOC_PREFIX
    payload = [f"{prefix}{text}" for text in texts] if prefix else list(texts)
    response = ollama.embed(model=model, input=payload)
    return response["embeddings"]


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["meta"] for chunk in chunks],
    )


def get_ids(file_name: str) -> list[str]:
    result = collection.get(where={"source": file_name}, include=[])
    return list(result["ids"])


def delete_ids(ids: list[str]) -> None:
    if ids:
        collection.delete(ids=ids)
