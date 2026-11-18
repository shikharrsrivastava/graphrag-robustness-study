"""ChromaDB VectorStore managing entity_descriptions, text_chunks, and community_reports collections."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings
from src.core.models import CanonicalEntity, Chunk, CommunityReport

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages persistent ChromaDB vector collections scoped to a specific Knowledge Base."""

    COLLECTION_ENTITY_DESCRIPTIONS = "entity_descriptions"
    COLLECTION_TEXT_CHUNKS = "text_chunks"
    COLLECTION_COMMUNITY_REPORTS = "community_reports"

    def __init__(
        self,
        kb_id: str,
        kb_vector_store_dir: Optional[Union[str, Path]] = None,
        base_dir: Union[str, Path] = ".",
        storage_root: Union[str, Path] = "knowledge_bases",
    ) -> None:
        if not kb_id or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string.")
        self.kb_id = kb_id.strip()

        if kb_vector_store_dir is not None:
            self.persist_dir = Path(kb_vector_store_dir)
        else:
            self.persist_dir = Path(base_dir) / storage_root / self.kb_id / "vector_store"

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize the 3 mandatory collections with cosine distance
        cosine_metadata = {"hnsw:space": "cosine"}
        self.entity_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_ENTITY_DESCRIPTIONS,
            metadata=cosine_metadata,
        )
        self.chunk_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_TEXT_CHUNKS,
            metadata=cosine_metadata,
        )
        self.community_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_COMMUNITY_REPORTS,
            metadata=cosine_metadata,
        )

    def clear(self) -> None:
        """Remove this KB's records before an explicit full reindex."""
        for collection in (self.entity_collection, self.chunk_collection, self.community_collection):
            ids = collection.get(where={"kb_id": self.kb_id}).get("ids", [])
            if ids:
                collection.delete(ids=ids)

    # --- Entity Descriptions Collection ---

    def upsert_entities(
        self, entities: List[CanonicalEntity], embeddings: List[List[float]]
    ) -> None:
        """Upsert canonical entities with their dense embeddings."""
        if not entities:
            return
        if len(entities) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(entities)} entities vs {len(embeddings)} embeddings"
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for e in entities:
            if e.kb_id != self.kb_id:
                raise ValueError(
                    f"Entity kb_id '{e.kb_id}' does not match store kb_id '{self.kb_id}'"
                )
            ids.append(e.entity_id)
            documents.append(f"{e.name} ({e.entity_type}): {e.description}")
            metadatas.append(
                {
                    "kb_id": self.kb_id,
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "entity_type": e.entity_type.upper(),
                }
            )

        self.entity_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_entities(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query entity_descriptions collection by cosine distance with optional entity_type filtering."""
        if entity_type:
            where_clause = {
                "$and": [
                    {"kb_id": self.kb_id},
                    {"entity_type": entity_type.strip().upper()},
                ]
            }
        else:
            where_clause = {"kb_id": self.kb_id}

        count = self.entity_collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.entity_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )
        return self._format_results(results)

    # --- Text Chunks Collection ---

    def upsert_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        """Upsert text chunks with their dense embeddings."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            if c.kb_id != self.kb_id:
                raise ValueError(
                    f"Chunk kb_id '{c.kb_id}' does not match store kb_id '{self.kb_id}'"
                )
            ids.append(c.chunk_id)
            documents.append(c.text)
            metadatas.append(
                {
                    "kb_id": self.kb_id,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "scope_id": c.scope_id or "",
                }
            )

        self.chunk_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        scope_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query text_chunks collection by cosine distance with optional scope_id filtering."""
        if scope_id:
            where_clause = {
                "$and": [
                    {"kb_id": self.kb_id},
                    {"scope_id": scope_id},
                ]
            }
        else:
            where_clause = {"kb_id": self.kb_id}

        count = self.chunk_collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.chunk_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )
        return self._format_results(results)

    # --- Community Reports Collection ---

    def upsert_community_reports(
        self, reports: List[CommunityReport], embeddings: List[List[float]]
    ) -> None:
        """Upsert community reports with their dense embeddings."""
        if not reports:
            return
        if len(reports) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(reports)} reports vs {len(embeddings)} embeddings"
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for r in reports:
            if r.kb_id != self.kb_id:
                raise ValueError(
                    f"Report kb_id '{r.kb_id}' does not match store kb_id '{self.kb_id}'"
                )
            ids.append(r.report_id)
            documents.append(r.summary_text)
            metadatas.append(
                {
                    "kb_id": self.kb_id,
                    "report_id": r.report_id,
                    "community_id": r.community_id,
                }
            )

        self.community_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_community_reports(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query community_reports collection by cosine distance."""
        where_clause = {"kb_id": self.kb_id}
        count = self.community_collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.community_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )
        return self._format_results(results)

    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize ChromaDB query results to a list of dicts."""
        formatted: List[Dict[str, Any]] = []
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return formatted

        ids = raw_results["ids"][0]
        distances = raw_results.get("distances", [[]])[0] if raw_results.get("distances") else [None] * len(ids)
        metadatas = raw_results.get("metadatas", [[]])[0] if raw_results.get("metadatas") else [{}] * len(ids)
        documents = raw_results.get("documents", [[]])[0] if raw_results.get("documents") else [""] * len(ids)

        for i in range(len(ids)):
            formatted.append(
                {
                    "id": ids[i],
                    "distance": distances[i],
                    "metadata": metadatas[i],
                    "document": documents[i],
                }
            )
        return formatted
