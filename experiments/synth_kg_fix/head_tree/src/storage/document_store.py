"""DocumentStore persisting and querying Document, Page, and Chunk records as JSONL."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CommunityReport,
    Document,
    EntityMention,
    Page,
    Relationship,
)

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages JSONL persistence and fast lookups for Document, Page, and Chunk objects.

    Ensures strict KB isolation: knowledge_bases/<kb_id>/data/processed/
    """

    def __init__(
        self,
        kb_id: str,
        storage_root: Union[str, Path] = "knowledge_bases",
        base_dir: Union[str, Path] = ".",
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if not kb_id or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string.")
        self.kb_id = kb_id.strip()

        if processed_dir is not None:
            self.processed_dir = Path(processed_dir)
        else:
            self.processed_dir = Path(base_dir) / storage_root / self.kb_id / "data" / "processed"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir = Path(base_dir) / storage_root / self.kb_id / "graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.documents_path = self.processed_dir / "documents.jsonl"
        self.pages_path = self.processed_dir / "pages.jsonl"
        self.chunks_path = self.processed_dir / "chunks.jsonl"
        self.entity_mentions_path = self.processed_dir / "entity_mentions.jsonl"
        self.triples_path = self.processed_dir / "triples.jsonl"
        self.mention_to_entity_path = self.processed_dir / "mention_to_entity.jsonl"
        self.entities_path = self.graph_dir / "entities.jsonl"
        self.relationships_path = self.graph_dir / "relationships.jsonl"
        self.communities_path = self.graph_dir / "communities.jsonl"
        self.community_reports_path = self.graph_dir / "community_reports.jsonl"

        self._doc_index: Dict[str, Document] = {}
        self._page_list: List[Page] = []
        self._chunk_index: Dict[str, Chunk] = {}
        self._mentions: List[EntityMention] = []
        self._triples: List[Relationship] = []
        self._canonical_entities: List[CanonicalEntity] = []
        self._mention_to_entity: Dict[str, str] = {}
        self._graph_relationships: List[Relationship] = []
        self._communities: List[Dict[str, Any]] = []
        self._community_reports: List[CommunityReport] = []

        self._load_all_indexes()

    def _load_all_indexes(self) -> None:
        """Populate in-memory indexes from existing JSONL files if they exist."""
        if self.documents_path.exists():
            self.load_documents()
        if self.pages_path.exists():
            self.load_pages()
        if self.chunks_path.exists():
            self.load_chunks()
        if self.entity_mentions_path.exists():
            self.load_entity_mentions()
        if self.triples_path.exists():
            self.load_triples()
        if self.mention_to_entity_path.exists():
            self.load_mention_to_entity()
        if self.entities_path.exists():
            self.load_canonical_entities()
        if self.relationships_path.exists():
            self.load_relationships()
        if self.communities_path.exists():
            self.load_communities()
        if self.community_reports_path.exists():
            self.load_community_reports()

    # --- Document Persistence & Retrieval ---

    def save_documents(self, documents: List[Document], append: bool = False) -> None:
        """Persist Document records to documents.jsonl."""
        for doc in documents:
            if doc.kb_id != self.kb_id:
                raise ValueError(
                    f"Document kb_id '{doc.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.documents_path, mode, encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.model_dump_json() + "\n")

        if append:
            for doc in documents:
                self._doc_index[doc.doc_id] = doc
        else:
            self._doc_index = {doc.doc_id: doc for doc in documents}

    def load_documents(self) -> List[Document]:
        """Load Document records from documents.jsonl, enforcing KB isolation."""
        if not self.documents_path.exists():
            return []

        docs: List[Document] = []
        with open(self.documents_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                doc = Document.model_validate_json(line_str)
                if doc.kb_id == self.kb_id:
                    docs.append(doc)

        self._doc_index = {doc.doc_id: doc for doc in docs}
        return docs

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Lookup a Document by doc_id."""
        doc = self._doc_index.get(doc_id)
        if doc is not None and doc.kb_id == self.kb_id:
            return doc
        return None

    def list_all_documents(self) -> List[Document]:
        """List all documents belonging to this KB."""
        return [doc for doc in self._doc_index.values() if doc.kb_id == self.kb_id]

    # --- Page Persistence & Retrieval ---

    def save_pages(self, pages: List[Page], append: bool = False) -> None:
        """Persist Page records to pages.jsonl."""
        for page in pages:
            if page.kb_id != self.kb_id:
                raise ValueError(
                    f"Page kb_id '{page.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.pages_path, mode, encoding="utf-8") as f:
            for page in pages:
                f.write(page.model_dump_json() + "\n")

        if append:
            self._page_list.extend(pages)
        else:
            self._page_list = list(pages)

    def load_pages(self) -> List[Page]:
        """Load Page records from pages.jsonl, enforcing KB isolation."""
        if not self.pages_path.exists():
            return []

        loaded_pages: List[Page] = []
        with open(self.pages_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                page = Page.model_validate_json(line_str)
                if page.kb_id == self.kb_id:
                    loaded_pages.append(page)

        self._page_list = loaded_pages
        return loaded_pages

    def get_pages_for_document(self, doc_id: str) -> List[Page]:
        """Return all pages for a given doc_id sorted ascending by page_number."""
        pages = [
            p for p in self._page_list if p.kb_id == self.kb_id and p.doc_id == doc_id
        ]
        pages.sort(key=lambda p: p.page_number)
        return pages

    # --- Chunk Persistence & Retrieval ---

    def save_chunks(self, chunks: List[Chunk], append: bool = False) -> None:
        """Persist Chunk records to chunks.jsonl and update in-memory lookup index."""
        for chunk in chunks:
            if chunk.kb_id != self.kb_id:
                raise ValueError(
                    f"Chunk kb_id '{chunk.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.chunks_path, mode, encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

        if append:
            for chunk in chunks:
                self._chunk_index[chunk.chunk_id] = chunk
        else:
            self._chunk_index = {chunk.chunk_id: chunk for chunk in chunks}

    def load_chunks(self) -> List[Chunk]:
        """Load Chunk records from chunks.jsonl, enforcing KB isolation."""
        if not self.chunks_path.exists():
            return []

        loaded_chunks: List[Chunk] = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                chunk = Chunk.model_validate_json(line_str)
                if chunk.kb_id == self.kb_id:
                    loaded_chunks.append(chunk)

        self._chunk_index = {chunk.chunk_id: chunk for chunk in loaded_chunks}
        return loaded_chunks

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fast O(1) lookup of a chunk by chunk_id, enforcing KB isolation."""
        if not self._chunk_index and self.chunks_path.exists():
            self.load_chunks()

        chunk = self._chunk_index.get(chunk_id)
        if chunk is not None and chunk.kb_id == self.kb_id:
            return chunk
        return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Lookup multiple chunks by their chunk_ids, preserving request order."""
        if not self._chunk_index and self.chunks_path.exists():
            self.load_chunks()

        matched: List[Chunk] = []
        for cid in chunk_ids:
            chunk = self._chunk_index.get(cid)
            if chunk is not None and chunk.kb_id == self.kb_id:
                matched.append(chunk)
        return matched

    def list_all_chunks(self) -> List[Chunk]:
        """Return all chunks in this store, enforcing KB isolation."""
        if not self._chunk_index and self.chunks_path.exists():
            self.load_chunks()

        return [c for c in self._chunk_index.values() if c.kb_id == self.kb_id]

    def get_chunks_for_doc(
        self, doc_id: str, scope_id: Optional[str] = None
    ) -> List[Chunk]:
        """Return all chunks belonging to a given doc_id sorted ascending by chunk_index,
        enforcing KB isolation and optional scope_id filtering.
        """
        if not self._chunk_index and self.chunks_path.exists():
            self.load_chunks()

        chunks = [
            c
            for c in self._chunk_index.values()
            if c.kb_id == self.kb_id and c.doc_id == doc_id
        ]
        if scope_id is not None:
            chunks = [
                c for c in chunks if c.scope_id is None or c.scope_id == scope_id
            ]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def get_document_title(self, doc_id: str) -> Optional[str]:
        """Get authoritative title for a document from metadata or structured chunk header.

        Strictly enforces:
        1. Explicit document metadata ('title' or 'document_title').
        2. Structured content header in first chunk ('Title: <Title>').
        Never infers title from filesystem paths or raw filenames.
        """
        if not self._doc_index and self.documents_path.exists():
            self.load_documents()

        doc = self.get_document(doc_id)
        if doc is None:
            return None

        # Priority 1: Document metadata
        for key in ("title", "document_title"):
            val = doc.metadata.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Priority 2: Structured header in chunk 0: 'Title: <Title>'
        chunks = self.get_chunks_for_doc(doc_id)
        if chunks:
            first_line = chunks[0].text.split("\n", 1)[0].strip()
            if first_line.startswith("Title:"):
                cand = first_line[len("Title:") :].strip()
                if cand:
                    return cand

        return None

    def load_document_titles(self) -> Dict[str, str]:
        """Return a mapping of doc_id -> authoritative document title for all documents in this store."""
        if not self._doc_index and self.documents_path.exists():
            self.load_documents()

        titles: Dict[str, str] = {}
        for doc_id in sorted(self._doc_index.keys()):
            title = self.get_document_title(doc_id)
            if title:
                titles[doc_id] = title
        return titles

    def save_all(
        self,
        documents: List[Document],
        pages: List[Page],
        chunks: List[Chunk],
        mentions: Optional[List[EntityMention]] = None,
        triples: Optional[List[Relationship]] = None,
        canonical_entities: Optional[List[CanonicalEntity]] = None,
        relationships: Optional[List[Relationship]] = None,
        mention_to_entity: Optional[Dict[str, str]] = None,
        communities: Optional[List[Dict[str, Any]]] = None,
        community_reports: Optional[List[CommunityReport]] = None,
    ) -> None:
        """Atomically persist documents, pages, chunks, and optional extraction/graph records."""
        self.save_documents(documents)
        self.save_pages(pages)
        self.save_chunks(chunks)
        if mentions is not None:
            self.save_entity_mentions(mentions)
        if triples is not None:
            self.save_triples(triples)
        if canonical_entities is not None:
            self.save_canonical_entities(canonical_entities)
        if relationships is not None:
            self.save_relationships(relationships)
        if mention_to_entity is not None:
            self.save_mention_to_entity(mention_to_entity)
        if communities is not None:
            self.save_communities(communities)
        if community_reports is not None:
            self.save_community_reports(community_reports)

    # --- Entity Mention Persistence & Retrieval ---

    def save_entity_mentions(
        self, mentions: List[EntityMention], append: bool = False
    ) -> None:
        """Persist EntityMention records to entity_mentions.jsonl."""
        for m in mentions:
            if m.kb_id != self.kb_id:
                raise ValueError(
                    f"EntityMention kb_id '{m.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.entity_mentions_path, mode, encoding="utf-8") as f:
            for m in mentions:
                f.write(m.model_dump_json() + "\n")

        if append:
            self._mentions.extend(mentions)
        else:
            self._mentions = list(mentions)

    def load_entity_mentions(self) -> List[EntityMention]:
        """Load EntityMention records from entity_mentions.jsonl, enforcing KB isolation."""
        if not self.entity_mentions_path.exists():
            return []

        loaded: List[EntityMention] = []
        with open(self.entity_mentions_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                mention = EntityMention.model_validate_json(line_str)
                if mention.kb_id == self.kb_id:
                    loaded.append(mention)

        self._mentions = loaded
        return loaded

    # --- Triples Persistence & Retrieval ---

    def save_triples(
        self, triples: List[Relationship], append: bool = False
    ) -> None:
        """Persist Relationship triple records to triples.jsonl."""
        for t in triples:
            if t.kb_id != self.kb_id:
                raise ValueError(
                    f"Relationship kb_id '{t.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.triples_path, mode, encoding="utf-8") as f:
            for t in triples:
                f.write(t.model_dump_json() + "\n")

        if append:
            self._triples.extend(triples)
        else:
            self._triples = list(triples)

    def load_triples(self) -> List[Relationship]:
        """Load Relationship triple records from triples.jsonl, enforcing KB isolation."""
        if not self.triples_path.exists():
            return []

        loaded: List[Relationship] = []
        with open(self.triples_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                rel = Relationship.model_validate_json(line_str)
                if rel.kb_id == self.kb_id:
                    loaded.append(rel)

        self._triples = loaded
        return loaded

    # --- Mention-to-Entity Mapping Persistence ---

    def save_mention_to_entity(self, mapping: Dict[str, str]) -> None:
        """Persist mention_id -> canonical_entity_id mapping to mention_to_entity.jsonl."""
        with open(self.mention_to_entity_path, "w", encoding="utf-8") as f:
            for mention_id, canonical_id in sorted(mapping.items()):
                f.write(
                    json.dumps(
                        {
                            "kb_id": self.kb_id,
                            "mention_id": mention_id,
                            "canonical_entity_id": canonical_id,
                        }
                    )
                    + "\n"
                )
        self._mention_to_entity = dict(mapping)

    def load_mention_to_entity(self) -> Dict[str, str]:
        """Load mention_id -> canonical_entity_id mapping from mention_to_entity.jsonl."""
        if not self.mention_to_entity_path.exists():
            return {}

        mapping: Dict[str, str] = {}
        with open(self.mention_to_entity_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                if data.get("kb_id", self.kb_id) == self.kb_id:
                    m_id = data.get("mention_id")
                    c_id = data.get("canonical_entity_id") or data.get("entity_id")
                    if m_id and c_id:
                        mapping[m_id] = c_id

        self._mention_to_entity = mapping
        return mapping

    # --- Canonical Entity Persistence (graph/entities.jsonl) ---

    def save_canonical_entities(
        self, entities: List[CanonicalEntity], append: bool = False
    ) -> None:
        """Persist CanonicalEntity records to graph/entities.jsonl."""
        for e in entities:
            if e.kb_id != self.kb_id:
                raise ValueError(
                    f"CanonicalEntity kb_id '{e.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.entities_path, mode, encoding="utf-8") as f:
            for e in entities:
                f.write(e.model_dump_json() + "\n")

        if append:
            self._canonical_entities.extend(entities)
        else:
            self._canonical_entities = list(entities)

    def load_canonical_entities(self) -> List[CanonicalEntity]:
        """Load CanonicalEntity records from graph/entities.jsonl, enforcing KB isolation."""
        if not self.entities_path.exists():
            return []

        loaded: List[CanonicalEntity] = []
        with open(self.entities_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                entity = CanonicalEntity.model_validate_json(line_str)
                if entity.kb_id == self.kb_id:
                    loaded.append(entity)

        self._canonical_entities = loaded
        return loaded

    def get_canonical_entity(self, entity_id: str) -> Optional[CanonicalEntity]:
        """Lookup a CanonicalEntity by its entity_id."""
        if not self._canonical_entities and self.entities_path.exists():
            self.load_canonical_entities()
        for e in self._canonical_entities:
            if e.entity_id == entity_id:
                return e
        return None

    # --- Graph Relationship Persistence (graph/relationships.jsonl) ---

    def save_relationships(
        self, relationships: List[Relationship], append: bool = False
    ) -> None:
        """Persist remapped Relationship records to graph/relationships.jsonl."""
        for r in relationships:
            if r.kb_id != self.kb_id:
                raise ValueError(
                    f"Relationship kb_id '{r.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.relationships_path, mode, encoding="utf-8") as f:
            for r in relationships:
                f.write(r.model_dump_json() + "\n")

        if append:
            self._graph_relationships.extend(relationships)
        else:
            self._graph_relationships = list(relationships)

    def load_relationships(self) -> List[Relationship]:
        """Load remapped Relationship records from graph/relationships.jsonl, enforcing KB isolation."""
        if not self.relationships_path.exists():
            return []

        loaded: List[Relationship] = []
        with open(self.relationships_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                rel = Relationship.model_validate_json(line_str)
                if rel.kb_id == self.kb_id:
                    loaded.append(rel)

        self._graph_relationships = loaded
        return loaded

    # --- Communities Persistence (graph/communities.jsonl) ---

    def save_communities(self, communities: List[Dict[str, Any]]) -> None:
        """Persist community assignment records to graph/communities.jsonl."""
        with open(self.communities_path, "w", encoding="utf-8") as f:
            for c in communities:
                f.write(json.dumps(c) + "\n")
        self._communities = list(communities)

    def load_communities(self) -> List[Dict[str, Any]]:
        """Load community assignment records from graph/communities.jsonl."""
        if not self.communities_path.exists():
            return []

        loaded: List[Dict[str, Any]] = []
        with open(self.communities_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                if data.get("kb_id", self.kb_id) == self.kb_id:
                    loaded.append(data)

        self._communities = loaded
        return loaded

    # --- Community Reports Persistence (graph/community_reports.jsonl) ---

    def save_community_reports(
        self, reports: List[CommunityReport], append: bool = False
    ) -> None:
        """Persist CommunityReport records to graph/community_reports.jsonl."""
        for r in reports:
            if r.kb_id != self.kb_id:
                raise ValueError(
                    f"CommunityReport kb_id '{r.kb_id}' does not match store kb_id '{self.kb_id}'"
                )

        mode = "a" if append else "w"
        with open(self.community_reports_path, mode, encoding="utf-8") as f:
            for r in reports:
                f.write(r.model_dump_json() + "\n")

        if append:
            self._community_reports.extend(reports)
        else:
            self._community_reports = list(reports)

    def load_community_reports(self) -> List[CommunityReport]:
        """Load CommunityReport records from graph/community_reports.jsonl, enforcing KB isolation."""
        if not self.community_reports_path.exists():
            return []

        loaded: List[CommunityReport] = []
        with open(self.community_reports_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                report = CommunityReport.model_validate_json(line_str)
                if report.kb_id == self.kb_id:
                    loaded.append(report)

        self._community_reports = loaded
        return loaded
