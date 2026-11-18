"""Directed igraph builder with vertex/edge attributes and store reconstruction."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import igraph
from src.core.models import CanonicalEntity, Relationship, ValidationStatus
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class IGraphBuilder:
    """Constructs directed igraph instances from canonical entities and validated relationships."""

    @classmethod
    def build_graph(
        cls, entities: List[CanonicalEntity], relationships: List[Relationship]
    ) -> igraph.Graph:
        """Construct directed igraph.Graph with sorted vertices and aggregated multigraph edges."""
        g = igraph.Graph(directed=True)

        # 1. Add vertices sorted by entity_id
        sorted_entities = sorted(entities, key=lambda e: e.entity_id)
        for e in sorted_entities:
            g.add_vertex(
                name=e.entity_id,
                entity_id=e.entity_id,
                label=e.name,
                entity_type=e.entity_type,
                description=e.description,
                aliases=list(e.aliases),
                source_chunk_ids=list(e.source_chunk_ids),
                source_doc_ids=list(e.source_doc_ids),
                kb_id=e.kb_id,
            )

        # Build vertex lookup set
        existing_nodes = set(g.vs["name"]) if g.vcount() > 0 else set()

        # 2. Filter & aggregate edges for VALIDATED relationships
        aggregated: Dict[Tuple[str, str, str, Optional[str]], Dict[str, Any]] = {}

        for r in relationships:
            if r.validation_status != ValidationStatus.VALIDATED:
                continue

            src = r.source_entity_id
            tgt = r.target_entity_id

            if src not in existing_nodes or tgt not in existing_nodes:
                logger.warning(
                    "Skipping relationship %s: endpoints %s -> %s not in graph vertices.",
                    r.relationship_id,
                    src,
                    tgt,
                )
                continue

            if src == tgt:
                logger.warning("Skipping self-loop relationship %s (%s -> %s).", r.relationship_id, src, tgt)
                continue

            key = (src, r.relation_type, tgt, r.scope_id)
            if key in aggregated:
                item = aggregated[key]
                item["source_chunk_ids"] = list(
                    dict.fromkeys(item["source_chunk_ids"] + r.source_chunk_ids)
                )
                item["evidence_quotes"] = list(
                    dict.fromkeys(item["evidence_quotes"] + r.evidence_quotes)
                )
                item["source_doc_ids"] = list(dict.fromkeys(item["source_doc_ids"] + r.source_doc_ids))
                item["support_count"] = len(item["source_chunk_ids"])
            else:
                aggregated[key] = {
                    "relationship_id": r.relationship_id,
                    "source_entity_id": src,
                    "target_entity_id": tgt,
                    "relation_type": r.relation_type,
                    "source_chunk_ids": list(dict.fromkeys(r.source_chunk_ids)),
                    "evidence_quotes": list(dict.fromkeys(r.evidence_quotes)),
                    "source_doc_ids": list(dict.fromkeys(r.source_doc_ids)),
                    "support_count": len(set(r.source_chunk_ids)),
                    "scope_id": r.scope_id,
                }

        # 3. Add edges with attributes
        for edge_data in aggregated.values():
            src_idx = g.vs.find(name=edge_data["source_entity_id"]).index
            tgt_idx = g.vs.find(name=edge_data["target_entity_id"]).index

            g.add_edge(
                src_idx,
                tgt_idx,
                relationship_id=edge_data["relationship_id"],
                source_entity_id=edge_data["source_entity_id"],
                target_entity_id=edge_data["target_entity_id"],
                relation_type=edge_data["relation_type"],
                source_chunk_ids=edge_data["source_chunk_ids"],
                source_doc_ids=edge_data["source_doc_ids"],
                evidence_quotes=edge_data["evidence_quotes"],
                support_count=edge_data["support_count"],
                scope_id=edge_data["scope_id"],
            )

        return g

    @classmethod
    def reconstruct_from_store(cls, doc_store: DocumentStore) -> igraph.Graph:
        """Reconstruct the graph directly from DocumentStore JSONL files."""
        entities = doc_store.load_canonical_entities()
        relationships = doc_store.load_relationships()
        return cls.build_graph(entities, relationships)
