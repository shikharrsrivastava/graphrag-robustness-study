"""Graph validation and manifest generation verifying ontology, provenance, and structure."""

from __future__ import annotations
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import igraph
from src.core.models import CanonicalEntity, Relationship
from src.core.ontology import DomainOntology

logger = logging.getLogger(__name__)


def compute_graph_fingerprint(graph: igraph.Graph) -> str:
    """Compute a deterministic SHA-256 fingerprint over sorted vertices and edges."""
    sorted_vertices = sorted(
        (v["entity_id"], v.attributes().get("label", ""), v.attributes().get("entity_type", ""))
        for v in graph.vs
    )
    sorted_edges = sorted(
        (
            e["source_entity_id"],
            e["relation_type"],
            e["target_entity_id"],
            tuple(sorted(e.attributes().get("source_chunk_ids", []))),
        )
        for e in graph.es
    )
    canonical_repr = json.dumps(
        {"vertices": sorted_vertices, "edges": sorted_edges}, sort_keys=True
    )
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


class GraphValidator:
    """Validates structural integrity, ontology compliance, and provenance references of the graph."""

    @classmethod
    def validate(
        cls,
        graph: igraph.Graph,
        entities: List[CanonicalEntity],
        relationships: List[Relationship],
        valid_chunk_ids: Optional[Set[str]] = None,
        ontology: Optional[DomainOntology] = None,
        manifest_path: Optional[Path] = None,
        kb_id: Optional[str] = None,
        prompt_versions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Validate graph invariants and write manifest.json."""
        invalid_records: List[Dict[str, Any]] = []
        entity_ids = {e.entity_id for e in entities}
        vertex_names = set(graph.vs["name"]) if graph.vcount() > 0 else set()

        # 1. Verify all edge endpoints exist in entities and vertices
        for e in graph.es:
            src = e["source_entity_id"]
            tgt = e["target_entity_id"]
            if src not in entity_ids or src not in vertex_names:
                invalid_records.append(
                    {"type": "missing_source_endpoint", "edge": (src, e["relation_type"], tgt)}
                )
            if tgt not in entity_ids or tgt not in vertex_names:
                invalid_records.append(
                    {"type": "missing_target_endpoint", "edge": (src, e["relation_type"], tgt)}
                )

        # 2. Verify all relation_type values comply with ontology
        rel_type_counts: Counter[str] = Counter()
        for e in graph.es:
            r_type = e["relation_type"]
            rel_type_counts[r_type] += 1
            if ontology is not None and not ontology.is_valid_relation_type(r_type):
                invalid_records.append(
                    {"type": "invalid_relation_type", "relation_type": r_type, "edge": (e["source_entity_id"], r_type, e["target_entity_id"])}
                )

        # 3. Verify referenced source_chunk_ids exist in valid_chunk_ids
        if valid_chunk_ids:
            for v in graph.vs:
                for cid in v.attributes().get("source_chunk_ids", []):
                    if cid not in valid_chunk_ids:
                        invalid_records.append(
                            {"type": "unknown_vertex_chunk_id", "entity_id": v["entity_id"], "chunk_id": cid}
                        )
            for e in graph.es:
                for cid in e.attributes().get("source_chunk_ids", []):
                    if cid not in valid_chunk_ids:
                        invalid_records.append(
                            {"type": "unknown_edge_chunk_id", "relationship_id": e.attributes().get("relationship_id"), "chunk_id": cid}
                        )

        # Structural metrics
        node_count = graph.vcount()
        edge_count = graph.ecount()
        connected_components_count = len(graph.connected_components(mode="weak")) if node_count > 0 else 0
        isolated_nodes = [v["entity_id"] for v in graph.vs if v.degree() == 0]
        fingerprint = compute_graph_fingerprint(graph)

        is_valid = len(invalid_records) == 0

        target_kb_id = kb_id
        if not target_kb_id and entities:
            target_kb_id = entities[0].kb_id
        elif not target_kb_id and graph.vcount() > 0:
            target_kb_id = graph.vs[0].attributes().get("kb_id")

        report: Dict[str, Any] = {
            "kb_id": target_kb_id or "unknown",
            "is_valid": is_valid,
            "prompt_versions": prompt_versions or {},
            "node_count": node_count,
            "edge_count": edge_count,
            "connected_components_count": connected_components_count,
            "isolated_nodes": isolated_nodes,
            "relation_type_distribution": dict(rel_type_counts),
            "invalid_records": invalid_records,
            "graph_fingerprint": fingerprint,
        }

        # Save manifest.json if path is provided
        if manifest_path is not None:
            p = Path(manifest_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report
