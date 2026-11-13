"""Directed graph traversal with ontology path alignment, cycle prevention, and chunk collection."""

from __future__ import annotations
from collections import deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import igraph

logger = logging.getLogger(__name__)


class TraversedPath:
    """Represents a validated path traversed through the knowledge graph."""

    def __init__(
        self,
        node_ids: List[str],
        node_labels: List[str],
        edge_ids: List[str],
        relation_types: List[str],
        chunk_ids: List[str],
        hop_count: int,
        alignment_score: float,
    ) -> None:
        self.node_ids = node_ids
        self.node_labels = node_labels
        self.edge_ids = edge_ids
        self.relation_types = relation_types
        self.chunk_ids = list(dict.fromkeys(chunk_ids))
        self.hop_count = hop_count
        self.alignment_score = alignment_score

    def to_path_string(self) -> str:
        """Render a readable representation of the traversal path."""
        parts = []
        for i, node_label in enumerate(self.node_labels):
            parts.append(node_label)
            if i < len(self.relation_types):
                parts.append(f"-[{self.relation_types[i]}]->")
        return " ".join(parts)


class GraphTraverser:
    """Traverses igraph instances starting from seed entity IDs."""

    def __init__(self, max_hops: int = 3, max_candidates: int = 20) -> None:
        self.max_hops = max_hops
        self.max_candidates = max_candidates

    def traverse(
        self,
        graph: igraph.Graph,
        seed_entity_ids: List[str],
        relationship_path: Optional[List[str]] = None,
        scope_id: Optional[str] = None,
    ) -> List[TraversedPath]:
        """Perform directed traversal from seed entities with cycle prevention and path alignment."""
        if not seed_entity_ids or graph.vcount() == 0:
            return []

        rel_path = [r.strip().upper() for r in (relationship_path or [])]
        existing_nodes = set(graph.vs["name"])
        valid_seeds = [s for s in seed_entity_ids if s in existing_nodes]

        if not valid_seeds:
            logger.info("None of the seed entities %s exist in the graph vertices.", seed_entity_ids)
            return []

        traversed_paths: List[TraversedPath] = []
        # Queue item: (curr_idx, [node_ids], [node_labels], [edge_ids], [rel_types], [chunk_ids])
        queue: deque = deque()

        for s_id in valid_seeds:
            v = graph.vs.find(name=s_id)
            v_chunks = list(v.attributes().get("source_chunk_ids") or [])
            v_label = v.attributes().get("label") or s_id
            queue.append((v.index, [s_id], [v_label], [], [], list(v_chunks)))

        visited_path_signatures: Set[Tuple[str, ...]] = set()

        while queue and len(traversed_paths) < self.max_candidates:
            curr_idx, node_ids, node_labels, edge_ids, rel_types, path_chunks = queue.popleft()
            curr_v = graph.vs[curr_idx]
            current_hop = len(edge_ids)

            # If path has at least 1 edge, record as a valid traversed path
            if current_hop > 0:
                sig = tuple(node_ids + rel_types)
                if sig not in visited_path_signatures:
                    visited_path_signatures.add(sig)

                    # Compute alignment score with expected relationship_path
                    score = 1.0
                    if rel_path:
                        matched_steps = sum(
                            1 for actual, expected in zip(rel_types, rel_path) if actual == expected
                        )
                        if rel_types == rel_path[: len(rel_types)]:
                            score = 2.0 + matched_steps * 1.0
                        else:
                            score = 0.5 + matched_steps * 0.5

                    traversed_paths.append(
                        TraversedPath(
                            node_ids=list(node_ids),
                            node_labels=list(node_labels),
                            edge_ids=list(edge_ids),
                            relation_types=list(rel_types),
                            chunk_ids=list(path_chunks),
                            hop_count=current_hop,
                            alignment_score=score,
                        )
                    )

            if current_hop >= self.max_hops:
                continue

            # Outgoing edges exploration
            out_edges = curr_v.out_edges()
            for edge in out_edges:
                edge_attrs = edge.attributes()
                # Scope filter
                edge_scope = edge_attrs.get("scope_id")
                if scope_id and edge_scope and edge_scope != scope_id:
                    continue

                tgt_idx = edge.target
                tgt_v = graph.vs[tgt_idx]
                tgt_id = tgt_v.attributes().get("name") or tgt_v.attributes().get("entity_id")

                # Cycle prevention: Do not revisit vertices on current path
                if tgt_id in node_ids:
                    continue

                tgt_label = tgt_v.attributes().get("label") or tgt_id
                rel_type = edge_attrs.get("relation_type", "")
                edge_id = edge_attrs.get("relationship_id", f"{curr_v['name']}->{tgt_id}")
                edge_chunks = edge_attrs.get("source_chunk_ids") or []
                tgt_chunks = tgt_v.attributes().get("source_chunk_ids") or []

                new_chunks = list(dict.fromkeys(path_chunks + list(edge_chunks) + list(tgt_chunks)))

                queue.append(
                    (
                        tgt_idx,
                        node_ids + [tgt_id],
                        node_labels + [tgt_label],
                        edge_ids + [edge_id],
                        rel_types + [rel_type],
                        new_chunks,
                    )
                )

        # Sort traversed paths by alignment_score DESC, hop_count ASC
        traversed_paths.sort(key=lambda p: (-p.alignment_score, p.hop_count))
        return traversed_paths[: self.max_candidates]
