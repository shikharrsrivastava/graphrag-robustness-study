"""Leiden community detection on undirected projection with chunk-weight aggregation."""

from __future__ import annotations
from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import igraph
import leidenalg
from src.core.configuration import LeidenConfig
from src.graph.graph_validator import compute_graph_fingerprint
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class LeidenCommunityDetector:
    """Executes Leiden community detection on an undirected projection of the directed graph."""

    def __init__(
        self,
        config: Optional[LeidenConfig] = None,
        doc_store: Optional[DocumentStore] = None,
    ) -> None:
        self.config = config or LeidenConfig()
        self.doc_store = doc_store

    def detect_communities(
        self,
        graph: igraph.Graph,
        kb_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run Leiden clustering on an undirected projection with distinct supporting chunk weights.

        Leaves the original directed graph unchanged.
        """
        fingerprint = compute_graph_fingerprint(graph)
        target_kb_id = kb_id or (self.doc_store.kb_id if self.doc_store else "default_kb")

        # 1. Build separate undirected projection G_undirected with identical vertices
        g_undirected = igraph.Graph(directed=False)
        for v in graph.vs:
            g_undirected.add_vertex(
                name=v["name"],
                entity_id=v["entity_id"],
                label=v.attributes().get("label", ""),
                entity_type=v.attributes().get("entity_type", ""),
            )

        # 2. Aggregate distinct supporting chunk IDs between unordered vertex pairs
        # pair_key: (min(u, v), max(u, v))
        pair_chunks: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for e in graph.es:
            u, v = e.source, e.target
            if u == v:
                continue
            pair_key = (min(u, v), max(u, v))
            chunk_ids = e.attributes().get("source_chunk_ids", [])
            for cid in chunk_ids:
                pair_chunks[pair_key].add(cid)

        # 3. Add undirected edges with weight = len(distinct supporting source_chunk_ids)
        for (u, v), chunks in pair_chunks.items():
            weight = max(1, len(chunks))
            g_undirected.add_edge(u, v, weight=weight)

        # 4. Run Leiden clustering with RBConfigurationVertexPartition
        weights_param = "weight" if g_undirected.ecount() > 0 else None
        partition = leidenalg.find_partition(
            g_undirected,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights_param,
            resolution_parameter=self.config.resolution_parameter,
            seed=self.config.random_seed,
        )

        try:
            quality = float(partition.quality())
        except Exception:
            quality = 0.0

        partition_id = (
            f"leiden_rb_res_{self.config.resolution_parameter}_seed_{self.config.random_seed}_{fingerprint[:8]}"
        )

        communities: List[Dict[str, Any]] = []
        # Sort clusters by size descending, then first member entity_id ascending
        clusters = list(partition)
        clusters.sort(key=lambda c: (-len(c), sorted([g_undirected.vs[idx]["entity_id"] for idx in c])))

        for comm_id, cluster_indices in enumerate(clusters):
            member_ids = sorted([g_undirected.vs[idx]["entity_id"] for idx in cluster_indices])
            comm_data = {
                "kb_id": target_kb_id,
                "community_id": comm_id,
                "member_entity_ids": member_ids,
                "size": len(member_ids),
                "partition_id": partition_id,
                "resolution": self.config.resolution_parameter,
                "seed": self.config.random_seed,
                "objective_value": quality,
                "graph_fingerprint": fingerprint,
            }
            communities.append(comm_data)

        # 5. Persist to DocumentStore if provided
        if self.doc_store is not None:
            self.doc_store.save_communities(communities)

        return communities
