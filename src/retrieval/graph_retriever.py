"""Graph evidence retriever using SeedEntityRetriever and GraphTraverser."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple
import igraph
from src.core.interfaces import Retriever
from src.core.models import Evidence, ProvenanceRoute, QueryPlan
from src.graph.graph_builder import IGraphBuilder
from src.graph.traversal import GraphTraverser, TraversedPath
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class GraphEvidenceRetriever:
    """Retrieves evidence chunks along graph traversal paths starting from resolved seed entities."""

    def __init__(
        self,
        doc_store: DocumentStore,
        seed_retriever: SeedEntityRetriever,
        graph: Optional[igraph.Graph] = None,
        traverser: Optional[GraphTraverser] = None,
        max_candidates: int = 20,
    ) -> None:
        self.doc_store = doc_store
        self.seed_retriever = seed_retriever
        self.graph = graph or IGraphBuilder.reconstruct_from_store(doc_store)
        self.traverser = traverser or GraphTraverser(max_candidates=max_candidates)
        self.max_candidates = max_candidates
        self.last_diagnostics: Dict[str, int] = {"resolved_seed_count": 0, "valid_path_count": 0, "evidence_count": 0}

    def retrieve(
        self,
        question: str,
        plan: QueryPlan,
        kb_id: str,
        scope_id: Optional[str] = None,
    ) -> List[Evidence]:
        """Retrieve graph evidence for the query plan, satisfying the Retriever protocol."""
        if not plan.seed_entities:
            return []

        # 1. Resolve seed entities
        seed_res = self.seed_retriever.resolve_seeds(plan.seed_entities)
        resolved_eids = seed_res.resolved_entity_ids
        self.last_diagnostics = {"resolved_seed_count": len(resolved_eids), "valid_path_count": 0, "evidence_count": 0}
        if not resolved_eids:
            logger.info("No seed entities could be resolved for query: %s", question)
            return []

        # Map each entity id back to its originating seed spec name
        eid_to_spec_name: Dict[str, Set[str]] = {}
        entities = self.doc_store.load_canonical_entities()
        for spec in plan.seed_entities:
            spec_eids, _ = self.seed_retriever._resolve_single_spec(spec, entities)
            for eid in spec_eids:
                eid_to_spec_name.setdefault(eid, set()).add(spec.name)

        # 2. Traverse graph
        traversed_paths = self.traverser.traverse(
            graph=self.graph,
            seed_entity_ids=resolved_eids,
            relationship_path=plan.relationship_path,
            scope_id=scope_id,
        )
        self.last_diagnostics["valid_path_count"] = len(traversed_paths)

        # 3. Aggregate supporting chunks
        chunk_path_map: Dict[str, List[str]] = {}
        chunk_edge_map: Dict[str, List[str]] = {}
        chunk_min_hop: Dict[str, int] = {}
        chunk_best_alignment: Dict[str, float] = {}
        chunk_support_count: Dict[str, int] = {}
        chunk_specs_map: Dict[str, Set[str]] = {}

        for path in traversed_paths:
            path_str = path.to_path_string()
            start_eid = path.node_ids[0] if path.node_ids else ""
            path_specs = eid_to_spec_name.get(start_eid, set())

            for cid in path.chunk_ids:
                chunk_path_map.setdefault(cid, []).append(path_str)
                chunk_edge_map.setdefault(cid, []).extend(path.edge_ids)
                chunk_support_count[cid] = chunk_support_count.get(cid, 0) + 1
                if path_specs:
                    chunk_specs_map.setdefault(cid, set()).update(path_specs)

                if cid not in chunk_min_hop or path.hop_count < chunk_min_hop[cid]:
                    chunk_min_hop[cid] = path.hop_count
                if (
                    cid not in chunk_best_alignment
                    or path.alignment_score > chunk_best_alignment[cid]
                ):
                    chunk_best_alignment[cid] = path.alignment_score

        # Also incorporate direct chunks from seed entity vertices (even if isolated or with 0 incident edges)
        for eid in resolved_eids:
            v_label = eid
            direct_chunks: List[str] = []
            if eid in self.graph.vs["name"]:
                v = self.graph.vs.find(name=eid)
                v_label = v.attributes().get("label") or eid
                direct_chunks = list(v.attributes().get("source_chunk_ids") or [])
            if not direct_chunks and hasattr(self.doc_store, "get_canonical_entity"):
                ent = self.doc_store.get_canonical_entity(eid)
                if ent:
                    v_label = ent.name
                    direct_chunks = list(ent.source_chunk_ids)

            seed_specs = eid_to_spec_name.get(eid, set())
            for cid in direct_chunks:
                direct_path_str = f"Seed Entity [{v_label} ({eid})]"
                chunk_path_map.setdefault(cid, []).append(direct_path_str)
                chunk_support_count[cid] = chunk_support_count.get(cid, 0) + 1
                chunk_min_hop[cid] = 0
                chunk_best_alignment[cid] = max(chunk_best_alignment.get(cid, 0.0), 1.0)
                if seed_specs:
                    chunk_specs_map.setdefault(cid, set()).update(seed_specs)

        if not chunk_path_map:
            return []

        # 4. Deterministic sorting:
        # Base key: (best_alignment DESC, min_hop ASC, support_count DESC, chunk_id ASC)
        def chunk_sort_key(cid: str):
            return (
                -chunk_best_alignment[cid],
                chunk_min_hop[cid],
                -chunk_support_count[cid],
                cid,
            )

        if len(plan.seed_entities) > 1:
            # Multi-seed query: balance representation across seed entities to prevent starvation
            spec_names = [s.name for s in plan.seed_entities]

            multi_seed_chunks = [
                cid for cid in chunk_path_map.keys()
                if len(chunk_specs_map.get(cid, set())) > 1
            ]
            multi_seed_chunks.sort(key=chunk_sort_key)

            spec_chunk_lists: List[List[str]] = []
            for sname in spec_names:
                cids_for_s = [
                    cid for cid in chunk_path_map.keys()
                    if cid not in multi_seed_chunks and sname in chunk_specs_map.get(cid, set())
                ]
                cids_for_s.sort(key=chunk_sort_key)
                spec_chunk_lists.append(cids_for_s)

            interleaved_single_seed: List[str] = []
            max_len = max((len(l) for l in spec_chunk_lists), default=0)
            for i in range(max_len):
                for l in spec_chunk_lists:
                    if i < len(l) and l[i] not in interleaved_single_seed:
                        interleaved_single_seed.append(l[i])

            handled = set(multi_seed_chunks) | set(interleaved_single_seed)
            remaining = [cid for cid in chunk_path_map.keys() if cid not in handled]
            remaining.sort(key=chunk_sort_key)

            sorted_chunk_ids = multi_seed_chunks + interleaved_single_seed + remaining
        else:
            sorted_chunk_ids = sorted(chunk_path_map.keys(), key=chunk_sort_key)

        # 5. Build Evidence objects
        evidence_list: List[Evidence] = []
        rank = 1
        for cid in sorted_chunk_ids[: self.max_candidates]:
            chunk = self.doc_store.get_chunk(cid)
            if chunk is None:
                continue

            unique_paths = list(dict.fromkeys(chunk_path_map[cid]))
            unique_edges = list(dict.fromkeys(chunk_edge_map.get(cid, [])))

            ev = Evidence(
                kb_id=kb_id,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_numbers=list(chunk.page_numbers),
                text=chunk.text,
                routes=[ProvenanceRoute.GRAPH],
                graph_rank=rank,
                vector_rank=None,
                vector_distance=None,
                rrf_score=0.0,
                supporting_graph_paths=unique_paths,
                supporting_edge_ids=unique_edges,
                community_ids=[],
            )
            evidence_list.append(ev)
            rank += 1

        self.last_diagnostics["evidence_count"] = len(evidence_list)
        return evidence_list
