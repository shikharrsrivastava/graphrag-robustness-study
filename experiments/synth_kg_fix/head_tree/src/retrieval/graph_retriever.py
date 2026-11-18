"""Graph evidence retriever using SeedEntityRetriever, GraphTraverser, and Bipartite Provenance Expansion."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import igraph
from src.core.configuration import RetrievalConfig
from src.core.interfaces import Retriever
from src.core.models import (
    CanonicalEntity,
    CooccurrenceProvenance,
    DocumentTitleProvenance,
    EntityDescriptionProvenance,
    Evidence,
    GraphProvenanceType,
    ProvenanceRoute,
    QueryPlan,
)
from src.graph.graph_builder import IGraphBuilder
from src.graph.traversal import GraphTraverser, TraversedPath
from src.retrieval.document_title_discoverer import DocumentTitleDiscoverer
from src.retrieval.entity_description_discoverer import EntityDescriptionCandidate
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class GraphEvidenceRetriever:
    """Retrieves evidence chunks along graph traversal paths starting from resolved seed entities,
    with optional 1-hop Entity-Chunk-Entity bipartite provenance expansion.
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        seed_retriever: SeedEntityRetriever,
        graph: Optional[igraph.Graph] = None,
        traverser: Optional[GraphTraverser] = None,
        max_candidates: int = 20,
        provenance_expansion_enabled: bool = False,
        max_chunks_per_cooccurring_entity: int = 2,
        max_cooccurring_entities_per_chunk: int = 5,
        max_expanded_candidates_per_query: int = 10,
        direct_seed_outranks_expanded: bool = True,
        document_title_discovery_enabled: bool = False,
        max_title_matches_per_seed: int = 2,
        max_chunks_per_title_match: int = 3,
        max_title_candidates_per_query: int = 6,
        entity_description_discovery_enabled: bool = False,
        entity_description_similarity_threshold: float = 0.65,
        max_description_candidates_per_seed: int = 3,
        max_description_candidates_per_query: int = 6,
        retrieval_config: Optional[RetrievalConfig] = None,
        title_discoverer: Optional[DocumentTitleDiscoverer] = None,
    ) -> None:
        self.doc_store = doc_store
        self.seed_retriever = seed_retriever
        self.graph = graph or IGraphBuilder.reconstruct_from_store(doc_store)
        self.traverser = traverser or GraphTraverser(max_candidates=max_candidates)

        if retrieval_config is not None:
            self.max_candidates = retrieval_config.max_graph_candidates
            self.provenance_expansion_enabled = retrieval_config.provenance_expansion_enabled
            self.max_chunks_per_cooccurring_entity = retrieval_config.max_chunks_per_cooccurring_entity
            self.max_cooccurring_entities_per_chunk = retrieval_config.max_cooccurring_entities_per_chunk
            self.max_expanded_candidates_per_query = retrieval_config.max_expanded_candidates_per_query
            self.direct_seed_outranks_expanded = retrieval_config.direct_seed_outranks_expanded
            self.document_title_discovery_enabled = retrieval_config.document_title_discovery_enabled
            self.max_title_matches_per_seed = retrieval_config.max_title_matches_per_seed
            self.max_chunks_per_title_match = retrieval_config.max_chunks_per_title_match
            self.max_title_candidates_per_query = retrieval_config.max_title_candidates_per_query
            self.entity_description_discovery_enabled = retrieval_config.entity_description_discovery_enabled
            self.entity_description_similarity_threshold = retrieval_config.entity_description_similarity_threshold
            self.max_description_candidates_per_seed = retrieval_config.max_description_candidates_per_seed
            self.max_description_candidates_per_query = retrieval_config.max_description_candidates_per_query
        else:
            self.max_candidates = max_candidates
            self.provenance_expansion_enabled = provenance_expansion_enabled
            self.max_chunks_per_cooccurring_entity = max_chunks_per_cooccurring_entity
            self.max_cooccurring_entities_per_chunk = max_cooccurring_entities_per_chunk
            self.max_expanded_candidates_per_query = max_expanded_candidates_per_query
            self.direct_seed_outranks_expanded = direct_seed_outranks_expanded
            self.document_title_discovery_enabled = document_title_discovery_enabled
            self.max_title_matches_per_seed = max_title_matches_per_seed
            self.max_chunks_per_title_match = max_chunks_per_title_match
            self.max_title_candidates_per_query = max_title_candidates_per_query
            self.entity_description_discovery_enabled = entity_description_discovery_enabled
            self.entity_description_similarity_threshold = entity_description_similarity_threshold
            self.max_description_candidates_per_seed = max_description_candidates_per_seed
            self.max_description_candidates_per_query = max_description_candidates_per_query

        # Synchronize configuration down to seed_retriever if applicable
        if self.entity_description_discovery_enabled and hasattr(self.seed_retriever, "entity_description_discovery_enabled"):
            self.seed_retriever.entity_description_discovery_enabled = True
            self.seed_retriever.entity_description_similarity_threshold = self.entity_description_similarity_threshold
            self.seed_retriever.max_description_candidates_per_seed = self.max_description_candidates_per_seed
            self.seed_retriever.max_description_candidates_per_query = self.max_description_candidates_per_query
            if hasattr(self.seed_retriever, "description_discoverer") and self.seed_retriever.description_discoverer:
                self.seed_retriever.description_discoverer.similarity_threshold = self.entity_description_similarity_threshold
                self.seed_retriever.description_discoverer.max_candidates_per_seed = self.max_description_candidates_per_seed
                self.seed_retriever.description_discoverer.max_candidates_per_query = self.max_description_candidates_per_query

        self.title_discoverer = title_discoverer or DocumentTitleDiscoverer(
            doc_store=self.doc_store,
            max_title_matches_per_seed=self.max_title_matches_per_seed,
            max_chunks_per_title_match=self.max_chunks_per_title_match,
            max_title_candidates_per_query=self.max_title_candidates_per_query,
        )

        # In-memory bipartite index caches
        self._bipartite_indexed: bool = False
        self._entities_by_id: Dict[str, CanonicalEntity] = {}
        self._chunk_to_entity_ids: Dict[str, List[str]] = {}
        self._entity_to_chunk_ids: Dict[str, List[str]] = {}

        self.last_diagnostics: Dict[str, Any] = {
            "resolved_seed_count": 0,
            "valid_path_count": 0,
            "evidence_count": 0,
            "provenance_expansion_enabled": self.provenance_expansion_enabled,
            "cooccurring_entities_count": 0,
            "expanded_chunks_count": 0,
            "candidate_fan_out": 0,
            "skipped_hub_entities": [],
            "fan_out_pruned": False,
            "document_title_discovery_enabled": self.document_title_discovery_enabled,
            "document_title_diagnostics": {
                "enabled": self.document_title_discovery_enabled,
                "matched_documents_count": 0,
                "discovered_chunks_count": 0,
                "ambiguous_titles_count": 0,
                "matches": [],
            },
            "entity_description_discovery_enabled": self.entity_description_discovery_enabled,
            "entity_description_diagnostics": {
                "enabled": self.entity_description_discovery_enabled,
                "matched_entities_count": 0,
                "matches": [],
            },
        }

    def _ensure_bipartite_index(self, force: bool = False) -> None:
        """Lazily construct in-memory bipartite index between entities and chunks."""
        if self._bipartite_indexed and not force:
            return
        entities = self.doc_store.load_canonical_entities()
        self._entities_by_id = {ent.entity_id: ent for ent in entities}
        self._chunk_to_entity_ids = {}
        self._entity_to_chunk_ids = {}
        for ent in entities:
            cids = list(ent.source_chunk_ids)
            self._entity_to_chunk_ids[ent.entity_id] = cids
            for cid in cids:
                self._chunk_to_entity_ids.setdefault(cid, []).append(ent.entity_id)
        self._bipartite_indexed = True

    def retrieve(
        self,
        question: str,
        plan: QueryPlan,
        kb_id: str,
        scope_id: Optional[str] = None,
        seed_resolution_result: Optional[SeedResolutionResult] = None,
    ) -> List[Evidence]:
        """Retrieve graph evidence for the query plan, satisfying the Retriever protocol."""
        self.last_diagnostics = {
            "resolved_seed_count": 0,
            "valid_path_count": 0,
            "evidence_count": 0,
            "provenance_expansion_enabled": self.provenance_expansion_enabled,
            "cooccurring_entities_count": 0,
            "expanded_chunks_count": 0,
            "candidate_fan_out": 0,
            "skipped_hub_entities": [],
            "fan_out_pruned": False,
            "document_title_discovery_enabled": self.document_title_discovery_enabled,
            "document_title_diagnostics": {
                "enabled": self.document_title_discovery_enabled,
                "matched_documents_count": 0,
                "discovered_chunks_count": 0,
                "ambiguous_titles_count": 0,
                "matches": [],
            },
        }

        if not plan.seed_entities:
            return []

        # 1. Resolve seed entities (or reuse passed seed_resolution_result)
        if seed_resolution_result is not None:
            seed_res = seed_resolution_result
        else:
            seed_res = self.seed_retriever.resolve_seeds(plan.seed_entities)
        resolved_eids = seed_res.resolved_entity_ids
        self.last_diagnostics["resolved_seed_count"] = len(resolved_eids)
        self.last_diagnostics["ambiguous_seeds"] = dict(seed_res.ambiguous_seeds)
        self.last_diagnostics["unresolved_seeds"] = list(seed_res.unresolved_seeds)
        self.last_diagnostics["seed_resolution_diagnostics"] = [
            d.model_dump() if hasattr(d, "model_dump") else d
            for d in getattr(seed_res, "resolution_diagnostics", ())
        ]
        if not resolved_eids and not self.document_title_discovery_enabled:
            logger.info("No seed entities could be resolved for query: %s", question)
            return []

        # Tracking structures for evidence aggregation
        chunk_path_map: Dict[str, List[str]] = {}
        chunk_edge_map: Dict[str, List[str]] = {}
        chunk_min_hop: Dict[str, int] = {}
        chunk_best_alignment: Dict[str, float] = {}
        chunk_support_count: Dict[str, int] = {}
        chunk_specs_map: Dict[str, Set[str]] = {}
        chunk_provenance_types: Dict[str, List[GraphProvenanceType]] = {}
        chunk_cooccurrence_details: Dict[str, List[CooccurrenceProvenance]] = {}
        chunk_title_details: Dict[str, List[DocumentTitleProvenance]] = {}
        chunk_desc_details: Dict[str, List[EntityDescriptionProvenance]] = {}

        if resolved_eids:
            # Map each entity id back to its originating seed spec name and detect type conflicts
            eid_to_spec_name: Dict[str, Set[str]] = {}
            eid_to_type_conflict: Dict[str, Tuple[str, str]] = {}
            eid_to_desc_candidate: Dict[str, EntityDescriptionCandidate] = {}

            # Collect description candidates from seed_res if present
            for dc in getattr(seed_res, "description_candidates", ()):
                eid_to_desc_candidate[dc.candidate_id] = dc

            for d in getattr(seed_res, "resolution_diagnostics", ()):
                if hasattr(d, "candidate_id") and d.candidate_id and hasattr(d, "status") and d.status in ("ACCEPTED", "AMBIGUOUS"):
                    eid_to_spec_name.setdefault(d.candidate_id, set()).add(d.seed_name)
                    if not d.type_compatible and d.expected_type and d.candidate_type:
                        eid_to_type_conflict[d.candidate_id] = (d.expected_type, d.candidate_type)
                    if getattr(d, "matching_method", None) == "ENTITY_DESCRIPTION_SEARCH":
                        if d.candidate_id not in eid_to_desc_candidate:
                            ent_desc = ""
                            if hasattr(self.doc_store, "get_canonical_entity"):
                                e_obj = self.doc_store.get_canonical_entity(d.candidate_id)
                                if e_obj:
                                    ent_desc = e_obj.description
                            eid_to_desc_candidate[d.candidate_id] = EntityDescriptionCandidate(
                                seed_name=d.seed_name,
                                expected_type=d.expected_type,
                                candidate_id=d.candidate_id,
                                candidate_name=d.candidate_name or d.candidate_id,
                                candidate_type=d.candidate_type or "UNKNOWN",
                                matched_description=ent_desc,
                                similarity_score=d.similarity_score or 0.0,
                                distance_score=round(1.0 - (d.similarity_score or 0.0), 4) if d.similarity_score else None,
                                matching_method="ENTITY_DESCRIPTION_SEARCH",
                                type_compatible=d.type_compatible,
                                confidence_score=d.confidence_score,
                                ranking=d.ranking,
                                is_ambiguous=d.status == "AMBIGUOUS",
                                candidate_ids=[d.candidate_id],
                                status=d.status,
                                reason=d.reason,
                            )
                    else:
                        # Exact or alias matches strictly supersede description discovery
                        eid_to_desc_candidate.pop(d.candidate_id, None)

            # 2. Traverse graph
            traversed_paths = self.traverser.traverse(
                graph=self.graph,
                seed_entity_ids=resolved_eids,
                relationship_path=plan.relationship_path,
                scope_id=scope_id,
            )
            self.last_diagnostics["valid_path_count"] = len(traversed_paths)

            # 3. Aggregate supporting chunks from graph traversal
            for path in traversed_paths:
                path_str = path.to_path_string()
                labeled_path_str = f"GRAPH_EDGE_PROVENANCE: {path_str}"
                start_eid = path.node_ids[0] if path.node_ids else ""
                path_specs = eid_to_spec_name.get(start_eid, set())

                for cid in path.chunk_ids:
                    chunk_path_map.setdefault(cid, []).append(labeled_path_str)
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
                    chunk_provenance_types.setdefault(cid, []).append(
                        GraphProvenanceType.GRAPH_EDGE_PROVENANCE
                    )

            # Direct chunks from seed entity vertices
            seed_direct_chunks_map: Dict[str, List[str]] = {}
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

                seed_direct_chunks_map[eid] = direct_chunks
                seed_specs = eid_to_spec_name.get(eid, set())
                conflict = eid_to_type_conflict.get(eid)
                conflict_suffix = f" [Type Conflict: expected {conflict[0]}, actual {conflict[1]}]" if conflict else ""
                desc_cand = eid_to_desc_candidate.get(eid)
                is_desc_seed = desc_cand is not None

                for cid in direct_chunks:
                    if is_desc_seed and desc_cand:
                        direct_path_str = (
                            f"ENTITY_DESCRIPTION_SEED_PROVENANCE: Description Seed [{v_label} ({eid})] "
                            f"(sim: {desc_cand.similarity_score:.3f}){conflict_suffix}"
                        )
                        prov_type = GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE
                        alignment_val = 0.90
                        prov_detail = desc_cand.to_provenance()
                        chunk_desc_details.setdefault(cid, []).append(prov_detail)
                    else:
                        direct_path_str = f"GRAPH_SEED_PROVENANCE: Seed Entity [{v_label} ({eid})]{conflict_suffix}"
                        prov_type = GraphProvenanceType.GRAPH_SEED_PROVENANCE
                        alignment_val = 1.00

                    chunk_path_map.setdefault(cid, []).append(direct_path_str)
                    chunk_support_count[cid] = chunk_support_count.get(cid, 0) + 1
                    chunk_min_hop[cid] = 0
                    chunk_best_alignment[cid] = max(chunk_best_alignment.get(cid, 0.0), alignment_val)
                    if seed_specs:
                        chunk_specs_map.setdefault(cid, set()).update(seed_specs)
                    chunk_provenance_types.setdefault(cid, []).append(prov_type)

            desc_matches_diag = []
            desc_matched_eids: Set[str] = set()
            desc_discovered_cids: Set[str] = set()
            ambig_desc_count = 0
            for eid, dc in eid_to_desc_candidate.items():
                desc_matched_eids.add(eid)
                if dc.is_ambiguous:
                    ambig_desc_count += 1
                cids_for_e = seed_direct_chunks_map.get(eid, [])
                desc_discovered_cids.update(cids_for_e)
                desc_matches_diag.append(dc.model_dump())

            self.last_diagnostics["entity_description_diagnostics"] = {
                "enabled": self.entity_description_discovery_enabled,
                "matched_entities_count": len(desc_matched_eids),
                "discovered_chunks_count": len(desc_discovered_cids),
                "ambiguous_descriptions_count": ambig_desc_count,
                "matches": desc_matches_diag,
            }

            # 3b. One-hop Entity-Chunk-Entity bipartite provenance expansion
            if self.provenance_expansion_enabled:
                self._ensure_bipartite_index()
                cooccurring_entities_seen: Set[str] = set()
                expanded_chunks_seen: Set[str] = set()
                skipped_hubs: List[str] = []
                fan_out_pruned = False
                total_expanded_added = 0
                existing_cids = set(chunk_path_map.keys())

                for seed_eid in resolved_eids:
                    seed_name = seed_eid
                    if seed_eid in self._entities_by_id:
                        seed_name = self._entities_by_id[seed_eid].name
                    elif seed_eid in self.graph.vs["name"]:
                        seed_name = self.graph.vs.find(name=seed_eid).attributes().get("label") or seed_eid

                    direct_cids = seed_direct_chunks_map.get(seed_eid, [])
                    seed_specs = eid_to_spec_name.get(seed_eid, set())

                    for direct_cid in direct_cids:
                        # Find entities co-occurring in this direct chunk
                        cooccur_eids = sorted([
                            e for e in self._chunk_to_entity_ids.get(direct_cid, [])
                            if e != seed_eid
                        ])

                        # Cap per direct chunk
                        if len(cooccur_eids) > self.max_cooccurring_entities_per_chunk:
                            cooccur_eids = cooccur_eids[: self.max_cooccurring_entities_per_chunk]

                        for cooccur_eid in cooccur_eids:
                            cooccurring_entities_seen.add(cooccur_eid)
                            cooccur_ent = self._entities_by_id.get(cooccur_eid)
                            cooccur_name = cooccur_ent.name if cooccur_ent else cooccur_eid

                            other_cids = sorted(self._entity_to_chunk_ids.get(cooccur_eid, []))
                            # Hub suppression
                            if len(other_cids) > self.max_chunks_per_cooccurring_entity:
                                if cooccur_eid not in skipped_hubs:
                                    skipped_hubs.append(cooccur_eid)
                                continue

                            # Expanded chunks excluding direct_cid and chunks already in direct/traversal candidate set
                            expanded_cids = [
                                c for c in other_cids
                                if c != direct_cid and c not in existing_cids
                            ]
                            for exp_cid in expanded_cids:
                                if scope_id:
                                    chunk_obj = self.doc_store.get_chunk(exp_cid)
                                    if chunk_obj and chunk_obj.scope_id and chunk_obj.scope_id != scope_id:
                                        continue

                                if total_expanded_added >= self.max_expanded_candidates_per_query:
                                    fan_out_pruned = True
                                    break

                                expanded_chunks_seen.add(exp_cid)
                                total_expanded_added += 1

                                cooccur_path_str = (
                                    f"ENTITY_CHUNK_COOCCURRENCE: Seed [{seed_name} ({seed_eid})] "
                                    f"-> Chunk [{direct_cid}] -> Co-occur [{cooccur_name} ({cooccur_eid})] "
                                    f"-> Chunk [{exp_cid}]"
                                )
                                chunk_path_map.setdefault(exp_cid, []).append(cooccur_path_str)
                                chunk_support_count[exp_cid] = chunk_support_count.get(exp_cid, 0) + 1

                                alignment_val = 0.85 if self.direct_seed_outranks_expanded else 1.0
                                if (
                                    exp_cid not in chunk_best_alignment
                                    or alignment_val > chunk_best_alignment[exp_cid]
                                ):
                                    chunk_best_alignment[exp_cid] = alignment_val

                                if exp_cid not in chunk_min_hop or 1 < chunk_min_hop[exp_cid]:
                                    chunk_min_hop[exp_cid] = 1

                                if seed_specs:
                                    chunk_specs_map.setdefault(exp_cid, set()).update(seed_specs)

                                chunk_provenance_types.setdefault(exp_cid, []).append(
                                    GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE
                                )

                                prov_detail = CooccurrenceProvenance(
                                    seed_entity_id=seed_eid,
                                    direct_chunk_id=direct_cid,
                                    cooccurring_entity_id=cooccur_eid,
                                    expanded_chunk_id=exp_cid,
                                    expansion_depth=1,
                                    skipped_hub_entities=list(skipped_hubs),
                                    fan_out_pruned=fan_out_pruned,
                                )
                                chunk_cooccurrence_details.setdefault(exp_cid, []).append(prov_detail)

                            if fan_out_pruned:
                                break
                        if fan_out_pruned:
                            break
                    if fan_out_pruned:
                        break

                if fan_out_pruned:
                    for details_list in chunk_cooccurrence_details.values():
                        for d in details_list:
                            d.fan_out_pruned = True

                self.last_diagnostics.update({
                    "cooccurring_entities_count": len(cooccurring_entities_seen),
                    "expanded_chunks_count": len(expanded_chunks_seen),
                    "candidate_fan_out": total_expanded_added,
                    "skipped_hub_entities": skipped_hubs,
                    "fan_out_pruned": fan_out_pruned,
                })

        # 3c. Document-title metadata-based seed discovery
        if self.document_title_discovery_enabled:
            title_candidates = self.title_discoverer.discover_title_candidates(
                seed_specs=plan.seed_entities,
                scope_id=scope_id,
            )
            matched_doc_ids: Set[str] = set()
            discovered_cids: Set[str] = set()
            ambig_count = 0
            for cand in title_candidates:
                matched_doc_ids.add(cand.doc_id)
                if cand.is_ambiguous:
                    ambig_count += 1
                for cid in cand.chunk_ids:
                    discovered_cids.add(cid)
                    path_str = f"DOCUMENT_TITLE_PROVENANCE: Document Title [{cand.matched_title} ({cand.doc_id})]"
                    chunk_path_map.setdefault(cid, []).append(path_str)
                    chunk_support_count[cid] = chunk_support_count.get(cid, 0) + 1

                    if cid not in chunk_best_alignment or 0.80 > chunk_best_alignment[cid]:
                        chunk_best_alignment[cid] = 0.80
                    if cid not in chunk_min_hop or 1 < chunk_min_hop[cid]:
                        chunk_min_hop[cid] = 1

                    chunk_specs_map.setdefault(cid, set()).add(cand.seed_name)
                    chunk_provenance_types.setdefault(cid, []).append(
                        GraphProvenanceType.DOCUMENT_TITLE_PROVENANCE
                    )

                    dt_prov = DocumentTitleProvenance(
                        seed_name=cand.seed_name,
                        document_title=cand.matched_title,
                        doc_id=cand.doc_id,
                        chunk_ids=list(cand.chunk_ids),
                        matching_method=cand.matching_method,
                        is_ambiguous=cand.is_ambiguous,
                        candidate_doc_ids=list(cand.candidate_doc_ids),
                    )
                    chunk_title_details.setdefault(cid, []).append(dt_prov)

            self.last_diagnostics["document_title_diagnostics"] = {
                "enabled": True,
                "matched_documents_count": len(matched_doc_ids),
                "discovered_chunks_count": len(discovered_cids),
                "ambiguous_titles_count": ambig_count,
                "matches": [cand.model_dump() for cand in title_candidates],
            }
        else:
            self.last_diagnostics["document_title_diagnostics"] = {
                "enabled": False,
                "matched_documents_count": 0,
                "discovered_chunks_count": 0,
                "ambiguous_titles_count": 0,
                "matches": [],
            }

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
            unique_prov_types = list(dict.fromkeys(chunk_provenance_types.get(cid, [])))
            cooccur_details = chunk_cooccurrence_details.get(cid, [])
            title_details = chunk_title_details.get(cid, [])

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
                graph_provenance_types=unique_prov_types,
                cooccurrence_details=cooccur_details,
                document_title_details=title_details,
                entity_description_details=chunk_desc_details.get(cid, []),
            )
            evidence_list.append(ev)
            rank += 1

        self.last_diagnostics["evidence_count"] = len(evidence_list)
        return evidence_list
