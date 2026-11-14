"""Evidence merger combining heterogeneous evidence streams by chunk_id."""

from __future__ import annotations
import logging
from typing import Dict, List
from src.core.models import Evidence, ProvenanceRoute

logger = logging.getLogger(__name__)


class EvidenceMerger:
    """Merges multiple evidence streams, combining provenance routes, ranks, and graph paths."""

    @classmethod
    def merge(cls, evidence_lists: List[List[Evidence]]) -> List[Evidence]:
        """Merge multiple evidence lists into deduplicated Evidence items grouped by chunk_id."""
        merged_dict: Dict[str, Evidence] = {}

        for ev_list in evidence_lists:
            for ev in ev_list:
                cid = ev.chunk_id
                if cid not in merged_dict:
                    # Deep copy of attributes to avoid mutation
                    merged_dict[cid] = Evidence(
                        kb_id=ev.kb_id,
                        chunk_id=ev.chunk_id,
                        doc_id=ev.doc_id,
                        filename=ev.filename,
                        page_numbers=list(ev.page_numbers),
                        text=ev.text,
                        routes=list(ev.routes),
                        graph_rank=ev.graph_rank,
                        vector_rank=ev.vector_rank,
                        vector_distance=ev.vector_distance,
                        rrf_score=ev.rrf_score,
                        supporting_graph_paths=list(ev.supporting_graph_paths),
                        supporting_edge_ids=list(ev.supporting_edge_ids),
                        community_ids=list(ev.community_ids),
                        graph_provenance_types=list(ev.graph_provenance_types),
                        cooccurrence_details=list(ev.cooccurrence_details),
                        document_title_details=list(ev.document_title_details),
                        entity_description_details=list(ev.entity_description_details),
                    )
                else:
                    existing = merged_dict[cid]

                    # Union routes
                    for route in ev.routes:
                        if route not in existing.routes:
                            existing.routes.append(route)

                    # Update ranks to best (minimum) rank observed
                    if ev.graph_rank is not None:
                        if existing.graph_rank is None or ev.graph_rank < existing.graph_rank:
                            existing.graph_rank = ev.graph_rank

                    if ev.vector_rank is not None:
                        if existing.vector_rank is None or ev.vector_rank < existing.vector_rank:
                            existing.vector_rank = ev.vector_rank
                            existing.vector_distance = ev.vector_distance

                    # Union supporting graph paths
                    for path in ev.supporting_graph_paths:
                        if path not in existing.supporting_graph_paths:
                            existing.supporting_graph_paths.append(path)

                    # Union supporting edge ids
                    for edge_id in ev.supporting_edge_ids:
                        if edge_id not in existing.supporting_edge_ids:
                            existing.supporting_edge_ids.append(edge_id)

                    # Union community ids
                    for comm_id in ev.community_ids:
                        if comm_id not in existing.community_ids:
                            existing.community_ids.append(comm_id)
                    existing.community_ids.sort()

                    # Union graph provenance types
                    for pt in ev.graph_provenance_types:
                        if pt not in existing.graph_provenance_types:
                            existing.graph_provenance_types.append(pt)

                    # Union cooccurrence details
                    for cd in ev.cooccurrence_details:
                        if cd not in existing.cooccurrence_details:
                            existing.cooccurrence_details.append(cd)

                    # Union document title details
                    for dtd in ev.document_title_details:
                        if dtd not in existing.document_title_details:
                            existing.document_title_details.append(dtd)

                    # Union entity description details
                    for edd in ev.entity_description_details:
                        if edd not in existing.entity_description_details:
                            existing.entity_description_details.append(edd)

        return list(merged_dict.values())
