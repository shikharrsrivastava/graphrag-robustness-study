"""Community summarization generating grounded CommunityReport objects and vector indexing."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import igraph
from pydantic import BaseModel, ConfigDict, Field
from src.core.interfaces import EmbeddingProvider
from src.core.models import CommunityReport, GroundedClaim
from src.extraction.schemas import OllamaLLMClient
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_COMMUNITY_SUMMARY_PROMPT = """You are an expert intelligence analyst and knowledge synthesizer. Your task is to generate a grounded, structured claim report for a community of related entities in a knowledge graph.

CRITICAL GROUNDING RULES:
1. Community co-membership does NOT imply a direct relationship. Two entities belonging to the same community must NEVER be assumed to have a relationship unless explicitly listed under VALIDATED INTERNAL RELATIONSHIPS.
2. Only output claims that are directly supported by the listed internal relationships.
3. Every claim must correspond to a specific relationship between a subject entity and an object entity with the exact direction and relation type provided in VALIDATED INTERNAL RELATIONSHIPS.
4. Do not hallucinate external facts, cross-member associations, or fabrications.
5. If no internal relationships connect certain members, do not invent connections between them.

COMMUNITY ID: {community_id}

MEMBER ENTITIES:
{entities_json}

VALIDATED INTERNAL RELATIONSHIPS:
{relationships_json}

INSTRUCTIONS:
1. Identify all verified relationship claims that connect these member entities based strictly on the provided internal relationships.
2. Return ONLY a valid JSON object matching the following schema with no markdown code blocks or surrounding text:

{
  "claims": [
    {
      "subject_id": "Entity ID of the subject (e.g. E001)",
      "subject_type": "Optional canonical entity type of the subject",
      "relation_type": "Exact relation type from VALIDATED INTERNAL RELATIONSHIPS",
      "object_id": "Entity ID of the object (e.g. E002)",
      "object_type": "Optional canonical entity type of the object",
      "supporting_edge_ids": ["Relationship/edge ID from VALIDATED INTERNAL RELATIONSHIPS"]
    }
  ],
  "overview": "A concise factual overview of the community theme based strictly on verified relationships."
}
"""


class GeneratedClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject_id: str = Field(..., min_length=1)
    subject_type: Optional[str] = None
    relation_type: str = Field(..., min_length=1)
    object_id: str = Field(..., min_length=1)
    object_type: Optional[str] = None
    supporting_edge_ids: List[str] = Field(default_factory=list)


class CommunitySummaryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: List[GeneratedClaim] = Field(default_factory=list)
    overview: Optional[str] = None
    summary_text: Optional[str] = None


class CommunitySummarizer:
    """Generates grounded CommunityReport summaries using structured claim validation and indexes them into ChromaDB."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        doc_store: Optional[DocumentStore] = None,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/community_summary.txt",
        prompt_template: Optional[str] = None,
        model_tag: Optional[str] = None,
        prompt_version: str = "v1.0",
    ) -> None:
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.model_tag = model_tag or self.llm_client.config.llm_model
        self.prompt_version = prompt_version

        if prompt_template is not None:
            self.prompt_template = prompt_template
        elif prompt_template_path is not None and Path(prompt_template_path).exists():
            self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        else:
            self.prompt_template = DEFAULT_COMMUNITY_SUMMARY_PROMPT

    @classmethod
    def validate_claims(
        cls,
        candidate_claims: List[GeneratedClaim],
        member_vertices: List[igraph.Vertex],
        community_edges: List[igraph.Edge],
    ) -> Tuple[List[GroundedClaim], int]:
        """Validate candidate claims against actual member vertices and internal community edges.

        Enforces:
        1. Subject and object entities must exist within the community.
        2. Canonical entity types must match if provided.
        3. Directed relationship (subject_id, relation_type, object_id) must exist in internal edges.
        4. Supporting edge IDs must match existing edges.

        Returns:
            Tuple of (verified_claims, unsupported_claims_rejected_count)
        """
        vertex_map: Dict[str, Dict[str, Any]] = {
            v["entity_id"]: {
                "name": v.attributes().get("label", "") or v["entity_id"],
                "entity_type": v.attributes().get("entity_type", ""),
            }
            for v in member_vertices
        }

        # Index internal edges by (source_entity_id, normalized_relation_type, target_entity_id)
        edges_by_triple: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        all_edge_ids: Set[str] = set()

        for e in community_edges:
            edge_id = e.attributes().get("relationship_id") or f"e_{e.index}"
            src = e["source_entity_id"]
            tgt = e["target_entity_id"]
            rel = e["relation_type"].strip().upper().replace(" ", "_")
            quotes = e.attributes().get("evidence_quotes", [])
            chunks = e.attributes().get("source_chunk_ids", [])

            all_edge_ids.add(edge_id)
            key = (src, rel, tgt)
            if key not in edges_by_triple:
                edges_by_triple[key] = []
            edges_by_triple[key].append({
                "edge_id": edge_id,
                "evidence_quotes": quotes,
                "source_chunk_ids": chunks,
            })

        verified_claims: List[GroundedClaim] = []
        rejected_count = 0
        seen_triples: Set[Tuple[str, str, str]] = set()

        for claim in candidate_claims:
            # 1. Existence check
            if claim.subject_id not in vertex_map or claim.object_id not in vertex_map:
                logger.warning(
                    "Rejected claim: endpoints (%s -> %s) not both in community vertices.",
                    claim.subject_id,
                    claim.object_id,
                )
                rejected_count += 1
                continue

            sub_info = vertex_map[claim.subject_id]
            obj_info = vertex_map[claim.object_id]

            # 2. Entity type consistency
            if claim.subject_type and claim.subject_type.strip().upper() != sub_info["entity_type"].strip().upper():
                logger.warning(
                    "Rejected claim: subject %s claimed as type %s but canonical type is %s.",
                    claim.subject_id,
                    claim.subject_type,
                    sub_info["entity_type"],
                )
                rejected_count += 1
                continue

            if claim.object_type and claim.object_type.strip().upper() != obj_info["entity_type"].strip().upper():
                logger.warning(
                    "Rejected claim: object %s claimed as type %s but canonical type is %s.",
                    claim.object_id,
                    claim.object_type,
                    obj_info["entity_type"],
                )
                rejected_count += 1
                continue

            # 3. Directed relationship existence
            norm_rel = claim.relation_type.strip().upper().replace(" ", "_")
            triple_key = (claim.subject_id, norm_rel, claim.object_id)
            if triple_key not in edges_by_triple:
                logger.warning(
                    "Rejected unsupported claim: (%s, %s, %s) does not exist in internal edges.",
                    claim.subject_id,
                    norm_rel,
                    claim.object_id,
                )
                rejected_count += 1
                continue

            matching_edges = edges_by_triple[triple_key]
            matching_edge_ids = {m["edge_id"] for m in matching_edges}

            # 4. Supporting edge IDs validation
            if claim.supporting_edge_ids:
                invalid_ids = [eid for eid in claim.supporting_edge_ids if eid not in matching_edge_ids]
                if invalid_ids:
                    logger.warning(
                        "Rejected claim: supporting edge IDs %s do not match valid edge IDs %s for triple %s.",
                        invalid_ids,
                        list(matching_edge_ids),
                        triple_key,
                    )
                    rejected_count += 1
                    continue
                resolved_edge_ids = sorted(list(set(claim.supporting_edge_ids)))
            else:
                resolved_edge_ids = sorted(list(matching_edge_ids))

            # Deduplicate verified triples
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            # Preserve source chunk IDs from matching edges
            collected_chunk_ids: Set[str] = set()
            for m in matching_edges:
                if not claim.supporting_edge_ids or m["edge_id"] in claim.supporting_edge_ids:
                    collected_chunk_ids.update(m.get("source_chunk_ids", []))

            verified_claims.append(
                GroundedClaim(
                    subject_id=claim.subject_id,
                    subject_name=sub_info["name"],
                    subject_type=sub_info["entity_type"],
                    relation_type=norm_rel,
                    object_id=claim.object_id,
                    object_name=obj_info["name"],
                    object_type=obj_info["entity_type"],
                    supporting_edge_ids=resolved_edge_ids,
                    source_chunk_ids=sorted(list(collected_chunk_ids)),
                )
            )

        return verified_claims, rejected_count

    @classmethod
    def render_summary(
        cls,
        community_id: int,
        member_vertices: List[igraph.Vertex],
        verified_claims: List[GroundedClaim],
    ) -> str:
        """Deterministically render a verified community summary from members and verified claims."""
        member_names = ", ".join(
            f"{v.attributes().get('label', '') or v['entity_id']} ({v.attributes().get('entity_type', '')})"
            for v in member_vertices
        )
        lines = [
            f"Community {community_id} comprises {len(member_vertices)} entities: {member_names}."
        ]
        if verified_claims:
            lines.append("Verified relationships:")
            for c in verified_claims:
                sub = c.subject_name or c.subject_id
                obj = c.object_name or c.object_id
                edges_str = f" [Edges: {', '.join(c.supporting_edge_ids)}]" if c.supporting_edge_ids else ""
                lines.append(f"- {sub} ({c.subject_id}) -[{c.relation_type}]-> {obj} ({c.object_id}){edges_str}")
        else:
            lines.append("No direct internal relationships exist between these members.")
        return "\n".join(lines)

    def summarize_communities(
        self,
        communities: List[Dict[str, Any]],
        graph: igraph.Graph,
        doc_store: Optional[DocumentStore] = None,
    ) -> List[CommunityReport]:
        """Generate grounded CommunityReport summaries for each community and optionally index them in ChromaDB."""
        active_doc_store = doc_store or self.doc_store
        reports: List[CommunityReport] = []

        for comm in communities:
            comm_id = comm["community_id"]
            member_ids: List[str] = comm["member_entity_ids"]
            member_set = set(member_ids)

            # 1. Gather member entities from graph vertices
            member_vertices = [v for v in graph.vs if v["entity_id"] in member_set]
            entities_data = [
                {
                    "entity_id": v["entity_id"],
                    "name": v.attributes().get("label", ""),
                    "entity_type": v.attributes().get("entity_type", ""),
                    "description": v.attributes().get("description", ""),
                    "aliases": v.attributes().get("aliases", []),
                }
                for v in member_vertices
            ]
            vertex_name_map = {
                v["entity_id"]: v.attributes().get("label", "") or v["entity_id"]
                for v in member_vertices
            }

            # 2. Gather internal relationships between members
            community_edges = [
                e
                for e in graph.es
                if e["source_entity_id"] in member_set and e["target_entity_id"] in member_set
            ]
            relationships_data = [
                {
                    "edge_id": e.attributes().get("relationship_id") or f"e_{e.index}",
                    "source_id": e["source_entity_id"],
                    "source_name": vertex_name_map.get(e["source_entity_id"], ""),
                    "relation_type": e["relation_type"],
                    "target_id": e["target_entity_id"],
                    "target_name": vertex_name_map.get(e["target_entity_id"], ""),
                    "evidence_quotes": e.attributes().get("evidence_quotes", []),
                    "source_chunk_ids": e.attributes().get("source_chunk_ids", []),
                }
                for e in community_edges
            ]

            # 3. Gather supporting chunk IDs for provenance tracking
            supporting_chunk_ids: Set[str] = set()
            for v in member_vertices:
                supporting_chunk_ids.update(v.attributes().get("source_chunk_ids", []))
            for e in community_edges:
                supporting_chunk_ids.update(e.attributes().get("source_chunk_ids", []))

            # 4. Generate summary: fast-path for singleton communities without internal edges
            if len(member_vertices) <= 1 and len(community_edges) == 0:
                if member_vertices:
                    v = member_vertices[0]
                    v_name = v.attributes().get("label", "") or v["entity_id"]
                    v_type = v.attributes().get("entity_type", "")
                    v_desc = v.attributes().get("description", "")
                    if v_desc:
                        summary_text = f"{v_name} ({v_type}): {v_desc}".strip()
                    else:
                        summary_text = f"Singleton community for entity {v_name} ({v_type})."
                else:
                    summary_text = f"Community {comm_id} containing members: {', '.join(member_ids)}."

                verified_claims: List[GroundedClaim] = []
                unsupported_rejected = 0
                verification_status = "VERIFIED"

            elif len(community_edges) == 0:
                # Multi-member community with zero internal edges: deterministic grounded fallback
                summary_text = self.render_summary(comm_id, member_vertices, [])
                verified_claims = []
                unsupported_rejected = 0
                verification_status = "FALLBACK"

            else:
                # Multi-entity community with internal edges: call LLM to synthesize structured claims
                prompt = (
                    self.prompt_template.replace("{community_id}", str(comm_id))
                    .replace("{entities_json}", json.dumps(entities_data, indent=2))
                    .replace("{relationships_json}", json.dumps(relationships_data, indent=2))
                )

                try:
                    output: CommunitySummaryOutput = self.llm_client.generate_json(
                        prompt, CommunitySummaryOutput
                    )
                    candidate_claims = output.claims
                    verified_claims, unsupported_rejected = self.validate_claims(
                        candidate_claims, member_vertices, community_edges
                    )
                    if verified_claims:
                        summary_text = self.render_summary(comm_id, member_vertices, verified_claims)
                        verification_status = "VERIFIED"
                    else:
                        summary_text = self.render_summary(comm_id, member_vertices, [])
                        verification_status = "FALLBACK"
                except Exception as e:
                    logger.warning("Failed LLM community summary for community %d: %s", comm_id, e)
                    summary_text = self.render_summary(comm_id, member_vertices, [])
                    verification_status = "FALLBACK"
                    verified_claims = []
                    unsupported_rejected = 0

            report_id = f"COMM_REPORT_{comm_id:04d}"
            target_kb_id = comm.get("kb_id") or (active_doc_store.kb_id if active_doc_store else "default_kb")

            report = CommunityReport(
                kb_id=target_kb_id,
                report_id=report_id,
                community_id=comm_id,
                partition_id=comm["partition_id"],
                member_entity_ids=member_ids,
                supporting_chunk_ids=sorted(list(supporting_chunk_ids)),
                summary_text=summary_text,
                model_tag=self.model_tag,
                prompt_version=self.prompt_version,
                verified_claims=verified_claims,
                unsupported_claims_rejected=unsupported_rejected,
                verification_status=verification_status,
            )
            reports.append(report)

        # 5. Save to DocumentStore if available
        if active_doc_store is not None:
            active_doc_store.save_community_reports(reports)

        # 6. Index into ChromaVectorStore if available
        if self.vector_store is not None and self.embedding_provider is not None:
            try:
                texts = [r.summary_text for r in reports]
                embeddings = self.embedding_provider.embed_batch(texts)
                self.vector_store.upsert_community_reports(reports, embeddings)
            except Exception as e:
                logger.warning("Failed to embed or upsert community reports to vector store: %s", e)

        return reports
