"""Core Pydantic data models preserving provenance across all 19 pipeline stages."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNCERTAIN = "UNCERTAIN"


class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    REJECTED_QUOTE_MISMATCH = "REJECTED_QUOTE_MISMATCH"
    REJECTED_ONTOLOGY = "REJECTED_ONTOLOGY"
    REJECTED_NEGATION = "REJECTED_NEGATION"
    PENDING = "PENDING"


class QueryIntent(str, Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"
    MIXED = "MIXED"


class ProvenanceRoute(str, Enum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    COMMUNITY = "COMMUNITY"


class GraphProvenanceType(str, Enum):
    GRAPH_SEED_PROVENANCE = "GRAPH_SEED_PROVENANCE"
    GRAPH_EDGE_PROVENANCE = "GRAPH_EDGE_PROVENANCE"
    ENTITY_CHUNK_COOCCURRENCE = "ENTITY_CHUNK_COOCCURRENCE"
    DOCUMENT_TITLE_PROVENANCE = "DOCUMENT_TITLE_PROVENANCE"


class DocumentTitleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_name: str = Field(..., min_length=1, description="Original seed entity query string")
    document_title: str = Field(..., min_length=1, description="Authoritative matched document title")
    doc_id: str = Field(..., min_length=1, description="Matched document identifier")
    chunk_ids: List[str] = Field(default_factory=list, description="Discovered chunk IDs for this document")
    matching_method: str = Field(
        default="EXACT_NORMALIZED",
        description="EXACT_NORMALIZED, PUNCTUATION_NORMALIZED, or ARTICLE_NORMALIZED",
    )
    is_ambiguous: bool = Field(default=False, description="True if multiple documents matched the title")
    candidate_doc_ids: List[str] = Field(
        default_factory=list,
        description="All document IDs matching the title if ambiguous",
    )


class CooccurrenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_entity_id: str = Field(..., min_length=1, description="Resolved seed entity ID that initiated the path")
    direct_chunk_id: str = Field(..., min_length=1, description="Direct chunk where the seed entity appears")
    cooccurring_entity_id: str = Field(..., min_length=1, description="Entity co-occurring in the direct chunk")
    expanded_chunk_id: str = Field(..., min_length=1, description="Target chunk where cooccurring entity also appears")
    expansion_depth: int = Field(default=1, ge=1, le=1, description="Bipartite expansion depth (strictly 1-hop)")
    skipped_hub_entities: List[str] = Field(default_factory=list, description="Hub entities pruned during this expansion")
    fan_out_pruned: bool = Field(default=False, description="Whether expansion was halted by fan-out limits")


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1, description="Isolated Knowledge Base ID")
    doc_id: str = Field(..., min_length=1, description="Stable document identifier")
    filename: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=1)
    total_pages: int = Field(..., ge=0)
    empty_pages: List[int] = Field(default_factory=list)
    is_scanned_or_empty: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(default="")
    char_count: int = Field(..., ge=0)
    is_empty: bool = False
    content_hash: str = Field(..., min_length=1)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    page_numbers: List[int] = Field(..., min_length=1, description="Pages spanned by this chunk")
    chunk_index: int = Field(..., ge=0, description="Deterministic order within document")
    text: str = Field(..., min_length=1)
    char_count: int = Field(..., ge=1)
    content_hash: str = Field(..., min_length=1)
    scope_id: Optional[str] = Field(default=None, description="Optional question/context scope filter")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    mention_id: str = Field(..., min_length=1)
    surface_name: str = Field(..., min_length=1)
    normalized_name: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    source_chunk_id: str = Field(..., min_length=1)
    source_doc_id: str = Field(..., min_length=1)
    page_numbers: List[int] = Field(default_factory=list)
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=0)

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, v: str) -> str:
        return v.strip().upper()


class CanonicalEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1, description="Stable canonical entity ID, e.g., E001")
    name: str = Field(..., min_length=1)
    normalized_name: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    aliases: List[str] = Field(default_factory=list)
    source_mention_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(..., min_length=1)
    source_doc_ids: List[str] = Field(..., min_length=1)

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, v: str) -> str:
        return v.strip().upper()


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    relationship_id: str = Field(..., min_length=1)
    source_entity_id: str = Field(..., min_length=1, description="Mention ID before resolution, Canonical ID after")
    target_entity_id: str = Field(..., min_length=1, description="Mention ID before resolution, Canonical ID after")
    relation_type: str = Field(..., min_length=1)
    source_chunk_ids: List[str] = Field(..., min_length=1)
    source_doc_ids: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(..., min_length=1)
    support_count: int = Field(default=1, ge=1)
    polarity: Polarity = Field(default=Polarity.POSITIVE)
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)
    uncalibrated_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scope_id: Optional[str] = Field(default=None)

    @field_validator("relation_type")
    @classmethod
    def normalize_relation_type(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(..., min_length=1)
    subject_name: str = Field(default="")
    subject_type: Optional[str] = Field(default=None)
    relation_type: str = Field(..., min_length=1)
    object_id: str = Field(..., min_length=1)
    object_name: str = Field(default="")
    object_type: Optional[str] = Field(default=None)
    supporting_edge_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("relation_type")
    @classmethod
    def normalize_relation_type(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")


class CommunityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    report_id: str = Field(..., min_length=1)
    community_id: int = Field(..., ge=0)
    partition_id: str = Field(..., min_length=1, description="Identifies Leiden seed/resolution/graph hash")
    member_entity_ids: List[str] = Field(..., min_length=1)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    summary_text: str = Field(..., min_length=1)
    model_tag: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    verified_claims: List[GroundedClaim] = Field(default_factory=list)
    unsupported_claims_rejected: int = Field(default=0, ge=0)
    verification_status: str = Field(default="UNVERIFIED")  # "VERIFIED", "FALLBACK", "UNVERIFIED"


class SeedEntitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    expected_type: Optional[str] = None


class ResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_name: str = Field(..., description="Original seed entity name string")
    expected_type: Optional[str] = Field(default=None, description="Expected entity type from SeedEntitySpec, or None")
    candidate_id: Optional[str] = Field(default=None, description="Canonical entity ID (e.g., E032)")
    candidate_name: Optional[str] = Field(default=None, description="Canonical entity name")
    candidate_type: Optional[str] = Field(default=None, description="Canonical entity type from KB")
    matching_method: Optional[str] = Field(default=None, description="EXACT_CANONICAL, PERSON_NAME_VARIATION, EXACT_ALIAS, AFFIX_STRIPPED, EMBEDDING, or NONE")
    type_compatible: bool = Field(default=False, description="Whether candidate_type matches expected_type")
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic ordinal ranking heuristic (not a calibrated probability or Bayesian posterior). "
            "Exact lexical matches (1.00/0.80), aliases (0.75/0.60), and embedding similarities represent different "
            "matching modalities and should not be interpreted as directly comparable probabilities."
        ),
    )
    similarity_score: Optional[float] = Field(default=None, description="Raw embedding cosine similarity if from vector search, else None")
    ranking: int = Field(default=1, ge=0, description="Rank within candidate pool for this seed")
    status: str = Field(..., description="ACCEPTED, AMBIGUOUS, REJECTED, or UNRESOLVED")
    reason: str = Field(..., description="Human-readable explanation of the resolution decision")


class EntityAttributeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(..., min_length=1)
    entity_name: str = Field(..., min_length=1)
    attribute_name: str = Field(..., min_length=1)
    extracted_value: str = Field(..., min_length=1)
    evidence_quote: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_verified: bool = Field(default=False)
    verification_notes: Optional[str] = Field(default=None)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    intent: QueryIntent = Field(default=QueryIntent.LOCAL)
    seed_entities: List[SeedEntitySpec] = Field(default_factory=list)
    target_type: Optional[str] = None
    relationship_path: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None
    invalid_relationship_types: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    filename: str = Field(default="")
    page_numbers: List[int] = Field(default_factory=list)
    text: str = Field(..., min_length=1)
    routes: List[ProvenanceRoute] = Field(..., min_length=1)
    graph_rank: Optional[int] = Field(default=None, ge=1)
    vector_rank: Optional[int] = Field(default=None, ge=1)
    vector_distance: Optional[float] = Field(default=None)
    rrf_score: float = Field(default=0.0, ge=0.0)
    supporting_graph_paths: List[str] = Field(default_factory=list)
    supporting_edge_ids: List[str] = Field(default_factory=list)
    community_ids: List[int] = Field(default_factory=list)
    graph_provenance_types: List[GraphProvenanceType] = Field(default_factory=list)
    cooccurrence_details: List[CooccurrenceProvenance] = Field(default_factory=list)
    document_title_details: List[DocumentTitleProvenance] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    query_plan: QueryPlan
    resolved_seed_entity_ids: List[str] = Field(default_factory=list)
    evidence_list: List[Evidence] = Field(default_factory=list)
    community_reports: List[CommunityReport] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[str] = None
    route_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    generated_answer: Optional[str] = None
    canonical_answer: Optional[str] = None
    answer_abstained: Optional[bool] = None
    cited_chunk_ids: List[str] = Field(default_factory=list)
    citation_references: List[Dict[str, Any]] = Field(default_factory=list)
