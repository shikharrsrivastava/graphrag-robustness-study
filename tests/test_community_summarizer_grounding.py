"""Comprehensive regression tests for graph-grounded community summarization.

Tests:
Test A: Unsupported relationship rejection (Film A / Film B sharing only COUNTRY hub;
        claiming Film A directed by Film B's director rejected).
Test B: Entity ID / type consistency (E018 is FILM, claiming it as PERSON rejected).
Test C: Valid relationship preservation (genuine DIRECTED edge preserved).
Test D: Directionality rejection (reversed relation rejected).
Test E: Safe fallback when validation fails or no edges exist.
Test F: Legacy report safety (unverified legacy report filtered in ContextBuilder).
Test G: Provenance preservation on accepted claims.
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import igraph
from src.core.models import (
    CommunityReport,
    GroundedClaim,
    QueryPlan,
    RetrievalResult,
)
from src.communities.community_summarizer import (
    CommunitySummarizer,
    CommunitySummaryOutput,
    GeneratedClaim,
)
from src.extraction.schemas import OllamaLLMClient
from src.retrieval.context_builder import ContextBuilder


def _build_test_graph() -> igraph.Graph:
    """Helper to construct an igraph with known entities and typed edges."""
    g = igraph.Graph(directed=True)
    # Add vertices
    vertices = [
        {"entity_id": "E001", "label": "Ed Wood (film)", "entity_type": "FILM", "description": "1994 biographical film"},
        {"entity_id": "E002", "label": "Deliver Us from Evil", "entity_type": "FILM", "description": "2014 horror film"},
        {"entity_id": "E003", "label": "United States", "entity_type": "COUNTRY", "description": "North American country"},
        {"entity_id": "E004", "label": "Scott Derrickson", "entity_type": "PERSON", "description": "American director"},
        {"entity_id": "E010", "label": "Tim Burton", "entity_type": "PERSON", "description": "American filmmaker"},
    ]
    for v in vertices:
        g.add_vertex(
            name=v["entity_id"],
            entity_id=v["entity_id"],
            label=v["label"],
            entity_type=v["entity_type"],
            description=v["description"],
            source_chunk_ids=[f"C_{v['entity_id']}"],
        )

    # Add edges
    # E001 (Ed Wood film) -> E003 (United States)
    g.add_edge(
        g.vs.find(name="E001").index,
        g.vs.find(name="E003").index,
        relationship_id="REL_001",
        source_entity_id="E001",
        target_entity_id="E003",
        relation_type="PRODUCED_IN",
        evidence_quotes=["Produced in the United States."],
        source_chunk_ids=["C_D001"],
    )
    # E002 (Deliver Us from Evil) -> E003 (United States)
    g.add_edge(
        g.vs.find(name="E002").index,
        g.vs.find(name="E003").index,
        relationship_id="REL_002",
        source_entity_id="E002",
        target_entity_id="E003",
        relation_type="PRODUCED_IN",
        evidence_quotes=["American supernatural horror film."],
        source_chunk_ids=["C_D006"],
    )
    # E004 (Scott Derrickson) -> E002 (Deliver Us from Evil)
    g.add_edge(
        g.vs.find(name="E004").index,
        g.vs.find(name="E002").index,
        relationship_id="REL_003",
        source_entity_id="E004",
        target_entity_id="E002",
        relation_type="DIRECTED",
        evidence_quotes=["Directed by Scott Derrickson."],
        source_chunk_ids=["C_D006"],
    )
    # E010 (Tim Burton) -> E001 (Ed Wood film)
    g.add_edge(
        g.vs.find(name="E010").index,
        g.vs.find(name="E001").index,
        relationship_id="REL_010",
        source_entity_id="E010",
        target_entity_id="E001",
        relation_type="DIRECTED",
        evidence_quotes=["Directed and produced by Tim Burton."],
        source_chunk_ids=["C_D001"],
    )
    return g


def test_claim_validation_unsupported_relationship_rejection() -> None:
    """Test A: Unsupported relationship rejection.
    Film A (E001) and Film B (E002) share only a COUNTRY hub (E003).
    A claim that Film A was directed by Film B's director (E004) must be strictly rejected.
    """
    g = _build_test_graph()
    member_vertices = [v for v in g.vs if v["entity_id"] in {"E001", "E002", "E003", "E004"}]
    community_edges = [
        e for e in g.es
        if e["source_entity_id"] in {"E001", "E002", "E003", "E004"}
        and e["target_entity_id"] in {"E001", "E002", "E003", "E004"}
    ]

    # LLM hallucinates that E004 (Scott Derrickson) directed E001 (Ed Wood film)
    hallucinated_claim = GeneratedClaim(
        subject_id="E004",
        relation_type="DIRECTED",
        object_id="E001",
        supporting_edge_ids=["REL_003"],  # Fabricated or misattributed edge ID
    )
    # Valid claim that E004 directed E002
    valid_claim = GeneratedClaim(
        subject_id="E004",
        relation_type="DIRECTED",
        object_id="E002",
        supporting_edge_ids=["REL_003"],
    )

    verified, rejected_count = CommunitySummarizer.validate_claims(
        [hallucinated_claim, valid_claim],
        member_vertices,
        community_edges,
    )

    assert rejected_count == 1
    assert len(verified) == 1
    # Only genuine relationship E004 -> E002 is verified
    assert verified[0].subject_id == "E004"
    assert verified[0].object_id == "E002"
    assert verified[0].relation_type == "DIRECTED"
    # E001 (Ed Wood) must NOT be connected to E004
    assert not any(c.subject_id == "E004" and c.object_id == "E001" for c in verified)


def test_claim_validation_entity_type_consistency() -> None:
    """Test B: Entity ID / type consistency.
    E002 is FILM in the graph. A claim presenting E002 as PERSON must be rejected.
    """
    g = _build_test_graph()
    member_vertices = [v for v in g.vs if v["entity_id"] in {"E004", "E002"}]
    community_edges = [
        e for e in g.es
        if e["source_entity_id"] in {"E004", "E002"}
        and e["target_entity_id"] in {"E004", "E002"}
    ]

    # Claim presenting target E002 as PERSON instead of FILM
    inconsistent_claim = GeneratedClaim(
        subject_id="E004",
        subject_type="PERSON",
        relation_type="DIRECTED",
        object_id="E002",
        object_type="PERSON",  # Mismatch: E002 is FILM!
        supporting_edge_ids=["REL_003"],
    )

    verified, rejected_count = CommunitySummarizer.validate_claims(
        [inconsistent_claim],
        member_vertices,
        community_edges,
    )

    assert rejected_count == 1
    assert len(verified) == 0


def test_claim_validation_valid_relationship_preservation() -> None:
    """Test C: Valid relationship preservation.
    Genuine DIRECTED edge (E010 -> E001) must be validated and accepted.
    """
    g = _build_test_graph()
    member_vertices = [v for v in g.vs if v["entity_id"] in {"E010", "E001"}]
    community_edges = [
        e for e in g.es
        if e["source_entity_id"] in {"E010", "E001"}
        and e["target_entity_id"] in {"E010", "E001"}
    ]

    valid_claim = GeneratedClaim(
        subject_id="E010",
        subject_type="PERSON",
        relation_type="DIRECTED",
        object_id="E001",
        object_type="FILM",
        supporting_edge_ids=["REL_010"],
    )

    verified, rejected_count = CommunitySummarizer.validate_claims(
        [valid_claim],
        member_vertices,
        community_edges,
    )

    assert rejected_count == 0
    assert len(verified) == 1
    claim = verified[0]
    assert claim.subject_id == "E010"
    assert claim.subject_name == "Tim Burton"
    assert claim.subject_type == "PERSON"
    assert claim.relation_type == "DIRECTED"
    assert claim.object_id == "E001"
    assert claim.object_name == "Ed Wood (film)"
    assert claim.object_type == "FILM"
    assert claim.supporting_edge_ids == ["REL_010"]
    assert "C_D001" in claim.source_chunk_ids


def test_claim_validation_directionality_rejection() -> None:
    """Test D: Directionality rejection.
    Graph has E010 (PERSON) -[DIRECTED]-> E001 (FILM).
    A reversed claim: E001 -[DIRECTED]-> E010 must be strictly rejected.
    """
    g = _build_test_graph()
    member_vertices = [v for v in g.vs if v["entity_id"] in {"E010", "E001"}]
    community_edges = [
        e for e in g.es
        if e["source_entity_id"] in {"E010", "E001"}
        and e["target_entity_id"] in {"E010", "E001"}
    ]

    reversed_claim = GeneratedClaim(
        subject_id="E001",  # Subject is FILM
        relation_type="DIRECTED",
        object_id="E010",  # Object is PERSON
        supporting_edge_ids=["REL_010"],
    )

    verified, rejected_count = CommunitySummarizer.validate_claims(
        [reversed_claim],
        member_vertices,
        community_edges,
    )

    assert rejected_count == 1
    assert len(verified) == 0


def test_safe_fallback_when_validation_fails_or_no_edges() -> None:
    """Test E: Safe fallback when validation fails or no edges exist.
    1. Multi-member community with zero edges -> FALLBACK summary.
    2. Multi-member community where LLM call fails -> FALLBACK summary.
    3. Multi-member community where all claims are rejected -> FALLBACK summary.
    """
    g = igraph.Graph(directed=True)
    g.add_vertex(name="E1", entity_id="E1", label="Entity One", entity_type="CONCEPT", description="First")
    g.add_vertex(name="E2", entity_id="E2", label="Entity Two", entity_type="CONCEPT", description="Second")
    # No edges

    mock_client = OllamaLLMClient(custom_generator=lambda p: json.dumps({"claims": []}))
    summarizer = CommunitySummarizer(llm_client=mock_client)

    # 1. Zero edges in community
    comm_no_edges = [{"community_id": 1, "member_entity_ids": ["E1", "E2"], "partition_id": "p1"}]
    reports = summarizer.summarize_communities(comm_no_edges, g)
    assert len(reports) == 1
    assert reports[0].verification_status == "FALLBACK"
    assert "No direct internal relationships exist between these members." in reports[0].summary_text
    assert "Entity One" in reports[0].summary_text
    assert "Entity Two" in reports[0].summary_text

    # 2. Edges exist, but LLM call throws exception
    g.add_edge(0, 1, relationship_id="R_TEMP", source_entity_id="E1", target_entity_id="E2", relation_type="RELATES_TO")

    def failing_generator(p: str) -> str:
        raise RuntimeError("LLM connection timeout")

    fail_client = OllamaLLMClient(custom_generator=failing_generator)
    fail_summarizer = CommunitySummarizer(llm_client=fail_client)
    comm_with_edge = [{"community_id": 2, "member_entity_ids": ["E1", "E2"], "partition_id": "p2"}]
    fail_reports = fail_summarizer.summarize_communities(comm_with_edge, g)
    assert len(fail_reports) == 1
    assert fail_reports[0].verification_status == "FALLBACK"
    assert "No direct internal relationships exist between these members." in fail_reports[0].summary_text

    # 3. Edges exist, but LLM returns all unsupported claims
    def hallucinating_generator(p: str) -> str:
        return json.dumps({
            "claims": [
                {"subject_id": "E1", "relation_type": "INVENTED_REL", "object_id": "E2"}
            ]
        })

    hallucinating_client = OllamaLLMClient(custom_generator=hallucinating_generator)
    h_summarizer = CommunitySummarizer(llm_client=hallucinating_client)
    h_reports = h_summarizer.summarize_communities(comm_with_edge, g)
    assert len(h_reports) == 1
    assert h_reports[0].verification_status == "FALLBACK"
    assert h_reports[0].unsupported_claims_rejected == 1
    assert len(h_reports[0].verified_claims) == 0


def test_legacy_report_safety_in_context_builder() -> None:
    """Test F: Legacy report safety in ContextBuilder.
    ContextBuilder must enforce an allowlist of {'VERIFIED', 'FALLBACK'}.
    UNVERIFIED, unknown, empty, or missing statuses must be omitted from context
    by default, while VERIFIED and FALLBACK reports are preserved.
    """
    unverified_legacy = CommunityReport(
        kb_id="kb1",
        report_id="LEGACY_COMM_0000",
        community_id=0,
        partition_id="part0",
        member_entity_ids=["E001", "E004"],
        summary_text="Hallucinated: Ed Wood was directed by Scott Derrickson.",
        verification_status="UNVERIFIED",
        model_tag="test",
        prompt_version="v1.0",
    )

    unknown_status_report = CommunityReport(
        kb_id="kb1",
        report_id="UNKNOWN_COMM_0003",
        community_id=3,
        partition_id="part3",
        member_entity_ids=["E001"],
        summary_text="Unknown status summary.",
        verification_status="CORRUPTED_OR_UNKNOWN",
        model_tag="test",
        prompt_version="v1.0",
    )

    empty_status_report = CommunityReport(
        kb_id="kb1",
        report_id="EMPTY_COMM_0004",
        community_id=4,
        partition_id="part4",
        member_entity_ids=["E001"],
        summary_text="Empty status summary.",
        verification_status="",
        model_tag="test",
        prompt_version="v1.0",
    )

    fallback_report = CommunityReport(
        kb_id="kb1",
        report_id="FALLBACK_COMM_0002",
        community_id=2,
        partition_id="part2",
        member_entity_ids=["E001", "E002"],
        summary_text="Fallback: Community comprises 2 entities with no direct relations.",
        verification_status="FALLBACK",
        model_tag="test",
        prompt_version="v1.0",
    )

    verified_report = CommunityReport(
        kb_id="kb1",
        report_id="VERIFIED_COMM_0001",
        community_id=1,
        partition_id="part1",
        member_entity_ids=["E004", "E002"],
        summary_text="Verified: Scott Derrickson directed Deliver Us from Evil.",
        verification_status="VERIFIED",
        model_tag="test",
        prompt_version="v1.0",
    )

    query_plan = QueryPlan(question="Test question?")
    res = RetrievalResult(
        kb_id="kb1",
        question="Test question?",
        query_plan=query_plan,
        community_reports=[unverified_legacy, unknown_status_report, empty_status_report, fallback_report, verified_report],
    )

    # Default ContextBuilder should only allow VERIFIED and FALLBACK
    builder = ContextBuilder(max_chars=4000)
    context = builder.build_context(res)

    assert "Hallucinated: Ed Wood was directed by Scott Derrickson." not in context
    assert "Unknown status summary." not in context
    assert "Empty status summary." not in context
    assert "[Community 0]" not in context
    assert "[Community 3]" not in context
    assert "[Community 4]" not in context

    # Verified and Fallback must both be included
    assert "Verified: Scott Derrickson directed Deliver Us from Evil." in context
    assert "[Community 1]" in context
    assert "Fallback: Community comprises 2 entities with no direct relations." in context
    assert "[Community 2]" in context

    # When allow_unverified_communities is explicitly enabled, all reports are included
    builder_allow = ContextBuilder(max_chars=4000, allow_unverified_communities=True)
    context_allow = builder_allow.build_context(res)
    assert "Hallucinated: Ed Wood was directed by Scott Derrickson." in context_allow
    assert "Unknown status summary." in context_allow
    assert "Empty status summary." in context_allow
    assert "Verified: Scott Derrickson directed Deliver Us from Evil." in context_allow
    assert "Fallback: Community comprises 2 entities with no direct relations." in context_allow


def test_provenance_preservation_on_accepted_claims() -> None:
    """Test G: Provenance preservation on accepted claims.
    Accepted claims must accurately preserve supporting_edge_ids and source_chunk_ids from graph edges.
    """
    g = _build_test_graph()
    member_vertices = [v for v in g.vs if v["entity_id"] in {"E004", "E002"}]
    community_edges = [
        e for e in g.es
        if e["source_entity_id"] in {"E004", "E002"}
        and e["target_entity_id"] in {"E004", "E002"}
    ]

    claim = GeneratedClaim(
        subject_id="E004",
        relation_type="DIRECTED",
        object_id="E002",
        supporting_edge_ids=["REL_003"],
    )

    verified, rejected_count = CommunitySummarizer.validate_claims(
        [claim],
        member_vertices,
        community_edges,
    )

    assert rejected_count == 0
    assert len(verified) == 1
    v = verified[0]
    assert v.supporting_edge_ids == ["REL_003"]
    assert "C_D006" in v.source_chunk_ids
    assert v.subject_name == "Scott Derrickson"
    assert v.object_name == "Deliver Us from Evil"

    # End-to-end report generation preserves claims and supporting chunk IDs
    def mock_gen(prompt: str) -> str:
        return json.dumps({
            "claims": [
                {
                    "subject_id": "E004",
                    "relation_type": "DIRECTED",
                    "object_id": "E002",
                    "supporting_edge_ids": ["REL_003"],
                }
            ]
        })

    client = OllamaLLMClient(custom_generator=mock_gen)
    summarizer = CommunitySummarizer(llm_client=client)
    comms = [{"community_id": 42, "member_entity_ids": ["E004", "E002"], "partition_id": "p42"}]
    reports = summarizer.summarize_communities(comms, g)

    assert len(reports) == 1
    rep = reports[0]
    assert rep.verification_status == "VERIFIED"
    assert len(rep.verified_claims) == 1
    assert rep.verified_claims[0].supporting_edge_ids == ["REL_003"]
    assert "C_D006" in rep.supporting_chunk_ids
    assert "Scott Derrickson (E004) -[DIRECTED]-> Deliver Us from Evil (E002) [Edges: REL_003]" in rep.summary_text
