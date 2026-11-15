"""Unit and regression tests for Milestone 2 defect fixes (A, B, C, D, E)."""

from pathlib import Path
import json
from unittest.mock import MagicMock
import pytest
import igraph

from src.core.configuration import EngineConfig, ResolutionConfig, RetrievalConfig, ValidationConfig, load_engine_config
from src.core.models import (
    CanonicalEntity,
    Chunk,
    EntityMention,
    Evidence,
    Polarity,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    RetrievalResult,
    SeedEntitySpec,
    ValidationStatus,
)
from src.core.ontology import DomainOntology, RelationConstraint
from src.extraction.entity_extractor import OllamaEntityExtractor
from src.extraction.relation_extractor import OllamaRelationshipExtractor
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import AnswerGenerationOutput, OllamaAnswerGenerator
from src.graph.graph_validator import GraphValidator
from src.resolution.entity_resolver import ConservativeEntityResolver
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult


@pytest.fixture
def academic_ontology() -> DomainOntology:
    return DomainOntology(
        domain_name="academic_test",
        version="1.1.0",
        entity_types={
            "PERSON": "Researchers, students, professors",
            "UNIVERSITY": "Universities and academic institutions",
            "PROJECT": "Research projects and initiatives",
            "ORGANIZATION": "Organizations and agencies",
            "TECHNOLOGY": "Technical methods and models",
        },
        relationship_types={
            "WORKS_UNDER": RelationConstraint(
                name="WORKS_UNDER",
                allowed_source_types=["PERSON"],
                allowed_target_types=["PERSON"],
            ),
            "STUDIES_AT": RelationConstraint(
                name="STUDIES_AT",
                allowed_source_types=["PERSON"],
                allowed_target_types=["UNIVERSITY", "ORGANIZATION"],
            ),
            "SUPERVISES": RelationConstraint(
                name="SUPERVISES",
                allowed_source_types=["PERSON"],
                allowed_target_types=["PERSON", "PROJECT"],
            ),
            "USES": RelationConstraint(
                name="USES",
                allowed_source_types=["PROJECT", "PERSON"],
                allowed_target_types=["TECHNOLOGY"],
            ),
        },
        generic_descriptors={
            "PROJECT": ["project", "program", "programme", "initiative"]
        },
    )


# ---------------------------------------------------------------------------
# Fix A(i): Endpoint-in-evidence verification and pronoun rejection detection
# ---------------------------------------------------------------------------
def test_fix_a_endpoint_not_in_evidence_rejection(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Prior to joining the faculty, she was a postdoctoral fellow at Stanford University.",
        char_count=84,
        content_hash="h1",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="Dr. Elena Varghese",
            normalized_name="dr. elena varghese",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="Aerosol Research Group",
            normalized_name="aerosol research group",
            entity_type="PROJECT",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
    ]

    mock_llm_output = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "Prior to joining the faculty, she was a postdoctoral fellow at Stanford University.",
                "polarity": "POSITIVE",
                "confidence": 0.9,
            }
        ]
    }
    mock_client = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output))
    extractor = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=mock_client,
        validation_config=ValidationConfig(endpoint_in_evidence_enabled=True),
    )

    results = extractor.extract_relationships(chunk, mentions)
    assert len(results) == 1
    rel = results[0]
    # M002 is completely absent from the evidence quote
    assert rel.validation_status == ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE
    # Quote starts with "Prior...", not pronoun
    assert not rel.quote_starts_with_pronoun


def test_fix_a_pronoun_rejection_flag(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="He works under Professor Dhillon at the institute.",
        char_count=50,
        content_hash="h1",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="Tomas Lindqvist",
            normalized_name="tomas lindqvist",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="Dhillon",
            normalized_name="dhillon",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
    ]

    mock_llm_output = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "WORKS_UNDER",
                "evidence_quote": "He works under Professor Dhillon at the institute.",
                "polarity": "POSITIVE",
                "confidence": 0.9,
            }
        ]
    }
    mock_client = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output))
    extractor = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=mock_client,
        validation_config=ValidationConfig(endpoint_in_evidence_enabled=True),
    )

    results = extractor.extract_relationships(chunk, mentions)
    assert len(results) == 1
    rel = results[0]
    # Source Tomas Lindqvist was referred to via pronoun "He", not named in quote
    assert rel.validation_status == ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE
    assert rel.quote_starts_with_pronoun is True


# ---------------------------------------------------------------------------
# Fix A(ii): Configurable negation and uncertainty filtering
# ---------------------------------------------------------------------------
def test_fix_a_uncertainty_and_negation_cues(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Elena Varghese possibly supervises SmokeWatch without formal grant approval.",
        char_count=77,
        content_hash="h1",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="Elena Varghese",
            normalized_name="elena varghese",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="SmokeWatch",
            normalized_name="smokewatch",
            entity_type="PROJECT",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
    ]

    # 1. Uncertainty cue "possibly"
    mock_llm_output_uncertain = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "Elena Varghese possibly supervises SmokeWatch",
                "polarity": "POSITIVE",
                "confidence": 0.8,
            }
        ]
    }
    client1 = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output_uncertain))
    extractor1 = OllamaRelationshipExtractor(ontology=academic_ontology, llm_client=client1)
    res1 = extractor1.extract_relationships(chunk, mentions)
    assert res1[0].validation_status == ValidationStatus.REJECTED_UNCERTAIN

    # 2. Negation cue "without"
    mock_llm_output_neg = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "SmokeWatch without formal grant approval",
                "polarity": "POSITIVE",
                "confidence": 0.8,
            }
        ]
    }
    client2 = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output_neg))
    extractor2 = OllamaRelationshipExtractor(ontology=academic_ontology, llm_client=client2)
    # Both endpoints in evidence: let's test quote containing both with negation
    mock_llm_output_neg2 = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "Elena Varghese possibly supervises SmokeWatch without formal grant approval.",
                "polarity": "POSITIVE",
                "confidence": 0.8,
            }
        ]
    }
    client3 = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output_neg2))
    # Uncertainty takes precedence or negation:
    extractor3 = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=client3,
        validation_config=ValidationConfig(uncertainty_cues=[]),  # disable uncertainty to test negation cue
    )
    res3 = extractor3.extract_relationships(chunk, mentions)
    assert res3[0].validation_status == ValidationStatus.REJECTED_NEGATION


# ---------------------------------------------------------------------------
# Fix A(iii): Deterministic sentence quote repair
# ---------------------------------------------------------------------------
def test_fix_a_quote_repair_behavior(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="SmokeWatch is a research initiative. SmokeWatch uses satellite imagery and a gradient boosting model to forecast air quality. Funding is ongoing.",
        char_count=146,
        content_hash="h1",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="SmokeWatch",
            normalized_name="smokewatch",
            entity_type="PROJECT",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="gradient boosting model",
            normalized_name="gradient boosting model",
            entity_type="TECHNOLOGY",
            source_chunk_id="C001",
            source_doc_id="D001",
            page_numbers=[1],
        ),
    ]

    # Non-verbatim quote generated by LLM (omitted "satellite imagery and ")
    mock_llm_output = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "USES",
                "evidence_quote": "SmokeWatch uses a gradient boosting model",
                "polarity": "POSITIVE",
                "confidence": 0.95,
            }
        ]
    }
    client = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output))

    # 1. With quote_repair_enabled = False -> Rejected
    extractor_disabled = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=client,
        validation_config=ValidationConfig(quote_repair_enabled=False),
    )
    res_disabled = extractor_disabled.extract_relationships(chunk, mentions)
    assert res_disabled[0].validation_status == ValidationStatus.REJECTED_QUOTE_MISMATCH
    assert res_disabled[0].repaired_quote is False

    # 2. With quote_repair_enabled = True -> Repaired and Validated
    extractor_enabled = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=client,
        validation_config=ValidationConfig(quote_repair_enabled=True),
    )
    res_enabled = extractor_enabled.extract_relationships(chunk, mentions)
    assert res_enabled[0].validation_status == ValidationStatus.VALIDATED
    assert res_enabled[0].repaired_quote is True
    assert res_enabled[0].evidence_quotes == [
        "SmokeWatch uses satellite imagery and a gradient boosting model to forecast air quality."
    ]


# ---------------------------------------------------------------------------
# Fix B: Prompt versions and manifest recording
# ---------------------------------------------------------------------------
def test_fix_b_prompt_versions_and_manifest(tmp_path: Path, academic_ontology: DomainOntology):
    cfg = load_engine_config(Path("configs/domains/academic.yaml"))
    assert cfg.prompt_versions.relationship_extraction == "v1.1"
    assert cfg.prompt_versions.entity_extraction == "v1.1"
    assert cfg.prompt_versions.answer_generation == "v1.0"

    manifest_file = tmp_path / "manifest.json"
    g = igraph.Graph(directed=True)
    report = GraphValidator.validate(
        graph=g,
        entities=[],
        relationships=[],
        manifest_path=manifest_file,
        prompt_versions=cfg.prompt_versions.model_dump(),
    )
    assert "prompt_versions" in report
    assert report["prompt_versions"]["relationship_extraction"] == "v1.1"
    assert report["prompt_versions"]["entity_extraction"] == "v1.1"


# ---------------------------------------------------------------------------
# Fix C: Rules C1 and C2
# ---------------------------------------------------------------------------
def test_fix_c1_non_person_generic_descriptor_normalization(academic_ontology: DomainOntology):
    resolver = ConservativeEntityResolver(
        ontology=academic_ontology,
        resolution_config=ResolutionConfig(enable_descriptor_normalization=True),
    )

    m1 = EntityMention(
        kb_id="k", mention_id="M1", surface_name="SmokeWatch project",
        normalized_name="smokewatch project", entity_type="PROJECT",
        source_chunk_id="C1", source_doc_id="D1",
    )
    m2 = EntityMention(
        kb_id="k", mention_id="M2", surface_name="SmokeWatch",
        normalized_name="smokewatch", entity_type="PROJECT",
        source_chunk_id="C2", source_doc_id="D2",
    )

    canonicals, mapping = resolver.resolve_mentions([m1, m2])
    # Should resolve to a single canonical entity
    assert len(canonicals) == 1
    assert mapping["M1"] == mapping["M2"]
    assert canonicals[0].name in ("SmokeWatch project", "SmokeWatch")


def test_fix_c2_person_single_token_surname_merge(academic_ontology: DomainOntology):
    resolver = ConservativeEntityResolver(
        ontology=academic_ontology,
        resolution_config=ResolutionConfig(enable_single_token_person_merge=True),
    )

    m1 = EntityMention(
        kb_id="k", mention_id="M1", surface_name="Professor Leela Varghese",
        normalized_name="professor leela varghese", entity_type="PERSON",
        source_chunk_id="C1", source_doc_id="D1",
    )
    m2 = EntityMention(
        kb_id="k", mention_id="M2", surface_name="Varghese",
        normalized_name="varghese", entity_type="PERSON",
        source_chunk_id="C2", source_doc_id="D2",
    )

    canonicals, mapping = resolver.resolve_mentions([m1, m2])
    assert len(canonicals) == 1
    assert mapping["M1"] == mapping["M2"]
    assert canonicals[0].name == "Professor Leela Varghese"


def test_fix_c2_person_ambiguous_surname_blocked(academic_ontology: DomainOntology):
    resolver = ConservativeEntityResolver(
        ontology=academic_ontology,
        resolution_config=ResolutionConfig(enable_single_token_person_merge=True),
    )

    m1 = EntityMention(
        kb_id="k", mention_id="M1", surface_name="Giuseppe Verdi",
        normalized_name="giuseppe verdi", entity_type="PERSON",
        source_chunk_id="C1", source_doc_id="D1",
    )
    m2 = EntityMention(
        kb_id="k", mention_id="M2", surface_name="Marco Verdi",
        normalized_name="marco verdi", entity_type="PERSON",
        source_chunk_id="C2", source_doc_id="D2",
    )
    m3 = EntityMention(
        kb_id="k", mention_id="M3", surface_name="Verdi",
        normalized_name="verdi", entity_type="PERSON",
        source_chunk_id="C3", source_doc_id="D3",
    )

    canonicals, mapping = resolver.resolve_mentions([m1, m2, m3])
    # Ambiguous surname: M3 must NOT merge into either M1 or M2!
    assert len(canonicals) == 3
    assert mapping["M3"] != mapping["M1"]
    assert mapping["M3"] != mapping["M2"]


# ---------------------------------------------------------------------------
# Fix D: Single seed resolution per query
# ---------------------------------------------------------------------------
def test_fix_d_single_seed_resolution_reuse():
    mock_seed_retriever = MagicMock(spec=SeedEntityRetriever)
    mock_seed_res = SeedResolutionResult(
        resolved_entity_ids=["E001"],
        ambiguous_seeds={},
        unresolved_seeds=[],
        resolution_diagnostics=[],
    )
    mock_seed_retriever.resolve_seeds.return_value = mock_seed_res

    plan = QueryPlan(
        question="Who works under Elena Varghese?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Elena Varghese", expected_type="PERSON")],
        relationship_path=["WORKS_UNDER"],
        target_type="PERSON",
    )

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E001", entity_id="E001")

    retriever = GraphEvidenceRetriever(
        doc_store=MagicMock(),
        seed_retriever=mock_seed_retriever,
        graph=g,
    )

    # 1. When seed_resolution_result is passed explicitly, resolve_seeds is NOT called
    retriever.retrieve(
        question=plan.question,
        plan=plan,
        kb_id="test_kb",
        seed_resolution_result=mock_seed_res,
    )
    assert mock_seed_retriever.resolve_seeds.call_count == 0

    # 2. When seed_resolution_result is None, resolve_seeds is called once
    retriever.retrieve(
        question=plan.question,
        plan=plan,
        kb_id="test_kb",
        seed_resolution_result=None,
    )
    assert mock_seed_retriever.resolve_seeds.call_count == 1


# ---------------------------------------------------------------------------
# Fix E: Context builder refinement and generation prompt
# ---------------------------------------------------------------------------
def test_fix_e_context_builder_modes():
    ev = Evidence(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        text="Arjun Menon studies at Kestrel Institute of Technology.",
        page_numbers=[1],
        routes=[ProvenanceRoute.GRAPH],
        supporting_graph_paths=[
            "GRAPH_SEED_PROVENANCE: E001 (Arjun Menon) [spec: Arjun Menon, method: EXACT]",
            "Arjun Menon -[STUDIES_AT]-> Kestrel Institute of Technology",
        ],
        rrf_score=0.9,
    )
    ret_res = RetrievalResult(
        kb_id="test_kb",
        question="Where does Arjun Menon study?",
        query_plan=QueryPlan(question="Where does Arjun Menon study?"),
        evidence_list=[ev],
    )

    # Mode False (legacy)
    cb_legacy = ContextBuilder(graph_context_mode=False)
    ctx_legacy = cb_legacy.build_context(ret_res)
    assert "### Supporting Graph Traversal Paths" in ctx_legacy
    assert "GRAPH_SEED_PROVENANCE:" in ctx_legacy

    # Mode True (Fix E)
    cb_refined = ContextBuilder(graph_context_mode=True)
    ctx_refined = cb_refined.build_context(ret_res)
    assert "### Extracted Graph Relationships (Incomplete Extracted Facts)" in ctx_refined
    assert "GRAPH_SEED_PROVENANCE:" not in ctx_refined
    assert "- [Chunk: C001] Arjun Menon -[STUDIES_AT]-> Kestrel Institute of Technology" in ctx_refined
    assert "A fact not present in the graph does NOT mean it is false" in ctx_refined

    # Mode "none"
    cb_none = ContextBuilder(graph_context_mode="none")
    ctx_none = cb_none.build_context(ret_res)
    assert "### Supporting Graph Traversal Paths" not in ctx_none
    assert "### Extracted Graph Relationships" not in ctx_none
    assert "GRAPH_SEED_PROVENANCE:" not in ctx_none
    assert "-[" not in ctx_none
    assert "### [Chunk: C001" in ctx_none


def test_fix_e_graph_context_mode_none_leaks_no_graph_text():
    from unittest.mock import MagicMock
    from src.core.models import CommunityReport
    from src.generation.answer_generator import OllamaAnswerGenerator

    ev = Evidence(
        kb_id="test_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        text="Arjun Menon studies at Kestrel Institute of Technology.",
        page_numbers=[1],
        routes=[ProvenanceRoute.GRAPH],
        supporting_graph_paths=[
            "GRAPH_SEED_PROVENANCE: E001 (Arjun Menon) [spec: Arjun Menon, method: EXACT]",
            "Arjun Menon -[STUDIES_AT]-> Kestrel Institute of Technology",
            "EXTRA_GRAPH_FACT: Kestrel Institute of Technology -[LOCATED_IN]-> California",
        ],
        rrf_score=0.9,
    )
    comm = CommunityReport(
        kb_id="test_kb",
        report_id="R001",
        community_id=0,
        partition_id="part_0",
        member_entity_ids=["E001"],
        supporting_chunk_ids=["C001"],
        summary_text="Graph community summary about Kestrel.",
        model_tag="test_model",
        prompt_version="v1.0",
        verification_status="VERIFIED",
    )
    ret_res = RetrievalResult(
        kb_id="test_kb",
        question="Where does Arjun Menon study?",
        query_plan=QueryPlan(question="Where does Arjun Menon study?"),
        evidence_list=[ev],
        community_reports=[comm],
    )

    cb_none = ContextBuilder(graph_context_mode="none")
    ctx_none = cb_none.build_context(ret_res)

    # Must NOT contain graph headers
    assert "### Supporting Graph Traversal Paths" not in ctx_none
    assert "### Extracted Graph Relationships" not in ctx_none
    assert "### Community Summaries" not in ctx_none

    # Must NOT leak graph paths, edges, or bookkeeping
    assert "GRAPH_SEED_PROVENANCE" not in ctx_none
    assert "-[" not in ctx_none
    assert "]->" not in ctx_none
    assert "STUDIES_AT" not in ctx_none
    assert "EXTRA_GRAPH_FACT" not in ctx_none
    assert "Graph community summary" not in ctx_none

    # Must contain the chunk text and header
    assert "### [Chunk: C001" in ctx_none
    assert "Arjun Menon studies at Kestrel Institute of Technology." in ctx_none

    # Verify generated prompt via OllamaAnswerGenerator
    from src.generation.answer_generator import AnswerGenerationOutput
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = AnswerGenerationOutput(
        answer="Arjun Menon studies at Kestrel [C001].",
        canonical_answer="Kestrel Institute of Technology",
        abstained=False,
        cited_chunk_ids=["C001"],
    )
    gen = OllamaAnswerGenerator(llm_client=mock_llm, context_builder=cb_none)
    gen.generate_structured(ret_res)
    prompt = mock_llm.generate_json.call_args[0][0]

    # Prompt must NOT leak any graph section or graph relationship edge
    assert "### Supporting Graph Traversal Paths" not in prompt
    assert "### Extracted Graph Relationships" not in prompt
    assert "### Community Summaries" not in prompt
    assert "GRAPH_SEED_PROVENANCE" not in prompt
    assert "-[" not in prompt
    assert "]->" not in prompt
    assert "STUDIES_AT" not in prompt
    assert "EXTRA_GRAPH_FACT" not in prompt
    assert "Graph community summary" not in prompt


# ---------------------------------------------------------------------------
# Part 2 / Item 1: analyze_runs Error Classification Tests
# ---------------------------------------------------------------------------
def test_analyze_runs_error_classification():
    from experiments.synthetic_academic.analyze_runs import evaluate_run

    questions = {
        "Q01": {
            "id": "Q01",
            "category": "factual",
            "answerable": True,
            "expected_answer_contains": ["expected_term"],
            "required_docs": ["D001"],
        },
        "Q02": {
            "id": "Q02",
            "category": "unanswerable",
            "answerable": False,
            "expected_answer_contains": [],
            "required_docs": [],
        },
    }
    results_with_error = [
        {
            "question_id": "Q01",
            "answer": "Error generating answer: JSON validation failed",
            "abstained": True,
            "evidence_list": [{"filename": "D1_students.txt", "chunk_id": "D001_C0000"}],
            "cited_chunk_ids": [],
        },
        {
            "question_id": "Q02",
            "answer": "Error generating answer: timeout",
            "abstained": True,
            "evidence_list": [],
            "cited_chunk_ids": [],
        },
    ]
    summary = evaluate_run(results_with_error, questions)
    assert summary["error_count"] == 2
    assert summary["correct_count"] == 0
    assert summary["answerable_correct"] == 0
    assert summary["abstention_correct"] == 0


# ---------------------------------------------------------------------------
# Part 2 / Item 2: Comparative Reasoning Mode Tests
# ---------------------------------------------------------------------------
def test_comparative_reasoning_mode_config_validation():
    rc_enforce = RetrievalConfig(comparative_reasoning_mode="enforce")
    assert rc_enforce.comparative_reasoning_mode == "enforce"

    rc_off = RetrievalConfig(comparative_reasoning_mode="OFF")
    assert rc_off.comparative_reasoning_mode == "off"

    with pytest.raises(ValueError):
        RetrievalConfig(comparative_reasoning_mode="unsupported_mode")


def test_comparative_reasoning_mode_off_passes_llm_answer():
    from src.generation.answer_generator import CandidateAttributeFact
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = AnswerGenerationOutput(
        answer="CityBreath was deployed first [D001_C0000].",
        canonical_answer="CityBreath",
        abstained=False,
        cited_chunk_ids=["D001_C0000"],
        extracted_attributes=[
            CandidateAttributeFact(
                entity_name="CityBreath",
                attribute_name="deployment_date",
                extracted_value="2021",
                evidence_quote="CityBreath launched in 2021",
                chunk_id="D001_C0000",
            )
        ],
    )
    plan = QueryPlan(
        question="Which system was deployed first, CityBreath or UrbanPulse?",
        intent=QueryIntent.LOCAL,
        seed_entities=[
            SeedEntitySpec(name="CityBreath"),
            SeedEntitySpec(name="UrbanPulse"),
        ],
        relationship_path=["deployment_date"],
    )
    ev = Evidence(
        kb_id="test_kb",
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="d1.txt",
        text="CityBreath launched in 2021.",
        routes=[ProvenanceRoute.VECTOR],
    )
    ret_res = RetrievalResult(
        kb_id="test_kb",
        question=plan.question,
        query_plan=plan,
        evidence_list=[ev],
    )

    gen_off = OllamaAnswerGenerator(
        llm_client=mock_llm,
        comparative_reasoning_mode="off",
    )
    output = gen_off.generate_structured(ret_res)
    assert not output.abstained
    assert output.canonical_answer == "CityBreath"
    assert "CityBreath was deployed first" in output.answer


def test_comparative_reasoning_mode_enforce_applies_deterministic_evaluation():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = AnswerGenerationOutput(
        answer="CityBreath was deployed first [D001_C0000].",
        canonical_answer="CityBreath",
        abstained=False,
        cited_chunk_ids=["D001_C0000"],
    )
    plan = QueryPlan(
        question="Which system was deployed first, CityBreath or UrbanPulse?",
        intent=QueryIntent.LOCAL,
        seed_entities=[
            SeedEntitySpec(name="CityBreath"),
            SeedEntitySpec(name="UrbanPulse"),
        ],
        relationship_path=["deployment_date"],
    )
    ev = Evidence(
        kb_id="test_kb",
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="d1.txt",
        text="CityBreath launched in 2021.",
        routes=[ProvenanceRoute.VECTOR],
    )
    ret_res = RetrievalResult(
        kb_id="test_kb",
        question=plan.question,
        query_plan=plan,
        evidence_list=[ev],
    )

    gen_enforce = OllamaAnswerGenerator(
        llm_client=mock_llm,
        comparative_reasoning_mode="enforce",
    )
    output = gen_enforce.generate_structured(ret_res)
    assert output.abstained is True
    assert "Neither" in output.answer or "unable" in output.answer.lower()


# ---------------------------------------------------------------------------
# Part 2 / Item 3: Answer Generation Prompt Version Selection Tests
# ---------------------------------------------------------------------------
def test_answer_generation_prompt_version_selection():
    mock_llm = MagicMock()
    gen_v1_0 = OllamaAnswerGenerator(llm_client=mock_llm, prompt_version="v1.0")
    assert "Chunk text is the primary authority" not in gen_v1_0.prompt_template
    assert "Do NOT abstain merely because the graph section lacks a relationship edge" not in gen_v1_0.prompt_template

    gen_v1_1 = OllamaAnswerGenerator(llm_client=mock_llm, prompt_version="v1.1")
    assert "Chunk text is the primary authority" in gen_v1_1.prompt_template
    assert "Do NOT abstain merely because the graph section lacks a relationship edge" in gen_v1_1.prompt_template


# ---------------------------------------------------------------------------
# Part 2 / Item 4: Fix A(i) Amendment: Flexible Endpoint Matching
# ---------------------------------------------------------------------------
def test_fix_a_flexible_endpoint_matching_honorific_and_surname(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C002",
        doc_id="D002",
        filename="d2.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Prof. Varghese advises PhD candidate Daniel Kim. Daniel Kim is co-advised by Professor Dhillon in the Distributed Systems Group.",
        char_count=130,
        content_hash="h2",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="Leela Varghese",
            normalized_name="leela varghese",
            entity_type="PERSON",
            source_chunk_id="C002",
            source_doc_id="D002",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="Daniel Kim",
            normalized_name="daniel kim",
            entity_type="PERSON",
            source_chunk_id="C002",
            source_doc_id="D002",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M003",
            surface_name="Ravi Dhillon",
            normalized_name="ravi dhillon",
            entity_type="PERSON",
            source_chunk_id="C002",
            source_doc_id="D002",
            page_numbers=[1],
        ),
    ]

    mock_llm_output = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "Prof. Varghese advises PhD candidate Daniel Kim.",
                "polarity": "POSITIVE",
                "confidence": 0.95,
            },
            {
                "source_mention_id": "M003",
                "target_mention_id": "M002",
                "relation_type": "SUPERVISES",
                "evidence_quote": "Daniel Kim is co-advised by Professor Dhillon in the Distributed Systems Group.",
                "polarity": "POSITIVE",
                "confidence": 0.95,
            },
        ]
    }
    mock_client = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output))
    extractor = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=mock_client,
        validation_config=ValidationConfig(endpoint_in_evidence_enabled=True),
    )

    results = extractor.extract_relationships(chunk, mentions)
    assert len(results) == 2
    assert results[0].validation_status == ValidationStatus.VALIDATED
    assert results[1].validation_status == ValidationStatus.VALIDATED


def test_fix_a_non_unique_surname_fails_on_surname_alone(academic_ontology: DomainOntology):
    chunk = Chunk(
        kb_id="test_kb",
        chunk_id="C003",
        doc_id="D003",
        filename="d3.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Researcher Kim met with Tomas Lindqvist.",
        char_count=40,
        content_hash="h3",
    )
    mentions = [
        EntityMention(
            kb_id="test_kb",
            mention_id="M001",
            surface_name="Daniel Kim",
            normalized_name="daniel kim",
            entity_type="PERSON",
            source_chunk_id="C003",
            source_doc_id="D003",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M002",
            surface_name="Min-jun Kim",
            normalized_name="min-jun kim",
            entity_type="PERSON",
            source_chunk_id="C003",
            source_doc_id="D003",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="test_kb",
            mention_id="M003",
            surface_name="Tomas Lindqvist",
            normalized_name="tomas lindqvist",
            entity_type="PERSON",
            source_chunk_id="C003",
            source_doc_id="D003",
            page_numbers=[1],
        ),
    ]

    mock_llm_output = {
        "relationships": [
            {
                "source_mention_id": "M001",
                "target_mention_id": "M003",
                "relation_type": "COLLABORATES_WITH",
                "evidence_quote": "Researcher Kim met with Tomas Lindqvist.",
                "polarity": "POSITIVE",
                "confidence": 0.9,
            }
        ]
    }
    mock_client = OllamaLLMClient(custom_generator=lambda p: json.dumps(mock_llm_output))
    extractor = OllamaRelationshipExtractor(
        ontology=academic_ontology,
        llm_client=mock_client,
        validation_config=ValidationConfig(endpoint_in_evidence_enabled=True),
    )

    results = extractor.extract_relationships(chunk, mentions)
    assert len(results) == 1
    assert results[0].validation_status == ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE


def test_abstention_with_null_extracted_attributes_validates_in_off_mode():
    """Verify that an abstention output with null extracted_value/evidence_quote validates without ValidationError."""
    raw_json_str = json.dumps({
        "reasoning": "Context lacks salary information.",
        "extracted_attributes": [
            {
                "entity_name": "Professor Varghese",
                "entity_id": None,
                "attribute_name": "salary",
                "extracted_value": None,
                "evidence_quote": None,
                "chunk_id": None,
            }
        ],
        "canonical_answer": "",
        "answer": "I am unable to answer this question. The evidence does not contain salary information.",
        "abstained": True,
        "cited_chunk_ids": [],
    })
    parsed = json.loads(raw_json_str)
    # This must validate without raising pydantic.ValidationError
    output = AnswerGenerationOutput.model_validate(parsed)
    assert output.abstained is True
    assert output.extracted_attributes[0].extracted_value in (None, "")


def test_generation_failure_sets_generation_error_and_flags_analyze_runs():
    """Verify that an LLM failure sets generation_error=True and is correctly flagged by analyze_runs."""
    from experiments.synthetic_academic.analyze_runs import evaluate_run

    mock_client = OllamaLLMClient(custom_generator=MagicMock(side_effect=RuntimeError("Ollama connection failed")))
    generator = OllamaAnswerGenerator(llm_client=mock_client, comparative_reasoning_mode="off")

    retrieval_result = RetrievalResult(
        kb_id="test_kb",
        question="What is the founding year?",
        query_plan=QueryPlan(question="What is the founding year?", intent=QueryIntent.LOCAL),
        evidence_list=[
            Evidence(
                kb_id="test_kb",
                chunk_id="D001_C0000",
                doc_id="D001",
                text="Some text about founding.",
                routes=[ProvenanceRoute.VECTOR],
            )
        ],
    )

    out = generator.generate_structured(retrieval_result)
    assert out.generation_error is True
    assert retrieval_result.generation_error is True

    record = {
        "question_id": "Q1",
        "question": "What is the founding year?",
        "answerable": True,
        "answer": out.answer,
        "abstained": out.abstained,
        "generation_error": retrieval_result.generation_error,
    }
    eval_res = evaluate_run([record], {"Q1": {"category": "test", "answerable": True}})
    assert eval_res["error_count"] == 1
    assert eval_res["per_question"][0]["is_error"] is True


def test_default_comparative_reasoning_mode_is_off_and_uniform_across_modes():
    """Verify that default comparative_reasoning_mode is 'off' and all 4 modes use it."""
    from unittest.mock import patch
    from src.engine import HybridGraphRAGEngine

    ret_cfg = RetrievalConfig()
    assert ret_cfg.comparative_reasoning_mode == "off"

    mock_llm = OllamaLLMClient(custom_generator=lambda p: json.dumps({
        "reasoning": "r", "answer": "a", "canonical_answer": "ca", "abstained": False, "cited_chunk_ids": []
    }))
    default_gen = OllamaAnswerGenerator(mock_llm)
    assert default_gen.comparative_reasoning_mode == "off"

    modes_seen = []
    def fake_init(self, llm_client, **kwargs):
        modes_seen.append(kwargs.get("comparative_reasoning_mode"))
        self.comparative_reasoning_mode = kwargs.get("comparative_reasoning_mode", "off")
        self.generate_structured = lambda ret_res: AnswerGenerationOutput(answer="ans", canonical_answer="c", abstained=False)

    with patch.object(OllamaAnswerGenerator, "__init__", fake_init):
        engine = MagicMock(spec=HybridGraphRAGEngine)
        engine.config = MagicMock()
        engine.config.retrieval = RetrievalConfig()
        engine.config.prompt_versions = MagicMock(answer_generation="v1.0")
        engine.config.resolution = MagicMock(similarity_candidate_threshold=0.82)
        engine.llm_client = mock_llm
        engine.kb_id = "test_kb"
        engine.get_or_load_graph.return_value = igraph.Graph(directed=True)
        engine.doc_store = MagicMock()
        engine.vector_store = MagicMock()
        engine.embedding_service = MagicMock()

        for mode in ["hybrid", "vector_only", "graph_only", "community_aware"]:
            with patch("src.engine.SeedEntityRetriever"), \
                 patch("src.engine.GraphEvidenceRetriever") as mock_ge, \
                 patch("src.engine.VectorEvidenceRetriever") as mock_ve, \
                 patch("src.engine.CommunityRetriever"), \
                 patch("src.engine.ContextBuilder"), \
                 patch("src.engine.OllamaQueryPlanner") as mock_qp, \
                 patch("src.engine.HybridRetrieverOrchestrator") as mock_ho:
                mock_qp.return_value.plan.return_value = QueryPlan(question="test question", intent=QueryIntent.LOCAL)
                mock_ge.return_value.retrieve.return_value = []
                mock_ge.return_value.last_diagnostics = {}
                mock_ve.return_value.retrieve.return_value = []
                mock_ho.return_value.retrieve_hybrid.return_value = RetrievalResult(
                    kb_id="test_kb",
                    question="test question",
                    query_plan=QueryPlan(question="test question", intent=QueryIntent.LOCAL),
                )
                HybridGraphRAGEngine.query(engine, "test question", mode=mode)

    assert len(modes_seen) == 4
    assert all(m == "off" for m in modes_seen)


