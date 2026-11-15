"""Regression tests for reliable, evidence-grounded answer generation in Hybrid GraphRAG.

Tests:
Test A: Two entities with the same nationality -> yes.
Test B: Two entities with different nationalities -> no.
Test C: Two entities born in different cities but sharing nationality -> yes.
Test D: Different birthplace must not be substituted for nationality.
Test E: Conrad Brooks' Baltimore birthplace must not be attributed to Ed Wood.
Test F: A missing attribute for one entity must cause abstention.
Test G: Synthetic citation IDs such as C001 must not survive as validated citations.
Test H: A real supporting chunk ID must be preserved.
Test I: A citation to a chunk that does not support the extracted fact must be rejected.
Test J: Canonical answer and final answer must not contradict each other.
Test K: Non-comparative factual questions must continue to work.
Test L: The implementation must work with different entity names and relationship types rather than hardcoded examples.
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.core.models import (
    Evidence,
    ProvenanceRoute,
    QueryPlan,
    RetrievalResult,
    SeedEntitySpec,
)
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import (
    AnswerGenerationOutput,
    CandidateAttributeFact,
    ComparativeReasoningEngine,
    FactValidator,
    OllamaAnswerGenerator,
)


def _make_chunk(chunk_id: str, doc_id: str, text: str) -> Evidence:
    return Evidence(
        kb_id="test_kb",
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        routes=[ProvenanceRoute.VECTOR, ProvenanceRoute.GRAPH],
        page_numbers=[1],
    )


def test_a_same_nationality_returns_yes() -> None:
    """Test A: Two entities with the same nationality -> yes."""
    c1 = _make_chunk("D002_C0000", "D002", "Scott Derrickson (born July 16, 1966) is an American director and screenwriter.")
    c2 = _make_chunk("D005_C0000", "D005", "Edward Davis Wood Jr. was an American filmmaker, actor, writer, and director.")

    query_plan = QueryPlan(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Ed Wood", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Both Scott Derrickson and Ed Wood are identified as American.",
        "extracted_attributes": [
            {
                "entity_name": "Scott Derrickson",
                "attribute_name": "NATIONALITY",
                "extracted_value": "American",
                "evidence_quote": "Scott Derrickson (born July 16, 1966) is an American director and screenwriter.",
                "chunk_id": "D002_C0000",
            },
            {
                "entity_name": "Ed Wood",
                "attribute_name": "NATIONALITY",
                "extracted_value": "American",
                "evidence_quote": "Edward Davis Wood Jr. was an American filmmaker, actor, writer, and director.",
                "chunk_id": "D005_C0000",
            },
        ],
        "canonical_answer": "yes",
        "answer": "They were both American.",
        "abstained": False,
        "cited_chunk_ids": ["D002_C0000", "D005_C0000"],
    }))

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(result)

    assert output.canonical_answer == "yes"
    assert output.abstained is False
    assert "same nationality" in output.answer
    assert set(output.cited_chunk_ids) == {"D002_C0000", "D005_C0000"}


def test_b_different_nationalities_returns_no() -> None:
    """Test B: Two entities with different nationalities -> no."""
    c1 = _make_chunk("D1_C1", "D1", "Christopher Nolan is a British-American filmmaker.")
    c2 = _make_chunk("D2_C1", "D2", "Hayao Miyazaki is a Japanese animator and director.")

    query_plan = QueryPlan(
        question="Were Christopher Nolan and Hayao Miyazaki of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Christopher Nolan", expected_type="PERSON"),
            SeedEntitySpec(name="Hayao Miyazaki", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Nolan is British while Miyazaki is Japanese.",
        "extracted_attributes": [
            {"entity_name": "Christopher Nolan", "attribute_name": "NATIONALITY", "extracted_value": "British", "evidence_quote": "Christopher Nolan is a British-American filmmaker.", "chunk_id": "D1_C1"},
            {"entity_name": "Hayao Miyazaki", "attribute_name": "NATIONALITY", "extracted_value": "Japanese", "evidence_quote": "Hayao Miyazaki is a Japanese animator and director.", "chunk_id": "D2_C1"},
        ],
        "canonical_answer": "no",
        "answer": "Nolan was British, while Miyazaki was Japanese.",
        "abstained": False,
    }))

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(result)

    assert output.canonical_answer == "no"
    assert output.abstained is False
    assert "not of the same nationality" in output.answer
    assert set(output.cited_chunk_ids) == {"D1_C1", "D2_C1"}


def test_c_different_birthplace_cities_same_nationality_returns_yes() -> None:
    """Test C: Two entities born in different cities but sharing nationality -> yes."""
    c1 = _make_chunk("C1", "D1", "Scott Derrickson was born in Denver, Colorado and is an American director.")
    c2 = _make_chunk("C2", "D2", "Ed Wood was born in Poughkeepsie, New York and was an American filmmaker.")

    query_plan = QueryPlan(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Ed Wood", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    # Even if LLM erroneously tried to suggest birthplaces, the comparative engine evaluates the requested NATIONALITY
    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Both are American.",
        "extracted_attributes": [
            {"entity_name": "Scott Derrickson", "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Scott Derrickson was born in Denver, Colorado and is an American director.", "chunk_id": "C1"},
            {"entity_name": "Ed Wood", "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Ed Wood was born in Poughkeepsie, New York and was an American filmmaker.", "chunk_id": "C2"},
        ],
        "canonical_answer": "yes",
        "answer": "Both are American.",
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert output.canonical_answer == "yes"
    assert "same nationality" in output.answer


def test_d_birthplace_not_substituted_for_nationality() -> None:
    """Test D: Different birthplace must not be substituted for nationality."""
    c1 = _make_chunk("C1", "D1", "Scott Derrickson was born in Denver, Colorado. He is an American director.")
    c2 = _make_chunk("C2", "D2", "Ed Wood was born in Poughkeepsie, New York. He was an American filmmaker.")

    # A candidate fact claiming a city name as nationality must be rejected by FactValidator
    fact_city = CandidateAttributeFact(
        entity_name="Scott Derrickson",
        attribute_name="NATIONALITY",
        extracted_value="Denver, Colorado",
        evidence_quote="Scott Derrickson was born in Denver, Colorado.",
        chunk_id="C1",
    )
    valid, reason = FactValidator.validate_fact(fact_city, [c1, c2], expected_attribute="NATIONALITY")
    assert valid is False
    assert "city/state, not a nationality" in reason


def test_e_conrad_brooks_birthplace_not_attributed_to_ed_wood() -> None:
    """Test E: Conrad Brooks' Baltimore birthplace must not be attributed to Ed Wood."""
    c_conrad = _make_chunk(
        "D009_C0000",
        "D009",
        "Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) was an American actor. "
        "He got his start in movies appearing in Ed Wood films such as Plan 9 from Outer Space.",
    )

    fact_misattributed = CandidateAttributeFact(
        entity_name="Ed Wood",
        attribute_name="BORN_IN",
        extracted_value="Baltimore, Maryland",
        evidence_quote="born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland",
        chunk_id="D009_C0000",
    )
    valid, reason = FactValidator.validate_fact(fact_misattributed, [c_conrad], expected_attribute="BORN_IN")
    assert valid is False
    assert "does not refer to target entity 'Ed Wood'" in reason


def test_f_missing_attribute_causes_clean_abstention() -> None:
    """Test F: A missing attribute for one entity must cause abstention."""
    c1 = _make_chunk("C1", "D1", "Scott Derrickson is an American director.")
    c2 = _make_chunk("C2", "D2", "Unknown Person is a mysterious director.")  # No nationality stated

    query_plan = QueryPlan(
        question="Were Scott Derrickson and Unknown Person of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Unknown Person", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Scott Derrickson is American but Unknown Person's nationality is not stated.",
        "extracted_attributes": [
            {"entity_name": "Scott Derrickson", "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Scott Derrickson is an American director.", "chunk_id": "C1"}
        ],
        "canonical_answer": "",
        "answer": "Unable to determine.",
        "abstained": True,
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert output.abstained is True
    assert output.canonical_answer == ""
    assert "unable to answer" in output.answer.lower()
    assert output.cited_chunk_ids == []


def test_g_synthetic_citation_ids_never_survive() -> None:
    """Test G: Synthetic citation IDs such as C001 must not survive as validated citations."""
    c_real = _make_chunk("REAL_CHUNK_001", "D001", "Scott Derrickson directed Doctor Strange.")

    query_plan = QueryPlan(question="What film did Scott Derrickson direct?")
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c_real],
    )

    # LLM emits synthetic placeholder IDs
    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Found film.",
        "canonical_answer": "Doctor Strange",
        "answer": "Scott Derrickson directed Doctor Strange [C001, D001, p.1].",
        "abstained": False,
        "cited_chunk_ids": ["C001", "C002"],
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert "C001" not in output.cited_chunk_ids
    assert "C002" not in output.cited_chunk_ids


def test_h_real_supporting_chunk_id_preserved() -> None:
    """Test H: A real supporting chunk ID must be preserved."""
    c_real = _make_chunk("REAL_CHUNK_999", "D001", "Scott Derrickson directed Doctor Strange.")

    query_plan = QueryPlan(question="What film did Scott Derrickson direct?")
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c_real],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Found film.",
        "canonical_answer": "Doctor Strange",
        "answer": "Scott Derrickson directed Doctor Strange [REAL_CHUNK_999].",
        "abstained": False,
        "cited_chunk_ids": ["REAL_CHUNK_999"],
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert output.cited_chunk_ids == ["REAL_CHUNK_999"]
    assert len(output.citation_references) == 1
    assert output.citation_references[0]["chunk_id"] == "REAL_CHUNK_999"


def test_i_citation_to_chunk_not_supporting_fact_rejected() -> None:
    """Test I: A citation to a chunk that does not support the extracted fact must be rejected."""
    c1 = _make_chunk("CHUNK_A", "D1", "Scott Derrickson was an American director.")
    c2 = _make_chunk("CHUNK_B", "D2", "A completely unrelated document about astrophysics.")

    # Candidate fact citing CHUNK_B with quote from CHUNK_A
    fact_mismatched = CandidateAttributeFact(
        entity_name="Scott Derrickson",
        attribute_name="NATIONALITY",
        extracted_value="American",
        evidence_quote="Scott Derrickson was an American director.",
        chunk_id="CHUNK_B",
    )
    valid, reason = FactValidator.validate_fact(fact_mismatched, [c1, c2], expected_attribute="NATIONALITY")
    assert valid is False
    assert "does not occur in chunk 'CHUNK_B'" in reason


def test_j_canonical_answer_and_final_answer_never_contradict() -> None:
    """Test J: Canonical answer and final answer must not contradict each other."""
    c1 = _make_chunk("C1", "D1", "Alice was an American researcher.")
    c2 = _make_chunk("C2", "D2", "Bob was a British researcher.")

    # In comparative mode:
    canonical, answer, abstained, cited_chunks, _ = ComparativeReasoningEngine.evaluate_comparison(
        "Alice",
        FactValidator.extract_entity_attribute_from_chunk(c1, "Alice", "NATIONALITY"),
        "Bob",
        FactValidator.extract_entity_attribute_from_chunk(c2, "Bob", "NATIONALITY"),
        "NATIONALITY",
    )
    # When canonical is 'no', answer must state they are NOT of the same nationality
    assert canonical == "no"
    assert "not of the same nationality" in answer
    assert "same nationality" not in answer.replace("not of the same nationality", "")


def test_k_non_comparative_factual_questions_continue_to_work() -> None:
    """Test K: Non-comparative factual questions must continue to work."""
    c = _make_chunk("CHUNK_001", "D001", "Doctor Strange was released in November 2016.")

    query_plan = QueryPlan(question="When was Doctor Strange released?")
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Doctor Strange was released in November 2016.",
        "canonical_answer": "November 2016",
        "answer": "Doctor Strange was released in November 2016 [CHUNK_001].",
        "abstained": False,
        "cited_chunk_ids": ["CHUNK_001"],
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert output.canonical_answer == "November 2016"
    assert "November 2016" in output.answer
    assert output.cited_chunk_ids == ["CHUNK_001"]
    assert output.abstained is False


def test_l_arbitrary_domain_and_relation_types() -> None:
    """Test L: The implementation must work with different entity names and relationship types rather than hardcoded examples."""
    c1 = _make_chunk("C_UNIV_1", "D1", "Dr. Sharma is a faculty member affiliated with IIT Delhi.")
    c2 = _make_chunk("C_UNIV_2", "D2", "Dr. Rao is a faculty member affiliated with IIT Delhi.")

    query_plan = QueryPlan(
        question="Did Dr. Sharma and Dr. Rao work at the same institution?",
        seed_entities=[
            SeedEntitySpec(name="Dr. Sharma", expected_type="PERSON"),
            SeedEntitySpec(name="Dr. Rao", expected_type="PERSON"),
        ],
        relationship_path=["WORKS_AT"],
    )
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Both Dr. Sharma and Dr. Rao are affiliated with IIT Delhi.",
        "extracted_attributes": [
            {"entity_name": "Dr. Sharma", "attribute_name": "WORKS_AT", "extracted_value": "IIT Delhi", "evidence_quote": "Dr. Sharma is a faculty member affiliated with IIT Delhi.", "chunk_id": "C_UNIV_1"},
            {"entity_name": "Dr. Rao", "attribute_name": "WORKS_AT", "extracted_value": "IIT Delhi", "evidence_quote": "Dr. Rao is a faculty member affiliated with IIT Delhi.", "chunk_id": "C_UNIV_2"},
        ],
        "canonical_answer": "yes",
        "answer": "Both worked at IIT Delhi.",
    }))

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    assert output.canonical_answer == "yes"
    assert "same works_at" in output.answer or "same institution" in output.answer or "same" in output.answer
    assert set(output.cited_chunk_ids) == {"C_UNIV_1", "C_UNIV_2"}


def test_m_explicit_question_attribute_priority_over_incorrect_relationship_path() -> None:
    """Explicit question attribute (NATIONALITY) takes priority over an incorrect QueryPlan relationship_path (BORN_IN)."""
    c1 = _make_chunk("D002_C0000", "D002", "Scott Derrickson (born July 16, 1966) is an American director and screenwriter.")
    c2 = _make_chunk("D005_C0000", "D005", "Edward Davis Wood Jr. was an American filmmaker, actor, writer, and director.")

    # Query planner generated BORN_IN as graph traversal hint, but the question explicitly asks for nationality
    query_plan = QueryPlan(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Ed Wood", expected_type="PERSON"),
        ],
        relationship_path=["BORN_IN"],
    )

    # 1. Verify target attribute resolution directly: explicit question cue overrides path
    target_attr = ComparativeReasoningEngine.extract_target_attribute(query_plan)
    assert target_attr == "NATIONALITY", f"Expected 'NATIONALITY' but got '{target_attr}'"

    # 2. Verify end-to-end answer generation correctly evaluates nationality, not birthplace
    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(custom_generator=lambda _p: json.dumps({
        "reasoning": "Both are identified as American.",
        "extracted_attributes": [
            {
                "entity_name": "Scott Derrickson",
                "attribute_name": "NATIONALITY",
                "extracted_value": "American",
                "evidence_quote": "Scott Derrickson (born July 16, 1966) is an American director and screenwriter.",
                "chunk_id": "D002_C0000",
            },
            {
                "entity_name": "Ed Wood",
                "attribute_name": "NATIONALITY",
                "extracted_value": "American",
                "evidence_quote": "Edward Davis Wood Jr. was an American filmmaker, actor, writer, and director.",
                "chunk_id": "D005_C0000",
            },
        ],
        "canonical_answer": "yes",
        "answer": "They were both American.",
        "abstained": False,
        "cited_chunk_ids": ["D002_C0000", "D005_C0000"],
    }))

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(result)

    assert output.canonical_answer == "yes"
    assert output.abstained is False
    assert "same nationality" in output.answer
    assert set(output.cited_chunk_ids) == {"D002_C0000", "D005_C0000"}
