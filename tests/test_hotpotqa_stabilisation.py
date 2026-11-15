"""Comprehensive regression test suite for HotpotQA answer generation stabilisation.

Tests:
1. Selection questions (returns entity name, not boolean yes/no).
2. Boolean comparisons (returns yes/no).
3. Opera composer versus composer with supporting evidence (returns yes).
4. Unsupported occupational equivalence (causes abstention).
5. Nullable entity IDs with valid provenance (accepted).
6. Missing or unsupported evidence still causing abstention.
7. Neighbourhood comparison (returns no for different neighbourhoods).
8. Correct neighbourhood-versus-city distinction (does not substitute city for neighbourhood).
9. Existing nationality comparisons (demonym and country normalization).
10. Existing bridge-question functionality (nullable entity_id with grounded facts).
11. Citation metric calculations (separate precision, coverage, and conditioned metrics).
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


def _make_chunk(chunk_id: str, doc_id: str, text: str, filename: str = "doc.txt") -> Evidence:
    return Evidence(
        kb_id="test_kb",
        chunk_id=chunk_id,
        doc_id=doc_id,
        filename=filename,
        text=text,
        routes=[ProvenanceRoute.VECTOR, ProvenanceRoute.GRAPH],
        page_numbers=[1],
    )


# ---------------------------------------------------------------------------
# Test 1: Selection questions
# ---------------------------------------------------------------------------
def test_1_selection_question_returns_entity_name() -> None:
    """Selection questions must return the selected entity name, not yes or no."""
    c1 = _make_chunk(
        "D002_C0000",
        "D002",
        "Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer.",
    )
    c2 = _make_chunk(
        "D007_C0000",
        "D007",
        "Robert Erskine Childers DSC was a British-born Irish writer and politician.",
    )

    query_plan = QueryPlan(
        question="Which writer was from England, Henry Roth or Robert Erskine Childers?",
        seed_entities=[
            SeedEntitySpec(name="Henry Roth", expected_type="PERSON"),
            SeedEntitySpec(name="Robert Erskine Childers", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )

    assert ComparativeReasoningEngine.is_selection_question(query_plan) is True
    assert ComparativeReasoningEngine.is_comparative_question(query_plan) is False

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    # Even if LLM erroneously emitted "no" or "yes", selection engine forces entity selection
    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Henry Roth was American, Robert Erskine Childers was British/English.",
            "extracted_attributes": [
                {
                    "entity_name": "Henry Roth",
                    "entity_id": None,
                    "attribute_name": "NATIONALITY",
                    "extracted_value": "American",
                    "evidence_quote": "Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer.",
                    "chunk_id": "D002_C0000",
                },
                {
                    "entity_name": "Robert Erskine Childers",
                    "entity_id": None,
                    "attribute_name": "NATIONALITY",
                    "extracted_value": "British",
                    "evidence_quote": "Robert Erskine Childers DSC was a British-born Irish writer and politician.",
                    "chunk_id": "D007_C0000",
                },
            ],
            "canonical_answer": "no",
            "answer": "Robert Erskine Childers was from England, not Henry Roth.",
            "abstained": False,
            "cited_chunk_ids": ["D002_C0000", "D007_C0000"],
        })
    )

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(result)

    assert output.canonical_answer in ("Robert Erskine Childers", "Robert Erskine Childers DSC")
    assert output.canonical_answer not in ("yes", "no")
    assert output.abstained is False
    assert set(output.cited_chunk_ids) == {"D002_C0000", "D007_C0000"}


# ---------------------------------------------------------------------------
# Test 2: Boolean comparisons
# ---------------------------------------------------------------------------
def test_2_boolean_comparison_returns_yes_or_no() -> None:
    """Boolean comparisons must return yes or no."""
    c1 = _make_chunk("D1_C1", "D1", "Scott Derrickson is an American director.")
    c2 = _make_chunk("D2_C1", "D2", "Ed Wood was an American filmmaker.")

    query_plan = QueryPlan(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Ed Wood", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )

    assert ComparativeReasoningEngine.is_selection_question(query_plan) is False
    assert ComparativeReasoningEngine.is_comparative_question(query_plan) is True

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Both are American.",
            "extracted_attributes": [
                {"entity_name": "Scott Derrickson", "entity_id": None, "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Scott Derrickson is an American director.", "chunk_id": "D1_C1"},
                {"entity_name": "Ed Wood", "entity_id": None, "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Ed Wood was an American filmmaker.", "chunk_id": "D2_C1"},
            ],
            "canonical_answer": "yes",
            "answer": "They were both American.",
            "abstained": False,
            "cited_chunk_ids": ["D1_C1", "D2_C1"],
        })
    )

    output = OllamaAnswerGenerator(llm).generate_structured(result)
    assert output.canonical_answer == "yes"
    assert output.abstained is False


# ---------------------------------------------------------------------------
# Test 3: Opera composer versus composer with supporting evidence
# ---------------------------------------------------------------------------
def test_3_opera_composer_with_supporting_evidence_returns_yes() -> None:
    """Opera composer vs Composer where evidence supports operas returns yes."""
    c1 = _make_chunk(
        "D004_C0000",
        "D004",
        "Giuseppe Fortunino Francesco Verdi was an Italian opera composer.",
    )
    c2 = _make_chunk(
        "D006_C0000",
        "D006",
        "Charles Louis Ambroise Thomas was a French composer, best known for his operas Mignon and Hamlet.",
    )

    query_plan = QueryPlan(
        question="Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?",
        seed_entities=[
            SeedEntitySpec(name="Giuseppe Verdi", expected_type="PERSON"),
            SeedEntitySpec(name="Ambroise Thomas", expected_type="PERSON"),
        ],
        relationship_path=["OCCUPATION"],
    )

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Verdi is an opera composer and Thomas is a composer best known for his operas.",
            "extracted_attributes": [
                {"entity_name": "Giuseppe Verdi", "entity_id": None, "attribute_name": "OCCUPATION", "extracted_value": "Opera composer", "evidence_quote": "Giuseppe Fortunino Francesco Verdi was an Italian opera composer.", "chunk_id": "D004_C0000"},
                {"entity_name": "Ambroise Thomas", "entity_id": None, "attribute_name": "OCCUPATION", "extracted_value": "Composer", "evidence_quote": "Charles Louis Ambroise Thomas was a French composer, best known for his operas Mignon and Hamlet.", "chunk_id": "D006_C0000"},
            ],
            "canonical_answer": "yes",
            "answer": "Both were opera composers.",
            "abstained": False,
            "cited_chunk_ids": ["D004_C0000", "D006_C0000"],
        })
    )

    output = OllamaAnswerGenerator(llm).generate_structured(result)
    assert output.canonical_answer == "yes"
    assert output.abstained is False
    assert set(output.cited_chunk_ids) == {"D004_C0000", "D006_C0000"}


# ---------------------------------------------------------------------------
# Test 4: Unsupported occupational equivalence
# ---------------------------------------------------------------------------
def test_4_unsupported_occupational_equivalence_causes_abstention() -> None:
    """If one entity's evidence only supports general composer without the requested opera qualification, abstain."""
    c1 = _make_chunk(
        "D004_C0000",
        "D004",
        "Giuseppe Fortunino Francesco Verdi was an Italian opera composer.",
    )
    c2 = _make_chunk(
        "D008_C0000",
        "D008",
        "John Towner Williams is an American composer and conductor, known for composing film scores.",
    )

    query_plan = QueryPlan(
        question="Are Giuseppe Verdi and John Williams both Opera composers ?",
        seed_entities=[
            SeedEntitySpec(name="Giuseppe Verdi", expected_type="PERSON"),
            SeedEntitySpec(name="John Williams", expected_type="PERSON"),
        ],
        relationship_path=["OCCUPATION"],
    )

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Verdi composed operas, John Williams composed film scores.",
            "extracted_attributes": [
                {"entity_name": "Giuseppe Verdi", "entity_id": None, "attribute_name": "OCCUPATION", "extracted_value": "Opera composer", "evidence_quote": "Giuseppe Fortunino Francesco Verdi was an Italian opera composer.", "chunk_id": "D004_C0000"},
                {"entity_name": "John Williams", "entity_id": None, "attribute_name": "OCCUPATION", "extracted_value": "Composer", "evidence_quote": "John Towner Williams is an American composer and conductor, known for composing film scores.", "chunk_id": "D008_C0000"},
            ],
            "canonical_answer": "yes",
            "answer": "Both were composers.",
            "abstained": False,
            "cited_chunk_ids": ["D004_C0000", "D008_C0000"],
        })
    )

    output = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce").generate_structured(result)
    # Must NOT return yes because John Williams is not verified as an opera composer
    assert output.abstained is True
    assert "opera composer was not found" in (output.abstention_reason or "")


# ---------------------------------------------------------------------------
# Test 5: Nullable entity IDs with valid provenance
# ---------------------------------------------------------------------------
def test_5_nullable_entity_id_with_valid_provenance_accepted() -> None:
    """Nullable entity_id must deserialize without error and validate against evidence chunks."""
    c1 = _make_chunk(
        "D001_C0000",
        "D001",
        "Winner is a South Korean boy group formed in 2013 by YG Entertainment.",
    )

    fact = CandidateAttributeFact(
        entity_name="Winner",
        entity_id=None,
        attribute_name="FORMED_BY",
        extracted_value="YG Entertainment",
        evidence_quote="Winner is a South Korean boy group formed in 2013 by YG Entertainment.",
        chunk_id="D001_C0000",
    )

    valid, reason = FactValidator.validate_fact(fact, [c1])
    assert valid is True
    assert reason is None
    assert fact.entity_id is None


# ---------------------------------------------------------------------------
# Test 6: Missing or unsupported evidence still causing abstention
# ---------------------------------------------------------------------------
def test_6_missing_evidence_causes_abstention() -> None:
    """A fact citing a missing chunk or hallucinatory quote must be rejected and cause abstention."""
    c1 = _make_chunk(
        "D001_C0000",
        "D001",
        "The building is located in Istanbul, Turkey.",
    )

    # Quote does not occur in chunk
    fact = CandidateAttributeFact(
        entity_name="The building",
        entity_id=None,
        attribute_name="LOCATION",
        extracted_value="Laleli",
        evidence_quote="The building is located in Laleli neighborhood.",
        chunk_id="D001_C0000",
    )

    valid, reason = FactValidator.validate_fact(fact, [c1])
    assert valid is False
    assert "Evidence quote does not occur" in (reason or "")


# ---------------------------------------------------------------------------
# Test 7: Neighbourhood comparison
# ---------------------------------------------------------------------------
def test_7_neighbourhood_comparison_returns_no() -> None:
    """Different neighbourhoods in the same city must return no."""
    c1 = _make_chunk(
        "D006_C0000",
        "D006",
        "The Laleli Mosque is an 18th-century Ottoman imperial mosque located in Laleli, Fatih, Istanbul, Turkey.",
    )
    c2 = _make_chunk(
        "D007_C0000",
        "D007",
        "The Esma Sultan Mansion located at Bosphorus in Ortaköy neighborhood of Istanbul, Turkey, is a multi-purpose event venue.",
    )

    query_plan = QueryPlan(
        question="Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        seed_entities=[
            SeedEntitySpec(name="Laleli Mosque", expected_type="LOCATION"),
            SeedEntitySpec(name="Esma Sultan Mansion", expected_type="LOCATION"),
        ],
        relationship_path=["LOCATED_IN"],
    )

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Laleli Mosque is in Laleli, Esma Sultan Mansion is in Ortaköy.",
            "extracted_attributes": [
                {"entity_name": "Laleli Mosque", "entity_id": None, "attribute_name": "LOCATED_IN:NEIGHBORHOOD", "extracted_value": "Laleli", "evidence_quote": "The Laleli Mosque is an 18th-century Ottoman imperial mosque located in Laleli, Fatih, Istanbul, Turkey.", "chunk_id": "D006_C0000"},
                {"entity_name": "Esma Sultan Mansion", "entity_id": None, "attribute_name": "LOCATED_IN:NEIGHBORHOOD", "extracted_value": "Ortaköy", "evidence_quote": "The Esma Sultan Mansion located at Bosphorus in Ortaköy neighborhood of Istanbul, Turkey, is a multi-purpose event venue.", "chunk_id": "D007_C0000"},
            ],
            "canonical_answer": "no",
            "answer": "Laleli Mosque is in Laleli, while Esma Sultan Mansion is in Ortaköy.",
            "abstained": False,
            "cited_chunk_ids": ["D006_C0000", "D007_C0000"],
        })
    )

    output = OllamaAnswerGenerator(llm).generate_structured(result)
    assert output.canonical_answer == "no"
    assert output.abstained is False
    assert set(output.cited_chunk_ids) == {"D006_C0000", "D007_C0000"}


# ---------------------------------------------------------------------------
# Test 8: Correct neighbourhood-versus-city distinction
# ---------------------------------------------------------------------------
def test_8_neighbourhood_vs_city_distinction() -> None:
    """Must not confuse a neighbourhood with the city containing it."""
    s1 = "The Laleli Mosque is located in Laleli, Fatih, Istanbul, Turkey."
    s2 = "The Blue Mosque is located in Istanbul, Turkey."

    # For s1, neighborhood is Laleli, city is Istanbul
    assert FactValidator.extract_location_from_text(s1, granularity="neighborhood") == "Laleli"
    assert FactValidator.extract_location_from_text(s1, granularity="city") == "Istanbul"

    # For s2, no neighborhood is mentioned, only city Istanbul.
    # Must NOT substitute Istanbul as neighborhood!
    assert FactValidator.extract_location_from_text(s2, granularity="neighborhood") is None
    assert FactValidator.extract_location_from_text(s2, granularity="city") == "Istanbul"


# ---------------------------------------------------------------------------
# Test 9: Existing nationality comparisons
# ---------------------------------------------------------------------------
def test_9_existing_nationality_comparison_still_works() -> None:
    """Nationality demonyms and country aliases must normalize and compare correctly."""
    c1 = _make_chunk("D1_C1", "D1", "Local H is an American rock band.")
    c2 = _make_chunk("D2_C1", "D2", "For Against was an American post-punk band.")

    query_plan = QueryPlan(
        question="Are Local H and For Against both from the United States?",
        seed_entities=[
            SeedEntitySpec(name="Local H", expected_type="ORGANIZATION"),
            SeedEntitySpec(name="For Against", expected_type="ORGANIZATION"),
        ],
        relationship_path=["NATIONALITY"],
    )

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Both are American.",
            "extracted_attributes": [
                {"entity_name": "Local H", "entity_id": None, "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "Local H is an American rock band.", "chunk_id": "D1_C1"},
                {"entity_name": "For Against", "entity_id": None, "attribute_name": "NATIONALITY", "extracted_value": "American", "evidence_quote": "For Against was an American post-punk band.", "chunk_id": "D2_C1"},
            ],
            "canonical_answer": "yes",
            "answer": "Both bands are American.",
            "abstained": False,
            "cited_chunk_ids": ["D1_C1", "D2_C1"],
        })
    )

    output = OllamaAnswerGenerator(llm).generate_structured(result)
    assert output.canonical_answer == "yes"
    assert output.abstained is False


# ---------------------------------------------------------------------------
# Test 10: Existing bridge-question functionality
# ---------------------------------------------------------------------------
def test_10_existing_bridge_question_still_works() -> None:
    """Bridge questions extracting non-entity values with nullable entity_id work smoothly."""
    c1 = _make_chunk(
        "D004_C0000",
        "D004",
        "Adriana Trigiani is an Italian American best-selling author, film director, based in Greenwich Village, New York City.",
    )

    query_plan = QueryPlan(
        question='The director of the romantic comedy "Big Stone Gap" is based in what New York city?',
        seed_entities=[
            SeedEntitySpec(name="Big Stone Gap", expected_type="CREATIVE_WORK"),
        ],
    )

    result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Adriana Trigiani directed Big Stone Gap and is based in Greenwich Village, New York City.",
            "extracted_attributes": [
                {
                    "entity_name": "Adriana Trigiani",
                    "entity_id": None,
                    "attribute_name": "BASED_IN",
                    "extracted_value": "Greenwich Village, New York City",
                    "evidence_quote": "Adriana Trigiani is an Italian American best-selling author, film director, based in Greenwich Village, New York City.",
                    "chunk_id": "D004_C0000",
                }
            ],
            "canonical_answer": "Greenwich Village, New York City",
            "answer": "Adriana Trigiani is based in Greenwich Village, New York City [D004_C0000].",
            "abstained": False,
            "cited_chunk_ids": ["D004_C0000"],
        })
    )

    output = OllamaAnswerGenerator(llm).generate_structured(result)
    assert output.canonical_answer == "Greenwich Village, New York City"
    assert output.abstained is False
    assert output.cited_chunk_ids == ["D004_C0000"]


# ---------------------------------------------------------------------------
# Test 11: Citation metric calculations
# ---------------------------------------------------------------------------
def test_11_citation_metric_calculations() -> None:
    """Zero-citation abstentions must not penalize citation precision; metrics separate precision and coverage."""
    # Simulate 3 benchmark questions:
    # Q1: Answered, 2 citations, both valid -> precision 1.0, coverage 1.0
    # Q2: Answered, 1 citation, valid -> precision 1.0, coverage 0.5
    # Q3: Abstained, 0 citations -> should NOT be counted as 0.0 invalid citation

    q_records = [
        {
            "question_type": "comparison",
            "is_regression_control": False,
            "status": "completed",
            "metrics": {
                "answer_metrics": {"exact_match": 1, "token_f1": 1.0, "abstained": False},
                "retrieval_metrics": {"recall_at_2": 1.0, "recall_at_5": 1.0},
                "citation_metrics": {
                    "total_citations": 2,
                    "valid_citations": 2,
                    "has_citations": True,
                    "citation_precision": 1.0,
                    "citation_gold_coverage": 1.0,
                },
            },
        },
        {
            "question_type": "bridge",
            "is_regression_control": False,
            "status": "completed",
            "metrics": {
                "answer_metrics": {"exact_match": 1, "token_f1": 1.0, "abstained": False},
                "retrieval_metrics": {"recall_at_2": 1.0, "recall_at_5": 1.0},
                "citation_metrics": {
                    "total_citations": 1,
                    "valid_citations": 1,
                    "has_citations": True,
                    "citation_precision": 1.0,
                    "citation_gold_coverage": 0.5,
                },
            },
        },
        {
            "question_type": "bridge",
            "is_regression_control": False,
            "status": "completed",
            "metrics": {
                "answer_metrics": {"exact_match": 0, "token_f1": 0.0, "abstained": True},
                "retrieval_metrics": {"recall_at_2": 1.0, "recall_at_5": 1.0},
                "citation_metrics": {
                    "total_citations": 0,
                    "valid_citations": 0,
                    "has_citations": False,
                    "citation_precision": None,
                    "citation_gold_coverage": 0.0,
                },
            },
        },
    ]

    # Calculate metrics according to FIX 5 specification
    total_emitted = sum(q["metrics"]["citation_metrics"]["total_citations"] for q in q_records)
    total_valid = sum(q["metrics"]["citation_metrics"]["valid_citations"] for q in q_records)
    precision_among_emitted = (total_valid / total_emitted) if total_emitted > 0 else 1.0

    questions_with_citations = sum(1 for q in q_records if q["metrics"]["citation_metrics"]["has_citations"])
    question_coverage = questions_with_citations / len(q_records)

    answered_records = [q for q in q_records if not q["metrics"]["answer_metrics"]["abstained"]]
    answered_em = sum(q["metrics"]["answer_metrics"]["exact_match"] for q in answered_records) / len(answered_records)

    # Assertions
    assert total_emitted == 3
    assert total_valid == 3
    assert precision_among_emitted == 1.0  # NOT 0.667 or 0.444!
    assert question_coverage == 2 / 3
    assert len(answered_records) == 2
    assert answered_em == 1.0


# ---------------------------------------------------------------------------
# Test 12: Grounded post-nominal suffix preservation (Q1)
# ---------------------------------------------------------------------------
def test_12_grounded_post_nominal_suffix_preservation() -> None:
    """Post-nominal honorifics and suffixes (DSC, Jr., OBE) must be preserved in selection canonical answers."""
    c1 = _make_chunk(
        "D002_C0000",
        "D002",
        "Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer.",
    )
    c2 = _make_chunk(
        "D007_C0000",
        "D007",
        "Robert Erskine Childers DSC (25 June 1870 – 24 November 1922) was a British author and politician.",
    )

    query_plan = QueryPlan(
        question="Which writer was from England, Henry Roth or Robert Erskine Childers?",
        seed_entities=[
            SeedEntitySpec(name="Henry Roth", expected_type="PERSON"),
            SeedEntitySpec(name="Robert Erskine Childers", expected_type="PERSON"),
        ],
        relationship_path=["NATIONALITY"],
    )

    retrieval_result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Henry Roth was American. Robert Erskine Childers was British.",
            "extracted_attributes": [
                {
                    "entity_name": "Henry Roth",
                    "entity_id": None,
                    "attribute_name": "NATIONALITY",
                    "extracted_value": "American",
                    "evidence_quote": "Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer.",
                    "chunk_id": "D002_C0000",
                },
                {
                    "entity_name": "Robert Erskine Childers",
                    "entity_id": None,
                    "attribute_name": "NATIONALITY",
                    "extracted_value": "British",
                    "evidence_quote": "Robert Erskine Childers DSC (25 June 1870 – 24 November 1922) was a British author and politician.",
                    "chunk_id": "D007_C0000",
                },
            ],
            "canonical_answer": "Robert Erskine Childers",
            "answer": "Robert Erskine Childers was British.",
            "abstained": False,
            "cited_chunk_ids": ["D002_C0000", "D007_C0000"],
        })
    )

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(retrieval_result)

    assert output.canonical_answer == "Robert Erskine Childers DSC"
    assert output.abstained is False


# ---------------------------------------------------------------------------
# Test 13: Multi-hop boy group & record label bridge (Q6)
# ---------------------------------------------------------------------------
def test_13_multi_hop_boy_group_record_label_bridge() -> None:
    """Multi-hop bridge question must resolve ultimate target (YG Entertainment), not distractor."""
    c1 = _make_chunk(
        "D003_C0000",
        "D003",
        "2014 S/S is the debut album of South Korean group WINNER, released in 2014 by record label YG Entertainment.",
    )
    c2 = _make_chunk(
        "D005_C0000",
        "D005",
        "Winner is a South Korean boy group formed in 2013 by YG Entertainment.",
    )
    c3 = _make_chunk(
        "D006_C0000",
        "D006",
        "Madtown is a South Korean boy group formed in 2014 by J. Tune Camp.",
    )

    query_plan = QueryPlan(
        question="2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        intent="LOCAL",
    )

    retrieval_result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2, c3],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "2014 S/S is the debut album of WINNER [D003_C0000], which was formed by YG Entertainment [D005_C0000]. Madtown was formed in 2014 by J. Tune Camp but did not release 2014 S/S.",
            "extracted_attributes": [
                {
                    "entity_name": "WINNER",
                    "entity_id": None,
                    "attribute_name": "FORMED_BY",
                    "extracted_value": "YG Entertainment",
                    "evidence_quote": "Winner is a South Korean boy group formed in 2013 by YG Entertainment.",
                    "chunk_id": "D005_C0000",
                }
            ],
            "canonical_answer": "YG Entertainment",
            "answer": "2014 S/S is the debut album of the South Korean boy group WINNER, which was formed by YG Entertainment [D003_C0000] [D005_C0000].",
            "abstained": False,
            "cited_chunk_ids": ["D003_C0000", "D005_C0000"],
        })
    )

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(retrieval_result)

    assert output.canonical_answer == "YG Entertainment"
    assert output.abstained is False
    assert "D006_C0000" not in output.cited_chunk_ids
    assert set(output.cited_chunk_ids) == {"D003_C0000", "D005_C0000"}


# ---------------------------------------------------------------------------
# Test 14: Multi-hop presidential administration term years bridge (Q7)
# ---------------------------------------------------------------------------
def test_14_multi_hop_presidential_term_bridge() -> None:
    """Bridge question asking for term years must provide the years, not the intermediate president."""
    c1 = _make_chunk(
        "D009_C0000",
        "D009",
        "Roger O. Egeberg served as Assistant Secretary for Health and Scientific Affairs during the Nixon administration.",
    )
    c2 = _make_chunk(
        "D003_C0000",
        "D003",
        "Richard Milhous Nixon was the 37th President of the United States from 1969 until 1974.",
    )

    query_plan = QueryPlan(
        question="Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?",
        intent="LOCAL",
    )

    retrieval_result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "Roger O. Egeberg served under President Richard Nixon [D009_C0000]. Richard Nixon served as President from 1969 until 1974 [D003_C0000].",
            "extracted_attributes": [
                {
                    "entity_name": "Richard Nixon",
                    "entity_id": None,
                    "attribute_name": "SERVED_YEARS",
                    "extracted_value": "1969 until 1974",
                    "evidence_quote": "Richard Milhous Nixon was the 37th President of the United States from 1969 until 1974.",
                    "chunk_id": "D003_C0000",
                }
            ],
            "canonical_answer": "1969 until 1974",
            "answer": "President Richard Nixon served from 1969 until 1974 [D003_C0000].",
            "abstained": False,
            "cited_chunk_ids": ["D009_C0000", "D003_C0000"],
        })
    )

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(retrieval_result)

    assert output.canonical_answer == "1969 until 1974"
    assert output.abstained is False
    assert set(output.cited_chunk_ids) == {"D009_C0000", "D003_C0000"}


# ---------------------------------------------------------------------------
# Test 15: Cross-domain multi-hop attribution (Domain Invariance)
# ---------------------------------------------------------------------------
def test_15_synthetic_multi_hop_domain_invariance() -> None:
    """Multi-hop bridge question in academic/scientific domain resolves correctly across hops."""
    c1 = _make_chunk(
        "ACAD_C01",
        "DOC_01",
        "The Raman effect was discovered in 1928 by C. V. Raman at the Indian Association for the Cultivation of Science located in Kolkata.",
    )
    c2 = _make_chunk(
        "ACAD_C02",
        "DOC_02",
        "Kolkata is the primary capital city of the state of West Bengal in India.",
    )

    query_plan = QueryPlan(
        question="The institution where the Raman effect was discovered is located in a city belonging to what state?",
        intent="LOCAL",
    )

    retrieval_result = RetrievalResult(
        kb_id="test_kb",
        question=query_plan.question,
        query_plan=query_plan,
        evidence_list=[c1, c2],
    )

    llm = OllamaLLMClient(
        custom_generator=lambda _p: json.dumps({
            "reasoning": "The Raman effect was discovered at the Indian Association for the Cultivation of Science in Kolkata [ACAD_C01]. Kolkata belongs to the state of West Bengal [ACAD_C02].",
            "extracted_attributes": [
                {
                    "entity_name": "Kolkata",
                    "entity_id": None,
                    "attribute_name": "STATE",
                    "extracted_value": "West Bengal",
                    "evidence_quote": "Kolkata is the primary capital city of the state of West Bengal in India.",
                    "chunk_id": "ACAD_C02",
                }
            ],
            "canonical_answer": "West Bengal",
            "answer": "The Indian Association for the Cultivation of Science is in Kolkata [ACAD_C01], which is located in West Bengal [ACAD_C02].",
            "abstained": False,
            "cited_chunk_ids": ["ACAD_C01", "ACAD_C02"],
        })
    )

    gen = OllamaAnswerGenerator(llm, comparative_reasoning_mode="enforce")
    output = gen.generate_structured(retrieval_result)

    assert output.canonical_answer == "West Bengal"
    assert output.abstained is False
    assert set(output.cited_chunk_ids) == {"ACAD_C01", "ACAD_C02"}
