#!/usr/bin/env python3
"""Unit tests for offline scoring logic (experiments/synth_kg/score.py)."""

from typing import Any, Dict, List
import pytest

from experiments.synth_kg.score import (
    compute_path_coverage,
    evaluate_single_answer,
    exact_sign_test,
    is_whole_word_match,
    normalize_doc_id,
    normalize_text,
    score_extraction,
    wilson_ci,
)


def test_doc_id_normalization():
    assert normalize_doc_id("D001") == "DOC_0001"
    assert normalize_doc_id("DOC_0001") == "DOC_0001"
    assert normalize_doc_id("D033") == "DOC_0033"
    assert normalize_doc_id("DOC_0033") == "DOC_0033"
    assert normalize_doc_id("D0060") == "DOC_0060"
    assert normalize_doc_id("") == ""



def test_text_normalization_and_word_matching():
    assert normalize_text("  Pryxeldor,   Iskorlan!  ") == "pryxeldor iskorlan"
    assert is_whole_word_match("Acme", "Welcome to Acme Corp.")
    assert is_whole_word_match("Acme Corp", "Welcome to Acme Corp.")
    # Must NOT match substring inside another word
    assert not is_whole_word_match("bar", "Welcome to foobar Corp.")
    assert not is_whole_word_match("Acme", "Welcome to AcmeTech.")


def test_answer_matching_and_failure_classes():
    gold_entities_by_type = {
        "EMPLOYEE": [
            {"id": "ENT_1", "canonical_name": "Pryxeldor Iskorlan", "aliases": ["P. Iskorlan"], "type": "EMPLOYEE"},
            {"id": "ENT_2", "canonical_name": "Karneth Vance", "aliases": ["K. Vance"], "type": "EMPLOYEE"},
        ],
        "PRODUCT": [
            {"id": "ENT_3", "canonical_name": "Zenthis Cloud", "aliases": ["Z-Cloud"], "type": "PRODUCT"},
        ],
    }

    q_answerable = {
        "id": "Q001",
        "group": "hop1",
        "question": "Who manages the team?",
        "gold_answer": "Pryxeldor Iskorlan",
        "gold_answer_aliases": ["P. Iskorlan"],
        "target_type": "EMPLOYEE",
    }

    # 1. Exact canonical name match
    rec_exact = {"answer": "The manager is Pryxeldor Iskorlan.", "abstained": False, "generation_error": False}
    corr, fail = evaluate_single_answer(q_answerable, rec_exact, gold_entities_by_type)
    assert corr is True
    assert fail is None

    # 2. Alias match
    rec_alias = {"answer": "Based on the documents, P. Iskorlan leads the department.", "abstained": False, "generation_error": False}
    corr, fail = evaluate_single_answer(q_answerable, rec_alias, gold_entities_by_type)
    assert corr is True
    assert fail is None

    # 3. Generation error overrides everything
    rec_err = {"answer": "Pryxeldor Iskorlan", "abstained": False, "generation_error": True}
    corr, fail = evaluate_single_answer(q_answerable, rec_err, gold_entities_by_type)
    assert corr is False
    assert fail == "ERROR"

    # 4. Abstained on answerable question
    rec_abs = {"answer": "I do not have enough information.", "abstained": True, "generation_error": False}
    corr, fail = evaluate_single_answer(q_answerable, rec_abs, gold_entities_by_type)
    assert corr is False
    assert fail == "ABSTAINED"

    # 5. Wrong entity of same type
    rec_wrong_ent = {"answer": "The project is managed by Karneth Vance.", "abstained": False, "generation_error": False}
    corr, fail = evaluate_single_answer(q_answerable, rec_wrong_ent, gold_entities_by_type)
    assert corr is False
    assert fail == "WRONG_ENTITY"

    # 6. Other wrong response
    rec_other = {"answer": "The project is managed by Quantum Computing Solutions.", "abstained": False, "generation_error": False}
    corr, fail = evaluate_single_answer(q_answerable, rec_other, gold_entities_by_type)
    assert corr is False
    assert fail == "OTHER"

    # 7. Unanswerable question
    q_unans = {
        "id": "Q002",
        "group": "unanswerable",
        "question": "When was X founded?",
        "gold_answer": None,
    }
    rec_unans_correct = {"answer": "I cannot answer this question based on the provided documents.", "abstained": True, "generation_error": False}
    corr, fail = evaluate_single_answer(q_unans, rec_unans_correct)
    assert corr is True
    assert fail is None

    rec_unans_wrong = {"answer": "It was founded in 2012.", "abstained": False, "generation_error": False}
    corr, fail = evaluate_single_answer(q_unans, rec_unans_wrong)
    assert corr is False
    assert fail == "OTHER"


def test_sign_test_and_confidence_intervals():
    # 0 discordant pairs -> p = 1.0
    assert exact_sign_test(0, 0) == 1.0
    # Symmetric counts -> p = 1.0
    assert exact_sign_test(5, 5) == 1.0
    # Strong asymmetry: 10 vs 0 -> p = 2 * (1/1024) = 0.001953125 (< 0.05)
    p_val = exact_sign_test(10, 0)
    assert p_val < 0.01

    # Wilson interval coverage
    low, high = wilson_ci(10, 20)
    assert 0.0 < low < 0.5 < high < 1.0
    assert wilson_ci(0, 0) == (0.0, 0.0)
    low_0, high_0 = wilson_ci(0, 10)
    assert low_0 == 0.0
    assert high_0 > 0.0


def test_extraction_scoring():
    gold_kg = {
        "entities": [
            {"id": "G1", "canonical_name": "Alpha Corp", "aliases": ["Alpha"], "type": "ORGANIZATION"},
            {"id": "G2", "canonical_name": "Beta Labs", "aliases": ["Beta"], "type": "ORGANIZATION"},
            {"id": "G3", "canonical_name": "Gamma Inc", "aliases": [], "type": "ORGANIZATION"},
        ],
        "edges": [
            {"id": "E1", "source": "G1", "target": "G2", "relation": "DEVELOPS", "evidence_quote": "Alpha Corp develops Beta Labs products."},
            {"id": "E2", "source": "G2", "target": "G3", "relation": "SERVES", "evidence_quote": "Beta Labs serves Gamma Inc."},
        ],
    }

    # Extracted: G1 is split into C1 and C2; G2 is extracted as C3; C4 is spurious
    canonical_entities = [
        {"entity_id": "C1", "name": "Alpha Corp", "aliases": []},
        {"entity_id": "C2", "name": "Alpha", "aliases": []},
        {"entity_id": "C3", "name": "Beta Labs", "aliases": []},
        {"entity_id": "C4", "name": "Nonexistent Entity", "aliases": []},
    ]

    # Validated relationships: C1 -> C3 (exact match to E1)
    validated_relationships = [
        {"source_entity_id": "C1", "target_entity_id": "C3", "relation_type": "DEVELOPS"},
        {"source_entity_id": "C3", "target_entity_id": "C4", "relation_type": "SERVES"},
    ]

    # Triples: E2 was extracted but rejected for hallucinated quote
    all_triples = [
        {
            "source_entity_id": "M1",
            "target_entity_id": "M2",
            "relation_type": "SERVES",
            "evidence_quotes": ["Beta Labs serves Gamma Inc."],
            "validation_status": "REJECTED_HALLUCINATED_QUOTE",
        }
    ]

    entity_mentions = [
        {"mention_id": "M1", "surface_name": "Beta Labs"},
        {"mention_id": "M2", "surface_name": "Gamma Inc"},
    ]

    res = score_extraction(
        gold_kg=gold_kg,
        canonical_entities=canonical_entities,
        validated_relationships=validated_relationships,
        all_extracted_triples=all_triples,
        entity_mentions=entity_mentions,
    )

    ent = res["entity_scoring"]
    # Recalled gold: G1 and G2 -> 2/3
    assert pytest.approx(ent["recall"], 0.01) == 2 / 3
    # Valid extracted: C1, C2, C3 -> 3/4
    assert pytest.approx(ent["precision"], 0.01) == 3 / 4
    # Split: G1 has both C1 and C2
    assert ent["splits"] == 1
    assert ent["false_merges"] == 0

    rel = res["relationship_scoring"]
    # Validated edges: C1->C3 is exact match, C3->C4 is not -> 1/2 precision
    assert pytest.approx(rel["exact_precision"], 0.01) == 0.5
    # Recalled gold edges: E1 recalled, E2 not -> 1/2 recall
    assert pytest.approx(rel["exact_recall"], 0.01) == 0.5

    fates = res["fate_breakdown"]["edge_fates"]
    assert fates["E1"]["fate"] == "VALIDATED_CORRECT"
    assert fates["E2"]["fate"] == "REJECTED: REJECTED_HALLUCINATED_QUOTE"


def test_path_coverage():
    gold_kg = {
        "entities": [
            {"id": "G1", "canonical_name": "E1", "aliases": []},
            {"id": "G2", "canonical_name": "E2", "aliases": []},
            {"id": "G3", "canonical_name": "E3", "aliases": []},
        ],
        "edges": [
            {"id": "EDGE1", "source": "G1", "target": "G2", "relation": "R1"},
            {"id": "EDGE2", "source": "G2", "target": "G3", "relation": "R2"},
        ],
    }

    canonical_entities = [
        {"entity_id": "C1", "name": "E1", "aliases": []},
        {"entity_id": "C2", "name": "E2", "aliases": []},
        {"entity_id": "C3", "name": "E3", "aliases": []},
    ]

    # Graph has C1-C2 and C2-C3
    validated_relationships = [
        {"source_entity_id": "C1", "target_entity_id": "C2", "relation_type": "R1"},
        {"source_entity_id": "C2", "target_entity_id": "C3", "relation_type": "R2"},
    ]

    questions = [
        {
            "id": "Q1",
            "group": "hop2",
            "hop_count": 2,
            "question": "What is the relation from E1 to E3?",
            "gold_answer": "E3",
            "gold_path": ["EDGE1", "EDGE2"],
        },
        {
            "id": "Q2",
            "group": "hop1",
            "hop_count": 1,
            "question": "What is the relation from E1 to E3 in 1 hop?",
            "gold_answer": "E3",
            "gold_path": ["EDGE1", "EDGE2"],
        },
    ]

    cov = compute_path_coverage(questions, canonical_entities, validated_relationships, gold_kg)
    # Q1 hop2: distance is 2 <= 2 -> True
    assert cov["question_path_coverage"]["Q1"] is True
    # Q2 hop1: distance is 2 <= 1 -> False
    assert cov["question_path_coverage"]["Q2"] is False
    assert cov["group_path_coverage"]["hop2"]["fraction"] == 1.0
    assert cov["group_path_coverage"]["hop1"]["fraction"] == 0.0
