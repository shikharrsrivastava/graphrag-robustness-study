"""Unit tests for Phase 1 Step 4 (Entity Extraction) and Step 5 (Relationship Extraction)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
from src.core.configuration import load_engine_config
from src.core.models import (
    Chunk,
    EntityMention,
    Polarity,
    Relationship,
    ValidationStatus,
)
from src.extraction.entity_extractor import OllamaEntityExtractor, normalize_entity_name
from src.extraction.relation_extractor import OllamaRelationshipExtractor
from src.extraction.schemas import (
    OllamaLLMClient,
    RawEntityExtractionOutput,
    RawEntityMentionItem,
    RawRelationshipExtractionOutput,
    RawRelationshipItem,
)
from src.storage.document_store import DocumentStore


@pytest.fixture
def academic_ontology():
    """Load the Academic domain ontology from configs/domains/academic.yaml."""
    cfg = load_engine_config(Path("configs/domains/academic.yaml"))
    return cfg.ontology


def test_normalize_entity_name() -> None:
    """Test surface name normalization: punctuation stripping, whitespace collapse, lowercase."""
    assert normalize_entity_name("  Professor Sharma,  ") == "professor sharma"
    assert normalize_entity_name("Rahul.") == "rahul"
    assert normalize_entity_name("  IIIT   Naya   Raipur  ") == "iiit naya raipur"
    assert normalize_entity_name('"Dr. Jane Doe"') == "dr. jane doe"


def test_entity_extraction_teaching_example(academic_ontology) -> None:
    """Test entity extraction on the teaching example with char offsets and ontology filtering."""
    text = "Rahul studies at IIIT Naya Raipur. Rahul works under Professor Sharma."
    chunk = Chunk(
        kb_id="academic_kb",
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="teaching.txt",
        page_numbers=[1],
        chunk_index=0,
        text=text,
        char_count=len(text),
        content_hash="dummy_hash",
    )

    # Mock LLM returns 3 valid academic entities and 1 invalid ontology type
    mock_payload = {
        "entities": [
            {
                "surface_name": "Rahul",
                "entity_type": "PERSON",
                "description": "Student at IIIT Naya Raipur",
            },
            {
                "surface_name": "IIIT Naya Raipur",
                "entity_type": "UNIVERSITY",
                "description": "Institute of Information Technology",
            },
            {
                "surface_name": "Professor Sharma",
                "entity_type": "PERSON",
                "description": "Faculty supervisor",
            },
            {
                "surface_name": "Cybernetic Satellite",
                "entity_type": "SPACECRAFT",  # Unknown type in academic ontology
                "description": "Orbital probe",
            },
        ]
    }

    mock_client = OllamaLLMClient(custom_generator=lambda prompt: json.dumps(mock_payload))
    extractor = OllamaEntityExtractor(ontology=academic_ontology, llm_client=mock_client)

    mentions = extractor.extract_entities(chunk)

    # The unknown SPACECRAFT type must be skipped
    assert len(mentions) == 3

    # Check mention 1: Rahul
    m1 = mentions[0]
    assert m1.mention_id == "D001_C0000_M001"
    assert m1.surface_name == "Rahul"
    assert m1.normalized_name == "rahul"
    assert m1.entity_type == "PERSON"
    assert m1.char_start == 0
    assert m1.char_end == 5
    assert text[m1.char_start:m1.char_end] == "Rahul"
    assert m1.kb_id == "academic_kb"
    assert m1.source_chunk_id == "D001_C0000"
    assert m1.source_doc_id == "D001"
    assert m1.page_numbers == [1]

    # Check mention 2: IIIT Naya Raipur
    m2 = mentions[1]
    assert m2.mention_id == "D001_C0000_M002"
    assert m2.surface_name == "IIIT Naya Raipur"
    assert m2.normalized_name == "iiit naya raipur"
    assert m2.entity_type == "UNIVERSITY"
    assert m2.char_start == 17
    assert m2.char_end == 33
    assert text[m2.char_start:m2.char_end] == "IIIT Naya Raipur"

    # Check mention 3: Professor Sharma
    m3 = mentions[2]
    assert m3.mention_id == "D001_C0000_M003"
    assert m3.surface_name == "Professor Sharma"
    assert m3.normalized_name == "professor sharma"
    assert m3.entity_type == "PERSON"
    assert m3.char_start == 53
    assert m3.char_end == 69
    assert text[m3.char_start:m3.char_end] == "Professor Sharma"


def test_ollama_llm_client_bounded_retry() -> None:
    """Test LLM client bounded retries on malformed JSON until success or exhaustion."""
    call_count = 0

    def flaky_generator(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "NOT VALID JSON {broken..."
        return json.dumps(
            {
                "entities": [
                    {
                        "surface_name": "Rahul",
                        "entity_type": "PERSON",
                        "description": "Student",
                    }
                ]
            }
        )

    client = OllamaLLMClient(custom_generator=flaky_generator)
    result = client.generate_json("test prompt", RawEntityExtractionOutput, max_retries=2)

    assert call_count == 2
    assert len(result.entities) == 1
    assert result.entities[0].surface_name == "Rahul"

    # Test error exhaustion
    def always_broken_generator(prompt: str) -> str:
        return "definitely not json"

    exhausted_client = OllamaLLMClient(custom_generator=always_broken_generator)
    with pytest.raises(ValueError, match="Failed to generate valid JSON"):
        exhausted_client.generate_json("test prompt", RawEntityExtractionOutput, max_retries=1)


def test_relationship_extraction_validations(academic_ontology) -> None:
    """Test relationship extraction validations: VALIDATED, REJECTED_QUOTE_MISMATCH, REJECTED_NEGATION, REJECTED_ONTOLOGY."""
    chunk_text = (
        "Rahul studies at IIIT Naya Raipur. "
        "Rahul works under Professor Sharma. "
        "Alice does not collaborate with Bob."
    )
    chunk = Chunk(
        kb_id="academic_kb",
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="teaching.txt",
        page_numbers=[1],
        chunk_index=0,
        text=chunk_text,
        char_count=len(chunk_text),
        content_hash="h_chunk",
    )

    mentions = [
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0000_M001",
            surface_name="Rahul",
            normalized_name="rahul",
            entity_type="PERSON",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0000_M002",
            surface_name="IIIT Naya Raipur",
            normalized_name="iiit naya raipur",
            entity_type="UNIVERSITY",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0000_M003",
            surface_name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0000_M004",
            surface_name="Alice",
            normalized_name="alice",
            entity_type="PERSON",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0000_M005",
            surface_name="Bob",
            normalized_name="bob",
            entity_type="PERSON",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
    ]

    mock_relationships = {
        "relationships": [
            # 1. Valid relationship (Rahul WORKS_UNDER Professor Sharma)
            {
                "source_mention_id": "D001_C0000_M001",
                "target_mention_id": "D001_C0000_M003",
                "relation_type": "WORKS_UNDER",
                "evidence_quote": "Rahul works under Professor Sharma.",
                "polarity": "POSITIVE",
                "confidence": 0.95,
            },
            # 2. Fabricated quote (Quote not in chunk)
            {
                "source_mention_id": "D001_C0000_M001",
                "target_mention_id": "D001_C0000_M002",
                "relation_type": "STUDIES_AT",
                "evidence_quote": "Rahul enrolled at Oxford University in England.",
                "polarity": "POSITIVE",
                "confidence": 0.90,
            },
            # 3. Explicit negative polarity
            {
                "source_mention_id": "D001_C0000_M004",
                "target_mention_id": "D001_C0000_M005",
                "relation_type": "WORKS_UNDER",
                "evidence_quote": "Alice does not collaborate with Bob.",
                "polarity": "NEGATIVE",
                "confidence": 0.85,
            },
            # 4. Negation pattern in quote with POSITIVE polarity flag
            {
                "source_mention_id": "D001_C0000_M004",
                "target_mention_id": "D001_C0000_M005",
                "relation_type": "WORKS_UNDER",
                "evidence_quote": "Alice does not collaborate with Bob.",
                "polarity": "POSITIVE",
                "confidence": 0.80,
            },
            # 5. Disallowed ontology triple (UNIVERSITY STUDIES_AT PERSON is reversed/invalid)
            {
                "source_mention_id": "D001_C0000_M002",
                "target_mention_id": "D001_C0000_M001",
                "relation_type": "STUDIES_AT",
                "evidence_quote": "Rahul studies at IIIT Naya Raipur.",
                "polarity": "POSITIVE",
                "confidence": 0.88,
            },
            # 6. Fallback matching by surface name instead of mention ID
            {
                "source_mention_id": "Rahul",
                "target_mention_id": "IIIT Naya Raipur",
                "relation_type": "STUDIES_AT",
                "evidence_quote": "Rahul studies at IIIT Naya Raipur.",
                "polarity": "POSITIVE",
                "confidence": 0.99,
            },
            # 7. Self-loop (should be skipped)
            {
                "source_mention_id": "D001_C0000_M001",
                "target_mention_id": "D001_C0000_M001",
                "relation_type": "STUDIES_AT",
                "evidence_quote": "Rahul studies at IIIT Naya Raipur.",
                "polarity": "POSITIVE",
            },
        ]
    }

    mock_client = OllamaLLMClient(custom_generator=lambda prompt: json.dumps(mock_relationships))
    extractor = OllamaRelationshipExtractor(ontology=academic_ontology, llm_client=mock_client)

    results = extractor.extract_relationships(chunk, mentions)

    # 7 candidates in mock: 1 is self loop (skipped), so 6 are returned
    assert len(results) == 6

    # 1. Valid relationship
    assert results[0].validation_status == ValidationStatus.VALIDATED
    assert results[0].relation_type == "WORKS_UNDER"
    assert results[0].source_entity_id == "D001_C0000_M001"
    assert results[0].target_entity_id == "D001_C0000_M003"
    assert results[0].relationship_id == "D001_C0000_R001"

    # 2. Fabricated quote
    assert results[1].validation_status == ValidationStatus.REJECTED_QUOTE_MISMATCH

    # 3. Explicit negative polarity
    assert results[2].validation_status == ValidationStatus.REJECTED_NEGATION

    # 4. Negation keyword in quote
    assert results[3].validation_status == ValidationStatus.REJECTED_NEGATION

    # 5. Invalid ontology constraint
    assert results[4].validation_status == ValidationStatus.REJECTED_ONTOLOGY

    # 6. Surface name fallback matching
    assert results[5].validation_status == ValidationStatus.VALIDATED
    assert results[5].source_entity_id == "D001_C0000_M001"
    assert results[5].target_entity_id == "D001_C0000_M002"
    assert results[5].relation_type == "STUDIES_AT"


def test_document_store_entity_mentions_and_triples_roundtrip(tmp_path: Path) -> None:
    """Test DocumentStore persistence and reload for entity_mentions.jsonl and triples.jsonl."""
    kb_id = "academic_kb"
    store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)

    m1 = EntityMention(
        kb_id=kb_id,
        mention_id="D001_C0000_M001",
        surface_name="Rahul",
        normalized_name="rahul",
        entity_type="PERSON",
        description="Student",
        source_chunk_id="D001_C0000",
        source_doc_id="D001",
        page_numbers=[1],
        char_start=0,
        char_end=5,
    )
    m2 = EntityMention(
        kb_id=kb_id,
        mention_id="D001_C0000_M002",
        surface_name="IIIT Naya Raipur",
        normalized_name="iiit naya raipur",
        entity_type="UNIVERSITY",
        description="Institute",
        source_chunk_id="D001_C0000",
        source_doc_id="D001",
        page_numbers=[1],
        char_start=17,
        char_end=33,
    )

    r1 = Relationship(
        kb_id=kb_id,
        relationship_id="D001_C0000_R001",
        source_entity_id="D001_C0000_M001",
        target_entity_id="D001_C0000_M002",
        relation_type="STUDIES_AT",
        source_chunk_ids=["D001_C0000"],
        source_doc_ids=["D001"],
        evidence_quotes=["Rahul studies at IIIT Naya Raipur."],
        polarity=Polarity.POSITIVE,
        validation_status=ValidationStatus.VALIDATED,
        uncalibrated_confidence=0.98,
    )

    store.save_entity_mentions([m1, m2])
    store.save_triples([r1])

    assert store.entity_mentions_path.exists()
    assert store.triples_path.exists()

    # Cold reload with fresh DocumentStore instance
    store2 = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    loaded_mentions = store2.load_entity_mentions()
    loaded_triples = store2.load_triples()

    assert len(loaded_mentions) == 2
    assert loaded_mentions == [m1, m2]

    assert len(loaded_triples) == 1
    assert loaded_triples == [r1]

    # Test KB isolation enforcement
    m_wrong = EntityMention(
        kb_id="wrong_kb",
        mention_id="M999",
        surface_name="Wrong",
        normalized_name="wrong",
        entity_type="PERSON",
        source_chunk_id="C999",
        source_doc_id="D999",
    )
    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_entity_mentions([m_wrong])

    r_wrong = Relationship(
        kb_id="wrong_kb",
        relationship_id="R999",
        source_entity_id="M1",
        target_entity_id="M2",
        relation_type="STUDIES_AT",
        source_chunk_ids=["C1"],
        evidence_quotes=["evidence"],
    )
    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_triples([r_wrong])
