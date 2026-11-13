"""Unit tests for Phase 1 Step 7 (Embedding Service) and Step 6 (Entity Resolution)."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List
import numpy as np
import pytest
from src.core.configuration import OllamaConfig, ResolutionConfig
from src.core.models import (
    CanonicalEntity,
    EntityMention,
    Polarity,
    Relationship,
    ValidationStatus,
)
from src.core.ontology import DomainOntology, RelationConstraint
from src.embeddings.embedding_service import (
    OllamaEmbeddingService,
    cosine_similarity_matrix,
)
from src.extraction.schemas import OllamaLLMClient
from src.resolution.entity_resolver import (
    ConservativeEntityResolver,
    EntityResolutionVerdict,
    strip_honorifics,
)
from src.storage.document_store import DocumentStore


# --- Test 1: Embedding Service Dimensions, Finiteness, and Cosine Similarity ---


def test_embedding_service_dimensions_and_finiteness() -> None:
    """Test embedding service validates 768-d vectors and rejects bad dimensions/NaNs."""
    config = OllamaConfig(expected_embedding_dim=768)

    # Valid custom embedder
    def valid_embedder(texts: List[str]) -> List[List[float]]:
        vec = [0.1] * 768
        return [list(vec) for _ in texts]

    service = OllamaEmbeddingService(config=config, custom_embedder=valid_embedder)

    vec = service.embed_text("test sentence")
    assert len(vec) == 768
    assert math.isfinite(vec[0])

    batch = service.embed_batch(["text 1", "text 2"])
    assert len(batch) == 2
    assert len(batch[0]) == 768
    assert len(batch[1]) == 768

    # Reject wrong dimension (e.g. 512)
    def wrong_dim_embedder(texts: List[str]) -> List[List[float]]:
        return [[0.1] * 512 for _ in texts]

    bad_dim_service = OllamaEmbeddingService(config=config, custom_embedder=wrong_dim_embedder)
    with pytest.raises(ValueError, match="dimension mismatch"):
        bad_dim_service.embed_text("test")

    # Reject NaN / Inf
    def nan_embedder(texts: List[str]) -> List[List[float]]:
        v = [0.1] * 768
        v[5] = float("nan")
        return [v for _ in texts]

    nan_service = OllamaEmbeddingService(config=config, custom_embedder=nan_embedder)
    with pytest.raises(ValueError, match="non-finite values"):
        nan_service.embed_text("test")

    # Reject zero-norm vectors
    def zero_norm_embedder(texts: List[str]) -> List[List[float]]:
        return [[0.0] * 768 for _ in texts]

    zero_service = OllamaEmbeddingService(config=config, custom_embedder=zero_norm_embedder)
    with pytest.raises(ValueError, match="zero or near-zero L2 norm"):
        zero_service.embed_text("test")


def test_cosine_similarity_matrix() -> None:
    """Test cosine similarity matrix calculation with L2 normalization."""
    # Orthogonal vectors in 3D
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0]])
    v3 = np.array([[-1.0, 0.0, 0.0]])

    sim_11 = cosine_similarity_matrix(v1, v1)
    assert np.isclose(sim_11[0, 0], 1.0)

    sim_12 = cosine_similarity_matrix(v1, v2)
    assert np.isclose(sim_12[0, 0], 0.0)

    sim_13 = cosine_similarity_matrix(v1, v3)
    assert np.isclose(sim_13[0, 0], -1.0)

    # Multi-row matrix
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, -1.0]])
    mat = cosine_similarity_matrix(a, b)
    assert mat.shape == (2, 2)
    assert np.isclose(mat[0, 0], 1.0)
    assert np.isclose(mat[0, 1], 0.0)
    assert np.isclose(mat[1, 0], 0.0)
    assert np.isclose(mat[1, 1], -1.0)


# --- Test 2: Exact & Honorific/Static Alias Resolution ---


def test_exact_and_honorific_alias_resolution() -> None:
    """Test exact and title-stripped alias resolution for PERSON entities."""
    ontology = DomainOntology(
        domain_name="academic",
        entity_types={"PERSON": "Researchers and faculty", "UNIVERSITY": "Institutes"},
        relationship_types={
            "WORKS_UNDER": RelationConstraint(name="WORKS_UNDER", description="Supervision")
        },
        static_aliases={"Professor Sharma": ["Prof. Sharma", "Dr. Sharma"]},
    )
    resolution_cfg = ResolutionConfig(require_type_match=True)
    resolver = ConservativeEntityResolver(resolution_config=resolution_cfg, ontology=ontology)

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
            surface_name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            description="Faculty advisor",
            source_chunk_id="D001_C0000",
            source_doc_id="D001",
            page_numbers=[1],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0001_M001",
            surface_name="Prof. Sharma",
            normalized_name="prof. sharma",
            entity_type="PERSON",
            description="PI on AI grant",
            source_chunk_id="D001_C0001",
            source_doc_id="D001",
            page_numbers=[2],
        ),
        EntityMention(
            kb_id="academic_kb",
            mention_id="D001_C0002_M001",
            surface_name="Dr. Sharma",
            normalized_name="dr. sharma",
            entity_type="PERSON",
            description="Senior supervisor",
            source_chunk_id="D001_C0002",
            source_doc_id="D001",
            page_numbers=[3],
        ),
    ]

    canonicals, mention_to_entity = resolver.resolve_mentions(mentions)

    # 4 mentions must collapse into exactly 2 canonical entities (Rahul and Professor Sharma)
    assert len(canonicals) == 2

    # Entity 1: Rahul
    c_rahul = canonicals[0]
    assert c_rahul.entity_id == "E001"
    assert c_rahul.name == "Rahul"
    assert c_rahul.source_mention_ids == ["D001_C0000_M001"]

    # Entity 2: Professor Sharma
    c_sharma = canonicals[1]
    assert c_sharma.entity_id == "E002"
    assert c_sharma.name == "Professor Sharma"
    assert set(c_sharma.aliases) == {"Prof. Sharma", "Dr. Sharma"}
    assert c_sharma.source_mention_ids == [
        "D001_C0000_M002",
        "D001_C0001_M001",
        "D001_C0002_M001",
    ]
    assert set(c_sharma.source_chunk_ids) == {
        "D001_C0000",
        "D001_C0001",
        "D001_C0002",
    }
    assert "Faculty advisor" in c_sharma.description
    assert "PI on AI grant" in c_sharma.description

    # Mapping
    assert mention_to_entity["D001_C0000_M001"] == "E001"
    assert mention_to_entity["D001_C0000_M002"] == "E002"
    assert mention_to_entity["D001_C0001_M001"] == "E002"
    assert mention_to_entity["D001_C0002_M001"] == "E002"


# --- Test 3: Embedding Candidate Discovery & LLM Ambiguity Verification ---


def test_embedding_discovery_and_llm_verification() -> None:
    """Test embedding discovery with auto-merge, LLM confirmation, and LLM conservative rejection."""
    ontology = DomainOntology(
        domain_name="tech",
        entity_types={"CONCEPT": "Technical and scientific concepts"},
        relationship_types={
            "RELATED_TO": RelationConstraint(name="RELATED_TO", description="Conceptual relation")
        },
    )
    resolution_cfg = ResolutionConfig(
        similarity_candidate_threshold=0.82,
        auto_merge_threshold=0.95,
        use_llm_verification_for_ambiguous=True,
    )

    # Vectors for controlled cosine similarity tests:
    # E001 (NLP): base vector [1.0, 0.0, 0.0, ...]
    # Mention 1 (Near-identical query): similarity 0.98 >= 0.95 (auto-merge)
    # Mention 2 (Ambiguous confirm): angle giving similarity 0.88 -> LLM returns True
    # Mention 3 (Ambiguous reject): angle giving similarity 0.85 -> LLM returns False
    dim = 768

    def mock_embed(text: str) -> List[float]:
        v = [0.0] * dim
        t = text.lower()
        if "natural language processing" in t:
            v[0] = 1.0
        elif "nlp technology" in t:
            # High similarity to NLP: 0.98
            v[0] = 0.98
            v[1] = math.sqrt(1 - 0.98**2)
        elif "computational linguistics" in t:
            # Ambiguous similarity: 0.88
            v[0] = 0.88
            v[1] = math.sqrt(1 - 0.88**2)
        elif "neuro-linguistic programming" in t:
            # Ambiguous similarity: 0.85
            v[0] = 0.85
            v[1] = math.sqrt(1 - 0.85**2)
        else:
            v[2] = 1.0
        return v

    embed_service = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [mock_embed(t) for t in texts],
    )

    # Mock LLM client deciding on ambiguous pairs
    def mock_llm_decision(prompt: str) -> str:
        if "Neuro-Linguistic Programming" in prompt:
            return json.dumps(
                {
                    "is_same_entity": False,
                    "reason": "Pseudoscience model, unrelated to AI NLP",
                }
            )
        elif "Computational Linguistics" in prompt:
            return json.dumps(
                {
                    "is_same_entity": True,
                    "reason": "Synonymous discipline in context",
                }
            )
        return json.dumps({"is_same_entity": False, "reason": "Default separate"})

    llm_client = OllamaLLMClient(custom_generator=mock_llm_decision)

    resolver = ConservativeEntityResolver(
        resolution_config=resolution_cfg,
        ontology=ontology,
        embedding_provider=embed_service,
        llm_client=llm_client,
    )

    mentions = [
        # Mention 1 -> Creates Canonical E001
        EntityMention(
            kb_id="tech_kb",
            mention_id="M1",
            surface_name="Natural Language Processing",
            normalized_name="natural language processing",
            entity_type="CONCEPT",
            description="Field of AI processing human language",
            source_chunk_id="C1",
            source_doc_id="D1",
        ),
        # Mention 2 -> Cosine sim = 0.98 >= 0.95 (Auto-merges into E001)
        EntityMention(
            kb_id="tech_kb",
            mention_id="M2",
            surface_name="NLP Technology",
            normalized_name="nlp technology",
            entity_type="CONCEPT",
            description="Language processing system",
            source_chunk_id="C2",
            source_doc_id="D1",
        ),
        # Mention 3 -> Cosine sim = 0.88 (Ambiguous band, LLM confirms -> merges into E001)
        EntityMention(
            kb_id="tech_kb",
            mention_id="M3",
            surface_name="Computational Linguistics",
            normalized_name="computational linguistics",
            entity_type="CONCEPT",
            description="Linguistic computing",
            source_chunk_id="C3",
            source_doc_id="D1",
        ),
        # Mention 4 -> Cosine sim = 0.85 (Ambiguous band, LLM rejects -> creates E002)
        EntityMention(
            kb_id="tech_kb",
            mention_id="M4",
            surface_name="Neuro-Linguistic Programming",
            normalized_name="neuro-linguistic programming",
            entity_type="CONCEPT",
            description="Alternative communication model",
            source_chunk_id="C4",
            source_doc_id="D1",
        ),
    ]

    canonicals, m_map = resolver.resolve_mentions(mentions)

    assert len(canonicals) == 2
    assert canonicals[0].entity_id == "E001"
    assert canonicals[0].name == "Natural Language Processing"
    assert "NLP Technology" in canonicals[0].aliases
    assert "Computational Linguistics" in canonicals[0].aliases

    assert canonicals[1].entity_id == "E002"
    assert canonicals[1].name == "Neuro-Linguistic Programming"

    assert m_map["M1"] == "E001"
    assert m_map["M2"] == "E001"
    assert m_map["M3"] == "E001"
    assert m_map["M4"] == "E002"


# --- Test 4: Relationship Endpoint Remapping, Aggregation, and DocumentStore ---


def test_relationship_remapping_and_document_store(tmp_path: Path) -> None:
    """Test remapping relationship endpoints to canonical IDs, aggregating duplicates, and DocumentStore persistence."""
    kb_id = "academic_kb"
    ontology = DomainOntology(
        domain_name="academic",
        entity_types={"PERSON": "Faculty and students"},
        relationship_types={
            "WORKS_UNDER": RelationConstraint(name="WORKS_UNDER", description="Supervision"),
            "CO_AUTHORED_WITH": RelationConstraint(name="CO_AUTHORED_WITH", description="Coauthorship"),
            "MENTORS": RelationConstraint(name="MENTORS", description="Mentorship"),
        },
        static_aliases={"Professor Sharma": ["Dr. Sharma"]},
    )
    resolver = ConservativeEntityResolver(ontology=ontology)

    mentions = [
        EntityMention(
            kb_id=kb_id,
            mention_id="M001",
            surface_name="Rahul",
            normalized_name="rahul",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
        ),
        EntityMention(
            kb_id=kb_id,
            mention_id="M002",
            surface_name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            source_chunk_id="C001",
            source_doc_id="D001",
        ),
        EntityMention(
            kb_id=kb_id,
            mention_id="M003",
            surface_name="Dr. Sharma",
            normalized_name="dr. sharma",
            entity_type="PERSON",
            source_chunk_id="C002",
            source_doc_id="D001",
        ),
    ]

    relationships = [
        # R1: Rahul (M001) WORKS_UNDER Professor Sharma (M002) in C001
        Relationship(
            kb_id=kb_id,
            relationship_id="C001_R001",
            source_entity_id="M001",
            target_entity_id="M002",
            relation_type="WORKS_UNDER",
            source_chunk_ids=["C001"],
            source_doc_ids=["D001"],
            evidence_quotes=["Rahul works under Professor Sharma."],
            validation_status=ValidationStatus.VALIDATED,
            uncalibrated_confidence=0.90,
        ),
        # R2: Rahul (M001) WORKS_UNDER Dr. Sharma (M003) in C002 (Duplicate edge after resolution)
        Relationship(
            kb_id=kb_id,
            relationship_id="C002_R001",
            source_entity_id="M001",
            target_entity_id="M003",
            relation_type="WORKS_UNDER",
            source_chunk_ids=["C002"],
            source_doc_ids=["D001"],
            evidence_quotes=["Rahul conducts research under Dr. Sharma."],
            validation_status=ValidationStatus.VALIDATED,
            uncalibrated_confidence=0.95,
        ),
        # R3: Distinct relation type between same pair (Rahul CO_AUTHORED_WITH Professor Sharma)
        Relationship(
            kb_id=kb_id,
            relationship_id="C003_R001",
            source_entity_id="M001",
            target_entity_id="M002",
            relation_type="CO_AUTHORED_WITH",
            source_chunk_ids=["C003"],
            source_doc_ids=["D001"],
            evidence_quotes=["Rahul co-authored a graph paper with Professor Sharma."],
            validation_status=ValidationStatus.VALIDATED,
            uncalibrated_confidence=0.99,
        ),
        # R4: Self-loop after resolution: Professor Sharma (M002) -> Dr. Sharma (M003) (must be dropped)
        Relationship(
            kb_id=kb_id,
            relationship_id="C004_R001",
            source_entity_id="M002",
            target_entity_id="M003",
            relation_type="MENTORS",
            source_chunk_ids=["C004"],
            evidence_quotes=["Professor Sharma is Dr. Sharma."],
            validation_status=ValidationStatus.VALIDATED,
        ),
    ]

    canonicals, m_map, remapped_rels = resolver.resolve_and_remap(mentions, relationships)

    assert len(canonicals) == 2
    assert m_map["M001"] == "E001"
    assert m_map["M002"] == "E002"
    assert m_map["M003"] == "E002"

    # Exactly 2 remapped relationships: aggregated WORKS_UNDER, distinct CO_AUTHORED_WITH. Self-loop dropped.
    assert len(remapped_rels) == 2

    r_work = next(r for r in remapped_rels if r.relation_type == "WORKS_UNDER")
    assert r_work.source_entity_id == "E001"
    assert r_work.target_entity_id == "E002"
    assert set(r_work.source_chunk_ids) == {"C001", "C002"}
    assert r_work.support_count == 2
    assert len(r_work.evidence_quotes) == 2
    assert r_work.uncalibrated_confidence == 0.95

    r_auth = next(r for r in remapped_rels if r.relation_type == "CO_AUTHORED_WITH")
    assert r_auth.source_entity_id == "E001"
    assert r_auth.target_entity_id == "E002"
    assert r_auth.support_count == 1

    # --- DocumentStore Persistence Test ---
    store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    store.save_mention_to_entity(m_map)
    store.save_canonical_entities(canonicals)
    store.save_relationships(remapped_rels)

    assert store.mention_to_entity_path.exists()
    assert store.entities_path.exists()
    assert store.relationships_path.exists()

    # Cold reload
    store2 = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    loaded_map = store2.load_mention_to_entity()
    loaded_entities = store2.load_canonical_entities()
    loaded_relationships = store2.load_relationships()

    assert loaded_map == m_map
    assert len(loaded_entities) == 2
    assert loaded_entities == canonicals
    assert len(loaded_relationships) == 2
    assert loaded_relationships == remapped_rels


def test_strip_honorifics_and_similarity_mismatch() -> None:
    """Test title stripping for varied prefixes and matrix dimension error handling."""
    assert strip_honorifics("Prof. Sharma") == "Sharma"
    assert strip_honorifics("Professor Jane Doe") == "Jane Doe"
    assert strip_honorifics("Dr. Alice Smith") == "Alice Smith"
    assert strip_honorifics("Doctor Bob") == "Bob"
    assert strip_honorifics("Assoc. Prof. Clark") == "Clark"
    assert strip_honorifics("Mr. Wayne") == "Wayne"
    assert strip_honorifics("Ms. Prince") == "Prince"
    assert strip_honorifics("Regular Name") == "Regular Name"

    # Dimension mismatch in similarity matrix
    a = np.ones((2, 768))
    b = np.ones((2, 512))
    with pytest.raises(ValueError, match="Dimension mismatch in cosine similarity"):
        cosine_similarity_matrix(a, b)


def test_document_store_graph_kb_isolation(tmp_path: Path) -> None:
    """Test DocumentStore rejects entities and relationships with mismatched kb_id."""
    store = DocumentStore(kb_id="kb_main", base_dir=tmp_path)

    bad_entity = CanonicalEntity(
        kb_id="kb_other",
        entity_id="E001",
        name="Bad Entity",
        normalized_name="bad entity",
        entity_type="CONCEPT",
        source_chunk_ids=["C1"],
        source_doc_ids=["D1"],
    )
    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_canonical_entities([bad_entity])

    bad_rel = Relationship(
        kb_id="kb_other",
        relationship_id="R001",
        source_entity_id="E001",
        target_entity_id="E002",
        relation_type="RELATES_TO",
        source_chunk_ids=["C1"],
        evidence_quotes=["evidence"],
    )
    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_relationships([bad_rel])


def is_live_ollama_available() -> bool:
    """Check if local Ollama server is running and reachable."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_live_ollama_available(), reason="Live Ollama instance not reachable")
def test_live_ollama_embedding_service() -> None:
    """Optional live test validating embeddinggemma:latest via Ollama /api/embed."""
    service = OllamaEmbeddingService()
    vec = service.embed_text("GraphRAG hybrid architecture evaluation")
    assert len(vec) == 768
    assert all(math.isfinite(x) for x in vec)
