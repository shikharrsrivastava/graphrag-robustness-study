"""Unit tests for Confidence-Aware Seed Entity Resolution (Task 4).

Tests transparent ranking policy, priority tiers (Tiers 1-6), ambiguity handling,
diagnostic decision generation, synthetic regression fixtures, and backwards compatibility.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pytest
from src.core.models import CanonicalEntity, ResolutionDecision, SeedEntitySpec
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


class MockDeterministicEmbedder:
    """Mock embedder returning controlled unit vectors for embedding fallback tests."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self._preset_vectors = {}

    def set_vector(self, text: str, vector: List[float]) -> None:
        self._preset_vectors[text] = vector

    def embed_text(self, text: str) -> List[float]:
        if text in self._preset_vectors:
            return list(self._preset_vectors[text])
        # Default orthogonal-like vector
        val = 1.0 / (self.dim ** 0.5)
        return [val] * self.dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def _build_test_entity(
    kb_id: str,
    entity_id: str,
    name: str,
    entity_type: str,
    aliases: Optional[List[str]] = None,
    description: str = "",
    chunks: Optional[List[str]] = None,
) -> CanonicalEntity:
    norm_name = name.lower().strip()
    return CanonicalEntity(
        kb_id=kb_id,
        entity_id=entity_id,
        name=name,
        normalized_name=norm_name,
        entity_type=entity_type.upper(),
        description=description or f"Description of {name}",
        aliases=aliases or [],
        source_chunk_ids=chunks or [f"C_{entity_id}"],
        source_doc_ids=[f"D_{entity_id}"],
    )


# --- 1. Exact canonical match with compatible type (Tier 1) ---

def test_exact_canonical_match_compatible_type(tmp_path: Path) -> None:
    kb_id = "test_tier1"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    ent = _build_test_entity(kb_id, "E001", "Apollo 11", "MISSION")
    ds.save_canonical_entities([ent])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Apollo 11", expected_type="MISSION")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    assert len(result.ambiguous_seeds) == 0
    assert len(result.unresolved_seeds) == 0
    assert len(result.resolution_diagnostics) == 1

    diag = result.resolution_diagnostics[0]
    assert diag.seed_name == "Apollo 11"
    assert diag.candidate_id == "E001"
    assert diag.matching_method == "EXACT_CANONICAL"
    assert diag.type_compatible is True
    assert diag.confidence_score == 1.00
    assert diag.similarity_score is None
    assert diag.status == "ACCEPTED"
    assert "compatible type" in diag.reason.lower()


# --- 2. Exact canonical match with conflicting type (Tier 2) ---

def test_exact_canonical_match_conflicting_type(tmp_path: Path) -> None:
    kb_id = "test_tier2"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # In KB, Apollo 11 is labeled SPACECRAFT, but query expects MISSION
    ent = _build_test_entity(kb_id, "E001", "Apollo 11", "SPACECRAFT")
    ds.save_canonical_entities([ent])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Apollo 11", expected_type="MISSION")
    result = retriever.resolve_seeds([spec])

    # Conflicting type must be retained, not discarded!
    assert result.resolved_entity_ids == ["E001"]
    assert len(result.ambiguous_seeds) == 0
    assert len(result.unresolved_seeds) == 0
    assert len(result.resolution_diagnostics) == 1

    diag = result.resolution_diagnostics[0]
    assert diag.candidate_id == "E001"
    assert diag.matching_method == "EXACT_CANONICAL"
    assert diag.type_compatible is False
    assert diag.confidence_score == 0.80
    assert diag.similarity_score is None
    assert diag.status == "ACCEPTED"
    assert "type conflict" in diag.reason.lower()
    assert "expected: MISSION" in diag.reason
    assert "actual: SPACECRAFT" in diag.reason


# --- 3. Exact alias match with conflicting type (Tier 4) ---

def test_exact_alias_match_conflicting_type(tmp_path: Path) -> None:
    kb_id = "test_tier4"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    ent = _build_test_entity(kb_id, "E001", "Project Apollo", "PROJECT", aliases=["Apollo Program"])
    ds.save_canonical_entities([ent])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Apollo Program", expected_type="ORGANIZATION")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    diag = result.resolution_diagnostics[0]
    assert diag.candidate_id == "E001"
    assert diag.matching_method == "EXACT_ALIAS"
    assert diag.type_compatible is False
    assert diag.confidence_score == 0.60
    assert diag.status == "ACCEPTED"
    assert "type conflict" in diag.reason.lower()


# --- 4. Multiple same-name canonical entities: compatible candidate wins ---

def test_multiple_same_name_canonical_entities_compatible_wins(tmp_path: Path) -> None:
    kb_id = "test_multi_compat"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # Three canonical entities named "Delta" with different types
    e1 = _build_test_entity(kb_id, "E001", "Delta", "ORGANIZATION")
    e2 = _build_test_entity(kb_id, "E002", "Delta", "LOCATION")
    e3 = _build_test_entity(kb_id, "E003", "Delta", "LETTER")
    ds.save_canonical_entities([e1, e2, e3])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Delta", expected_type="ORGANIZATION")
    result = retriever.resolve_seeds([spec])

    # Only compatible entity E001 is accepted; E002 and E003 are rejected and recorded
    assert result.resolved_entity_ids == ["E001"]
    assert len(result.ambiguous_seeds) == 0

    decisions = {d.candidate_id: d for d in result.resolution_diagnostics}
    assert decisions["E001"].status == "ACCEPTED"
    assert decisions["E001"].type_compatible is True
    assert decisions["E001"].confidence_score == 1.00

    assert decisions["E002"].status == "REJECTED"
    assert decisions["E002"].type_compatible is False
    assert "superseded" in decisions["E002"].reason.lower()

    assert decisions["E003"].status == "REJECTED"
    assert decisions["E003"].type_compatible is False


# --- 5. Ambiguous exact matches with no compatible candidate ---

def test_ambiguous_exact_matches_no_compatible_candidate(tmp_path: Path) -> None:
    kb_id = "test_ambig_no_compat"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Delta", "ORGANIZATION")
    e2 = _build_test_entity(kb_id, "E002", "Delta", "LOCATION")
    ds.save_canonical_entities([e1, e2])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    # Expected type WORK does not match either ORGANIZATION or LOCATION
    spec = SeedEntitySpec(name="Delta", expected_type="WORK")
    result = retriever.resolve_seeds([spec])

    # Both must be preserved as ambiguous, neither discarded!
    assert set(result.resolved_entity_ids) == {"E001", "E002"}
    assert "Delta" in result.ambiguous_seeds
    assert set(result.ambiguous_seeds["Delta"]) == {"E001", "E002"}

    for d in result.resolution_diagnostics:
        assert d.status == "AMBIGUOUS"
        assert d.type_compatible is False
        assert d.confidence_score == 0.80
        assert "type conflict" in d.reason.lower()


# --- 6. Missing expected_type (behaves identically to baseline) ---

def test_missing_expected_type_baseline_equivalence(tmp_path: Path) -> None:
    kb_id = "test_no_type"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Springfield", "LOCATION")
    ds.save_canonical_entities([e1])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Springfield", expected_type=None)
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    assert len(result.ambiguous_seeds) == 0
    diag = result.resolution_diagnostics[0]
    assert diag.type_compatible is True
    assert diag.confidence_score == 1.00
    assert diag.status == "ACCEPTED"


# --- 7. Case normalization ---

def test_case_normalization(tmp_path: Path) -> None:
    kb_id = "test_case"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "The Rolling Stones", "ORGANIZATION")
    ds.save_canonical_entities([e1])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="tHe RoLLiNg sToNeS", expected_type="ORGANIZATION")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    assert result.resolution_diagnostics[0].status == "ACCEPTED"


# --- 8. Type normalization ---

def test_type_normalization(tmp_path: Path) -> None:
    kb_id = "test_type_norm"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Acme Corp", "ORGANIZATION")
    ds.save_canonical_entities([e1])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    # Lowercase with whitespace
    spec = SeedEntitySpec(name="Acme Corp", expected_type="  organization  ")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    diag = result.resolution_diagnostics[0]
    assert diag.expected_type == "ORGANIZATION"
    assert diag.type_compatible is True
    assert diag.confidence_score == 1.00


# --- 9. Embedding fallback with existing threshold (0.82) ---

def test_embedding_fallback_existing_threshold(tmp_path: Path) -> None:
    kb_id = "test_emb_thresh"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Project Blue Book", "PROJECT")
    ds.save_canonical_entities([e1])

    dim = 768
    # Create two vectors with cosine similarity ~ 0.85
    # Unit vector 1: [1, 0, 0, ...]
    v1 = [0.0] * dim
    v1[0] = 1.0
    # Unit vector 2: [0.85, (1 - 0.85^2)^0.5, 0, ...]
    v2 = [0.0] * dim
    v2[0] = 0.85
    v2[1] = (1.0 - 0.85 ** 2) ** 0.5

    embedder.set_vector("Project Blue Book", v1)
    embedder.set_vector("Blue Book Investigation", v2)

    vstore.upsert_entities([e1], [v1])

    retriever = SeedEntityRetriever(ds, vstore, embedder, similarity_threshold=0.82)
    # Query with non-matching lexical surface form
    spec = SeedEntitySpec(name="Blue Book Investigation", expected_type="PROJECT")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E001"]
    diag = result.resolution_diagnostics[0]
    assert diag.matching_method == "EMBEDDING"
    assert diag.similarity_score is not None
    assert diag.similarity_score >= 0.82
    assert diag.status == "ACCEPTED"


# --- 10. Rejection of low-confidence embedding candidates (< 0.82) ---

def test_rejection_of_low_confidence_embedding_candidates(tmp_path: Path) -> None:
    kb_id = "test_emb_reject"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Project Blue Book", "PROJECT")
    ds.save_canonical_entities([e1])

    dim = 768
    v1 = [0.0] * dim
    v1[0] = 1.0
    # Vector with similarity 0.75 (< 0.82)
    v2 = [0.0] * dim
    v2[0] = 0.75
    v2[1] = (1.0 - 0.75 ** 2) ** 0.5

    embedder.set_vector("Project Blue Book", v1)
    embedder.set_vector("Completely Unrelated Query", v2)

    vstore.upsert_entities([e1], [v1])

    retriever = SeedEntityRetriever(ds, vstore, embedder, similarity_threshold=0.82)
    spec = SeedEntitySpec(name="Completely Unrelated Query", expected_type="PROJECT")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == []
    assert "Completely Unrelated Query" in result.unresolved_seeds

    # Check diagnostic rejection
    statuses = [d.status for d in result.resolution_diagnostics]
    assert "REJECTED" in statuses
    assert "UNRESOLVED" in statuses
    rejected_diag = [d for d in result.resolution_diagnostics if d.status == "REJECTED"][0]
    assert "below threshold" in rejected_diag.reason.lower()


# --- 11. Candidate limit behaviour (max_candidates_per_seed) ---

def test_candidate_limit_behaviour(tmp_path: Path) -> None:
    kb_id = "test_limit"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # Create 8 entities named "Paris" with different types
    entities = [
        _build_test_entity(kb_id, f"E00{i}", "Paris", f"TYPE_{i}")
        for i in range(1, 9)
    ]
    ds.save_canonical_entities(entities)

    retriever = SeedEntityRetriever(ds, vstore, embedder, max_candidates_per_seed=3)
    spec = SeedEntitySpec(name="Paris", expected_type="UNKNOWN")
    result = retriever.resolve_seeds([spec])

    # Exactly 3 candidates returned
    assert len(result.resolved_entity_ids) == 3
    assert len(result.ambiguous_seeds["Paris"]) == 3

    accepted_or_ambig = [d for d in result.resolution_diagnostics if d.status == "AMBIGUOUS"]
    rejected = [d for d in result.resolution_diagnostics if d.status == "REJECTED"]
    assert len(accepted_or_ambig) == 3
    assert len(rejected) == 5
    assert "exceeded max candidates limit" in rejected[0].reason.lower()


def test_max_candidates_per_seed_validation(tmp_path: Path) -> None:
    """Verify that max_candidates_per_seed enforces lower-bound ge=1 validation."""
    from pydantic import ValidationError
    from src.core.configuration import RetrievalConfig

    kb_id = "test_bound_val"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # 1. Default value must remain 5
    retriever_default = SeedEntityRetriever(ds, vstore, embedder)
    assert retriever_default.max_candidates_per_seed == 5

    # 2. Value 1 must be accepted
    retriever_1 = SeedEntityRetriever(ds, vstore, embedder, max_candidates_per_seed=1)
    assert retriever_1.max_candidates_per_seed == 1

    # 3. Value 0 must be rejected
    with pytest.raises(ValueError, match="max_candidates_per_seed must be >= 1"):
        SeedEntityRetriever(ds, vstore, embedder, max_candidates_per_seed=0)

    # 4. Negative values must be rejected
    with pytest.raises(ValueError, match="max_candidates_per_seed must be >= 1"):
        SeedEntityRetriever(ds, vstore, embedder, max_candidates_per_seed=-1)
    with pytest.raises(ValueError, match="max_candidates_per_seed must be >= 1"):
        SeedEntityRetriever(ds, vstore, embedder, max_candidates_per_seed=-10)

    # 5. RetrievalConfig pydantic schema validation
    cfg_valid = RetrievalConfig(max_candidates_per_seed=1)
    assert cfg_valid.max_candidates_per_seed == 1

    with pytest.raises(ValidationError):
        RetrievalConfig(max_candidates_per_seed=0)

    with pytest.raises(ValidationError):
        RetrievalConfig(max_candidates_per_seed=-2)


# --- 12. Multiple seed specifications ---

def test_multiple_seed_specifications(tmp_path: Path) -> None:
    kb_id = "test_multi_specs"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    e1 = _build_test_entity(kb_id, "E001", "Entity Alpha", "TYPE_A")
    e2 = _build_test_entity(kb_id, "E002", "Entity Beta", "TYPE_B")
    ds.save_canonical_entities([e1, e2])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    specs = [
        SeedEntitySpec(name="Entity Alpha", expected_type="TYPE_A"),
        SeedEntitySpec(name="Entity Beta", expected_type="TYPE_X"),  # Conflicting type
        SeedEntitySpec(name="Nonexistent Gamma"),  # Unresolved
    ]
    result = retriever.resolve_seeds(specs)

    assert result.resolved_entity_ids == ["E001", "E002"]
    assert result.unresolved_seeds == ["Nonexistent Gamma"]
    assert len(result.ambiguous_seeds) == 0

    dec_by_seed = {}
    for d in result.resolution_diagnostics:
        dec_by_seed.setdefault(d.seed_name, []).append(d)

    assert dec_by_seed["Entity Alpha"][0].type_compatible is True
    assert dec_by_seed["Entity Alpha"][0].confidence_score == 1.00
    assert dec_by_seed["Entity Beta"][0].type_compatible is False
    assert dec_by_seed["Entity Beta"][0].confidence_score == 0.80
    assert dec_by_seed["Nonexistent Gamma"][0].status == "UNRESOLVED"


# --- 13. Scope isolation & entity identity preservation ---

def test_scope_isolation_and_identity_preservation(tmp_path: Path) -> None:
    kb_id_a = "kb_alpha"
    kb_id_b = "kb_beta"

    ds_a = DocumentStore(kb_id=kb_id_a, processed_dir=tmp_path / "proc_a")
    vstore_a = ChromaVectorStore(kb_id=kb_id_a, kb_vector_store_dir=tmp_path / "vstore_a")

    ds_b = DocumentStore(kb_id=kb_id_b, processed_dir=tmp_path / "proc_b")
    vstore_b = ChromaVectorStore(kb_id=kb_id_b, kb_vector_store_dir=tmp_path / "vstore_b")

    embedder = MockDeterministicEmbedder()

    # Same entity name in both KBs, distinct canonical IDs
    e_a = _build_test_entity(kb_id_a, "E001", "Shared Name", "PERSON", chunks=["C_A1"])
    e_b = _build_test_entity(kb_id_b, "E099", "Shared Name", "PERSON", chunks=["C_B1"])

    ds_a.save_canonical_entities([e_a])
    ds_b.save_canonical_entities([e_b])

    retriever_a = SeedEntityRetriever(ds_a, vstore_a, embedder)
    retriever_b = SeedEntityRetriever(ds_b, vstore_b, embedder)

    res_a = retriever_a.resolve_seeds([SeedEntitySpec(name="Shared Name")])
    res_b = retriever_b.resolve_seeds([SeedEntitySpec(name="Shared Name")])

    # Enforce strict KB isolation and preserved distinct entity IDs
    assert res_a.resolved_entity_ids == ["E001"]
    assert res_b.resolved_entity_ids == ["E099"]
    assert res_a.resolution_diagnostics[0].candidate_id == "E001"
    assert res_b.resolution_diagnostics[0].candidate_id == "E099"


# --- 14. Synthetic regression: Local H pattern (multi-type exact match with conflict) ---

def test_regression_local_h_synthetic_fixture(tmp_path: Path) -> None:
    kb_id = "test_local_h_fixture"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # Matches Q4 KB structure: Local H extracted under 3 types, For Against under FILM
    e_film = _build_test_entity(kb_id, "E032", "Local H", "FILM", chunks=["C_FILM_1", "C_FILM_2"])
    e_org = _build_test_entity(kb_id, "E074", "Local H", "ORGANIZATION", chunks=["C_BAND"])
    e_person = _build_test_entity(kb_id, "E082", "Local H", "PERSON", chunks=["C_PERSON"])
    e_for_against = _build_test_entity(kb_id, "E051", "For Against", "FILM", chunks=["C_FA"])

    ds.save_canonical_entities([e_film, e_org, e_person, e_for_against])

    retriever = SeedEntityRetriever(ds, vstore, embedder)

    # Query with expected_type="WORK" (which conflicts with FILM, ORG, PERSON)
    specs = [
        SeedEntitySpec(name="Local H", expected_type="WORK"),
        SeedEntitySpec(name="For Against", expected_type="WORK"),
    ]
    result = retriever.resolve_seeds(specs)

    # All 3 Local H candidates must be preserved as ambiguous, NOT discarded
    assert set(result.ambiguous_seeds["Local H"]) == {"E032", "E074", "E082"}
    # For Against resolved uniquely to E051 despite type conflict
    assert "For Against" not in result.ambiguous_seeds
    assert "E051" in result.resolved_entity_ids

    # All 4 entity IDs are returned for downstream exploration
    assert set(result.resolved_entity_ids) == {"E032", "E074", "E082", "E051"}

    # Diagnostics show explicit type conflict for all candidates
    for d in result.resolution_diagnostics:
        assert d.type_compatible is False
        assert d.confidence_score == 0.80
        assert "type conflict" in d.reason.lower()


# --- 15. Synthetic regression: Laleli Mosque pattern (exact match with conflicting type) ---

def test_regression_laleli_mosque_synthetic_fixture(tmp_path: Path) -> None:
    kb_id = "test_laleli_fixture"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # Matches Q5 KB structure: Laleli Mosque is BOOK, Laleli is LOCATION
    e_mosque = _build_test_entity(kb_id, "E032", "Laleli Mosque", "BOOK", chunks=["C_MOSQUE"])
    e_area = _build_test_entity(kb_id, "E033", "Laleli", "LOCATION", chunks=["C_MOSQUE"])
    ds.save_canonical_entities([e_mosque, e_area])

    retriever = SeedEntityRetriever(ds, vstore, embedder)

    # Seed expects LOCATION, but Laleli Mosque vertex is BOOK
    spec = SeedEntitySpec(name="Laleli Mosque", expected_type="LOCATION")
    result = retriever.resolve_seeds([spec])

    assert result.resolved_entity_ids == ["E032"]
    assert len(result.ambiguous_seeds) == 0
    diag = result.resolution_diagnostics[0]
    assert diag.candidate_id == "E032"
    assert diag.type_compatible is False
    assert diag.confidence_score == 0.80
    assert "expected: LOCATION" in diag.reason
    assert "actual: BOOK" in diag.reason


# --- 16. Synthetic regression: Missing canonical vertex (Esma Sultan Mansion) cannot be recovered ---

def test_regression_missing_vertex_cannot_be_recovered(tmp_path: Path) -> None:
    kb_id = "test_missing_vertex"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder()

    # KB contains person "Esma Sultan", but NO vertex for "Esma Sultan Mansion"
    e_person = _build_test_entity(kb_id, "E001", "Esma Sultan", "PERSON", chunks=["C_PERSON"])
    ds.save_canonical_entities([e_person])

    retriever = SeedEntityRetriever(ds, vstore, embedder)
    spec = SeedEntitySpec(name="Esma Sultan Mansion", expected_type="LOCATION")
    result = retriever.resolve_seeds([spec])

    # Cannot resolve because no vertex exists and vector similarity < 0.82
    assert result.resolved_entity_ids == []
    assert "Esma Sultan Mansion" in result.unresolved_seeds
    assert any(d.status == "UNRESOLVED" for d in result.resolution_diagnostics)


# --- 17. Task 4 Canonical ID Validation: Phantom IDs absent from DocumentStore are rejected ---

def test_task4_phantom_id_rejected_single_and_mixed(tmp_path: Path) -> None:
    """Task 4 canonical-entity validation: ChromaDB records absent from DocumentStore are rejected.

    Tests both single phantom (rejected -> UNRESOLVED) and mixed (phantom ranked 1st rejected,
    valid canonical ranked 2nd above threshold is accepted).
    """
    kb_id = "test_t4_phantom"
    ds = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = MockDeterministicEmbedder(dim=4)

    retriever = SeedEntityRetriever(
        doc_store=ds,
        vector_store=vstore,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=False,
    )

    # 1. Single phantom case: ChromaDB returns ID absent from DocumentStore with sim >= 0.82
    query_single = "Query Single Phantom"
    embedder.set_vector(query_single, [1.0, 0.0, 0.0, 0.0])

    phantom_single = _build_test_entity(kb_id, "E_PHANTOM_999", "Phantom Book", "BOOK")
    embedder.set_vector("Phantom Book (BOOK): Description of Phantom Book", [1.0, 0.0, 0.0, 0.0])
    vstore.upsert_entities([phantom_single], [embedder.embed_text("Phantom Book (BOOK): Description of Phantom Book")])
    # Deliberately NOT saved to ds!

    res_single = retriever.resolve_seeds([SeedEntitySpec(name=query_single, expected_type="BOOK")])
    assert res_single.resolved_entity_ids == []
    assert query_single in res_single.unresolved_seeds
    phantom_decs = [d for d in res_single.resolution_diagnostics if d.candidate_id == "E_PHANTOM_999"]
    assert len(phantom_decs) == 1
    assert phantom_decs[0].status == "REJECTED"
    assert "absent from knowledge base document store" in phantom_decs[0].reason

    # 2. Mixed case: phantom ID ranked first (sim 0.95), valid canonical ID ranked second (sim 0.88 >= 0.82)
    # The valid canonical entity must be chosen as the resolved seed.
    query_mixed = "Query Mixed Phantom Valid"
    embedder.set_vector(query_mixed, [0.0, 1.0, 0.0, 0.0])

    phantom_mixed = _build_test_entity(kb_id, "E_PHANTOM_888", "Phantom Top", "BOOK")
    valid_mixed = _build_test_entity(kb_id, "E_VALID_111", "Valid Second", "BOOK")
    # Save ONLY the valid entity to DocumentStore
    ds.save_canonical_entities([valid_mixed])

    # Phantom dot product with [0, 1, 0, 0] = 0.95
    sim_phantom = 0.95
    embedder.set_vector("Phantom Top (BOOK): Description of Phantom Top", [0.0, sim_phantom, (1.0 - sim_phantom**2)**0.5, 0.0])
    # Valid dot product with [0, 1, 0, 0] = 0.88
    sim_valid = 0.88
    embedder.set_vector("Valid Second (BOOK): Description of Valid Second", [0.0, sim_valid, (1.0 - sim_valid**2)**0.5, 0.0])

    vstore.upsert_entities(
        [phantom_mixed, valid_mixed],
        [
            embedder.embed_text("Phantom Top (BOOK): Description of Phantom Top"),
            embedder.embed_text("Valid Second (BOOK): Description of Valid Second"),
        ],
    )

    res_mixed = retriever.resolve_seeds([SeedEntitySpec(name=query_mixed, expected_type="BOOK")])
    # Valid canonical entity is accepted
    assert res_mixed.resolved_entity_ids == ["E_VALID_111"]
    # Phantom is rejected
    rej_phantom = [d for d in res_mixed.resolution_diagnostics if d.candidate_id == "E_PHANTOM_888"]
    assert len(rej_phantom) == 1
    assert rej_phantom[0].status == "REJECTED"
    assert "absent from knowledge base document store" in rej_phantom[0].reason
    # Valid is accepted
    acc_valid = [d for d in res_mixed.resolution_diagnostics if d.candidate_id == "E_VALID_111"]
    assert len(acc_valid) == 1
    assert acc_valid[0].status == "ACCEPTED"

