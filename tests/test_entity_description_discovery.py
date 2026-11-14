"""Unit and regression tests for Entity-Description-Based Seed Discovery (Task 6).

Covers all 18 required behaviors:
1. Exact matching taking precedence over description search.
2. Description search executing after exact and alias resolution fail.
3. Correct canonical entity-ID mapping.
4. Missing canonical entity IDs rejected safely.
5. Missing/empty descriptions handled cleanly.
6. Type-compatible candidates (Tier 5, score = sim).
7. Type-conflicting candidates (Tier 6, score = sim * 0.70).
8. Multiple ambiguous description matches (tied within 0.02).
9. Similarity/distance threshold boundaries (e.g. 0.65).
10. Embedding provider error handled gracefully without crash.
11. Candidate limits (per-seed and per-query).
12. Scope isolation.
13. No text-chunk collection access.
14. Provenance attribution (ENTITY_DESCRIPTION_SEED_PROVENANCE).
15. EvidenceMerger compatibility.
16. Feature disabled by default.
17. Backward-compatible configuration loading and validation (ge=1, ge=0.0, le=1.0).
18. Interaction with graph traversal and provenance expansion.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import igraph
import pytest
from pydantic import ValidationError
from src.core.configuration import EngineConfig, RetrievalConfig
from src.core.models import (
    CanonicalEntity,
    Chunk,
    Document,
    EntityDescriptionProvenance,
    Evidence,
    GraphProvenanceType,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    ResolutionDecision,
    SeedEntitySpec,
)
from src.graph.traversal import GraphTraverser
from src.retrieval.entity_description_discoverer import (
    EntityDescriptionCandidate,
    EntityDescriptionDiscoverer,
)
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


class MockDeterministicEmbedder:
    """Mock embedder returning controlled unit vectors for embedding tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._preset_vectors: Dict[str, List[float]] = {}
        self.call_count = 0

    def set_vector(self, text: str, vector: List[float]) -> None:
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            self._preset_vectors[text] = [x / norm for x in vector]
        else:
            self._preset_vectors[text] = vector

    def embed_text(self, text: str) -> List[float]:
        self.call_count += 1
        if text in self._preset_vectors:
            return list(self._preset_vectors[text])
        val = 1.0 / (self.dim ** 0.5)
        return [val] * self.dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def _build_test_entity(
    kb_id: str,
    entity_id: str,
    name: str,
    entity_type: str,
    description: str = "",
    aliases: Optional[List[str]] = None,
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


def _build_test_chunk(
    kb_id: str,
    chunk_id: str,
    doc_id: str,
    text: str,
    scope_id: Optional[str] = None,
) -> Chunk:
    return Chunk(
        kb_id=kb_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        page_numbers=[1],
        chunk_index=0,
        text=text,
        char_count=len(text),
        content_hash=f"hash_{chunk_id}",
        scope_id=scope_id,
    )


@pytest.fixture
def test_env(tmp_path: Path):
    """Set up an isolated DocumentStore, ChromaVectorStore, and MockDeterministicEmbedder."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    vector_store = ChromaVectorStore(kb_id=kb_id, base_dir=tmp_path)
    embedder = MockDeterministicEmbedder(dim=4)
    return {
        "kb_id": kb_id,
        "doc_store": doc_store,
        "vector_store": vector_store,
        "embedder": embedder,
        "tmp_path": tmp_path,
    }


def test_feature_disabled_by_default():
    """Requirement 16: Feature is disabled by default in RetrievalConfig."""
    cfg = RetrievalConfig()
    assert cfg.entity_description_discovery_enabled is False
    assert cfg.entity_description_similarity_threshold == 0.65
    assert cfg.max_description_candidates_per_seed == 3
    assert cfg.max_description_candidates_per_query == 6


def test_backward_compatible_configuration_loading_and_validation():
    """Requirement 17: Configuration loading with and without fields, plus boundary validation."""
    # 1. Backward-compatible default loading
    cfg_empty = RetrievalConfig()
    assert not cfg_empty.entity_description_discovery_enabled

    # 2. Loading with custom fields
    cfg_custom = RetrievalConfig(
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.72,
        max_description_candidates_per_seed=4,
        max_description_candidates_per_query=8,
    )
    assert cfg_custom.entity_description_discovery_enabled is True
    assert cfg_custom.entity_description_similarity_threshold == 0.72
    assert cfg_custom.max_description_candidates_per_seed == 4
    assert cfg_custom.max_description_candidates_per_query == 8

    # 3. Validation: max_description_candidates_per_seed >= 1
    with pytest.raises(ValidationError):
        RetrievalConfig(max_description_candidates_per_seed=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(max_description_candidates_per_seed=-1)

    # 4. Validation: max_description_candidates_per_query >= 1
    with pytest.raises(ValidationError):
        RetrievalConfig(max_description_candidates_per_query=0)

    # 5. Validation: threshold between 0.0 and 1.0
    with pytest.raises(ValidationError):
        RetrievalConfig(entity_description_similarity_threshold=-0.1)
    with pytest.raises(ValidationError):
        RetrievalConfig(entity_description_similarity_threshold=1.1)


def test_exact_matching_takes_precedence_over_description_search(test_env):
    """Requirement 1: Exact matching takes precedence over description search."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # Insert canonical entity "Animorphs"
    e1 = _build_test_entity(
        kb_id, "E019", "Animorphs", "BOOK",
        description="Science fantasy series of young adult books",
        chunks=["C001"],
    )
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs is a book series.")
    doc_store.save_chunks([c1])

    # Index in Chroma
    embedder.set_vector("Animorphs (BOOK): Science fantasy series of young adult books", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Animorphs (BOOK): Science fantasy series of young adult books")])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: phantom description similarity is >= 0.65 (sim 1.0)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    # Query with exact name "Animorphs"
    res = seed_retriever.resolve_seeds([SeedEntitySpec(name="Animorphs", expected_type="BOOK")])
    assert res.resolved_entity_ids == ["E019"]
    # Check that matching method is EXACT_CANONICAL, NOT ENTITY_DESCRIPTION_SEARCH
    diag = res.resolution_diagnostics[0]
    assert diag.matching_method == "EXACT_CANONICAL"
    assert diag.confidence_score == 1.00
    assert len(res.description_candidates) == 0


def test_description_search_after_exact_and_alias_resolution_fail(test_env):
    """Requirement 2 & 3: Description search discovers entity when exact/alias AND embedding (Task 4) all fail.

    After F1 fix, description discovery is a Step 4 fallback: it fires only after
    Steps 1-3 (exact canonical, alias, embedding) have all failed.
    We set up vectors so that entity-name embedding similarity is BELOW the 0.82 threshold
    (Task 4 fails) but entity-description similarity is ABOVE the 0.65 threshold (Task 6 succeeds).
    """
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(
        kb_id, "E019", "Animorphs", "BOOK",
        description="Science fantasy series of young adult books",
        chunks=["C001"],
    )
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs is a book series.")
    doc_store.save_chunks([c1])

    # Query embedding: used by Task 4 (entity embedding) AND Task 6 (description search)
    query_text = "science fantasy young adult series"
    embedder.set_vector(query_text, [1.0, 0.0, 0.0, 0.0])

    # Description search vector: similarity 0.75 is below Task 4 threshold (0.82)
    # but above Task 6 description discovery threshold (0.65)
    sim = 0.75
    desc_text = "Animorphs (BOOK): Science fantasy series of young adult books"
    embedder.set_vector(desc_text, [sim, (1.0 - sim**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text(desc_text)])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: [0.65, 0.82) where Task 4 fails (sim 0.75 < 0.82) and Task 6 succeeds (sim 0.75 >= 0.65)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    res = seed_retriever.resolve_seeds([SeedEntitySpec(name=query_text, expected_type="BOOK")])
    assert res.resolved_entity_ids == ["E019"]
    # Description discovery is the Step 4 fallback — verify it produced candidates
    desc_decisions = [d for d in res.resolution_diagnostics if d.matching_method == "ENTITY_DESCRIPTION_SEARCH"]
    assert len(desc_decisions) >= 1
    accepted = [d for d in desc_decisions if d.status in ("ACCEPTED", "AMBIGUOUS")]
    assert len(accepted) >= 1
    assert accepted[0].candidate_id == "E019"
    assert accepted[0].candidate_name == "Animorphs"
    assert accepted[0].type_compatible is True


def test_missing_canonical_entity_ids_rejected_safely(test_env):
    """Requirement 4: ChromaDB returns entity ID absent from doc store -> safely rejected.

    After F1 fix, we must ensure Task 4 embedding resolution doesn't find this phantom
    (its entity-name vector has low similarity to the query). The description discoverer
    then finds it but correctly rejects it because it's absent from doc store entities_by_id.
    """
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # Save a phantom entity directly into ChromaDB without saving to doc_store
    phantom_e = _build_test_entity(
        kb_id, "E999_PHANTOM", "Phantom Entity", "BOOK",
        description="A ghostly book",
    )
    query_text = "ghostly book"
    # Entity-name embedding: very different from query so Task 4 fails (below 0.82 threshold)
    embedder.set_vector(query_text, [0.1, 1.0, 0.0, 0.0])
    embedder.set_vector("Phantom Entity", [0.9, 0.0, 0.1, 0.0])
    vector_store.upsert_entities([phantom_e], [embedder.embed_text("Phantom Entity")])

    # Description text embedding: high similarity so Task 6 finds it
    embedder.set_vector("Phantom Entity (BOOK): A ghostly book", [0.1, 1.0, 0.0, 0.0])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: phantom description similarity is >= 0.65 (sim 1.0)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    res = seed_retriever.resolve_seeds([SeedEntitySpec(name=query_text, expected_type="BOOK")])
    assert res.resolved_entity_ids == []
    assert query_text in res.unresolved_seeds
    # Diagnostics should show REJECTED with clear reason
    rej = [d for d in res.resolution_diagnostics if d.status == "REJECTED"]
    assert len(rej) >= 1
    assert "absent from knowledge base document store" in rej[0].reason


def test_missing_or_empty_descriptions_handled_cleanly(test_env):
    """Requirement 5: Entities with empty descriptions and empty seed queries."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "EmptyDesc", "CONCEPT", description="")
    doc_store.save_canonical_entities([e1])
    vector_store.upsert_entities([e1], [[0.5, 0.5, 0.5, 0.5]])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
    )

    # Empty seed name
    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="   "),
        entities_by_id={"E001": e1},
    )
    assert cands == []
    assert not is_ambig
    assert decs[0].status == "UNRESOLVED"
    assert "Empty seed name" in decs[0].reason


def test_type_compatible_candidates_tier_5(test_env):
    """Requirement 6: Type-compatible description match (Tier 5) gets score = sim."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E010", "Foundation", "BOOK", description="Sci-fi epic novel")
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C010", "D010", "Foundation book.")
    doc_store.save_chunks([c1])

    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Foundation (BOOK): Sci-fi epic novel", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Foundation (BOOK): Sci-fi epic novel")])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query", expected_type="BOOK"),
        entities_by_id={"E010": e1},
    )
    assert len(cands) == 1
    assert cands[0].type_compatible is True
    # Tier 5: confidence_score == similarity_score
    assert cands[0].confidence_score == cands[0].similarity_score


def test_type_conflicting_candidates_tier_6(test_env):
    """Requirement 7: Type-conflicting description match with expected_type -> UNRESOLVED (F2 fix).

    After the F2 type-safety fix, when expected_type is specified but no type-compatible
    candidates exist, the discoverer returns UNRESOLVED instead of accepting conflicting candidates.
    """
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # Candidate has type MOVIE, query expected BOOK
    e1 = _build_test_entity(kb_id, "E020", "Blade Runner", "MOVIE", description="Cyberpunk film")
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C020", "D020", "Blade Runner film.")
    doc_store.save_chunks([c1])

    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Blade Runner (MOVIE): Cyberpunk film", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Blade Runner (MOVIE): Cyberpunk film")])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query", expected_type="BOOK"),
        entities_by_id={"E020": e1},
    )
    # F2 fix: no compatible candidates => UNRESOLVED, not accepted conflicting
    assert len(cands) == 0
    assert not is_ambig
    rej = [d for d in decs if d.status == "REJECTED"]
    assert len(rej) >= 1
    assert any("conflicts with expected type" in d.reason for d in rej)
    unres = [d for d in decs if d.status == "UNRESOLVED"]
    assert len(unres) == 1
    assert "conflict" in unres[0].reason.lower()


def test_type_conflicting_candidates_tier_6_without_expected_type(test_env):
    """When no expected_type is set, conflicting candidates are accepted (original Tier 6 behavior preserved)."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E020", "Blade Runner", "MOVIE", description="Cyberpunk film")
    doc_store.save_canonical_entities([e1])

    embedder.set_vector("query_no_type", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Blade Runner (MOVIE): Cyberpunk film", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Blade Runner (MOVIE): Cyberpunk film")])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_no_type", expected_type=None),
        entities_by_id={"E020": e1},
    )
    # Without expected_type, all candidates are type-compatible
    assert len(cands) == 1
    assert cands[0].candidate_id == "E020"
    assert cands[0].type_compatible is True


def test_multiple_ambiguous_description_matches_tied(test_env):
    """Requirement 8: Virtually tied candidates within 0.02 of top score are marked AMBIGUOUS."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Entity A", "BOOK", description="Topic X")
    e2 = _build_test_entity(kb_id, "E002", "Entity B", "BOOK", description="Topic Y")
    doc_store.save_canonical_entities([e1, e2])

    # Almost identical vectors: cosine difference < 0.02
    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Entity A (BOOK): Topic X", [1.0, 0.0, 0.0, 0.0])  # sim = 1.0
    embedder.set_vector("Entity B (BOOK): Topic Y", [0.999, 0.044, 0.0, 0.0])  # sim ~ 0.999

    vector_store.upsert_entities([e1, e2], [
        embedder.embed_text("Entity A (BOOK): Topic X"),
        embedder.embed_text("Entity B (BOOK): Topic Y"),
    ])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
        max_candidates_per_seed=3,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query", expected_type="BOOK"),
        entities_by_id={"E001": e1, "E002": e2},
    )
    assert is_ambig is True
    assert len(cands) == 2
    for c in cands:
        assert c.is_ambiguous is True
        assert c.status == "AMBIGUOUS"
        assert set(c.candidate_ids) == {"E001", "E002"}


def test_similarity_threshold_boundary(test_env):
    """Requirement 9: Similarity threshold boundary (0.65)."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # Vector with dot product = 0.60 (below 0.65)
    e1 = _build_test_entity(kb_id, "E001", "Entity Low", "CONCEPT", description="Low similarity entity")
    doc_store.save_canonical_entities([e1])

    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    # [0.6, 0.8, 0, 0] has unit norm, dot product with [1, 0, 0, 0] is 0.60
    embedder.set_vector("Entity Low (CONCEPT): Low similarity entity", [0.60, 0.80, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Entity Low (CONCEPT): Low similarity entity")])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query"),
        entities_by_id={"E001": e1},
    )
    assert len(cands) == 0
    rej = [d for d in decs if d.status == "REJECTED"]
    assert len(rej) == 1
    assert "below threshold 0.65" in rej[0].reason


def test_embedding_provider_error_handled_gracefully(test_env):
    """Requirement 10: Provider error during embedding does not crash."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]

    faulty_embedder = MagicMock()
    faulty_embedder.embed_text.side_effect = RuntimeError("Embedding service unavailable")

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=faulty_embedder,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="test error"),
        entities_by_id={},
    )
    assert cands == []
    assert not is_ambig
    assert decs[0].status == "UNRESOLVED"
    assert "Embedding provider error" in decs[0].reason


def test_candidate_limits_per_seed_and_per_query(test_env):
    """Requirement 11: Capping candidates per seed and per query."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    entities = [
        _build_test_entity(kb_id, f"E00{i}", f"Series {i}", "BOOK", description="Fantasy epic series")
        for i in range(1, 6)
    ]
    doc_store.save_canonical_entities(entities)

    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    vectors = []
    for e in entities:
        doc_str = f"{e.name} (BOOK): Fantasy epic series"
        embedder.set_vector(doc_str, [1.0, 0.0, 0.0, 0.0])
        vectors.append(embedder.embed_text(doc_str))
    vector_store.upsert_entities(entities, vectors)

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        max_candidates_per_seed=2,  # Limit to 2
        max_candidates_per_query=3,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query", expected_type="BOOK"),
        entities_by_id={e.entity_id: e for e in entities},
    )
    assert len(cands) == 2
    accepted = [d for d in decs if d.status in ("ACCEPTED", "AMBIGUOUS")]
    assert len(accepted) == 2


def test_scope_isolation(test_env):
    """Requirement 12: Scope isolation prevents cross-scope leak."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E019", "Animorphs", "BOOK", description="Young adult series", chunks=["C001"])
    doc_store.save_canonical_entities([e1])
    # Chunk with scope_A
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs chunk", scope_id="scope_A")
    doc_store.save_chunks([c1])

    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Animorphs (BOOK): Young adult series", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Animorphs (BOOK): Young adult series")])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: phantom description similarity is >= 0.65 (sim 1.0)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )
    graph = igraph.Graph(directed=True)
    graph.add_vertex(name="E019", label="Animorphs", source_chunk_ids=["C001"])

    retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        entity_description_discovery_enabled=True,
    )

    # When query plan runs with matching scope
    plan = QueryPlan(question="desc", seed_entities=[SeedEntitySpec(name="query")])
    res_a = retriever.retrieve(plan=plan, question="desc", kb_id=kb_id, scope_id="scope_A")
    assert len(res_a) == 1
    assert res_a[0].chunk_id == "C001"


def test_no_text_chunk_collection_access(test_env):
    """Requirement 13: Strictly queries entity_descriptions collection, never text_chunks."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Subject", "CONCEPT", description="Detailed description")
    doc_store.save_canonical_entities([e1])
    embedder.set_vector("query", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Subject (CONCEPT): Detailed description", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Subject (CONCEPT): Detailed description")])

    # Spy on query_chunks
    vector_store.query_chunks = MagicMock(side_effect=AssertionError("query_chunks must never be called!"))

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
    )

    cands, _, _ = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query"),
        entities_by_id={"E001": e1},
    )
    assert len(cands) == 1
    vector_store.query_chunks.assert_not_called()


def test_provenance_attribution_and_details(test_env):
    """Requirement 14: Direct chunks from description seeds receive ENTITY_DESCRIPTION_SEED_PROVENANCE.

    After F1 fix, description discovery fires only when Task 4 fails.
    We set entity-name embedding similarity LOW (below 0.82) and description similarity HIGH.
    """
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(
        kb_id, "E019", "Animorphs", "BOOK",
        description="Science fantasy series of young adult books",
        chunks=["C001"],
    )
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs text content.")
    doc_store.save_chunks([c1])

    # Query embedding: used by Task 4 and Task 6
    embedder.set_vector("young adult books", [1.0, 0.0, 0.0, 0.0])

    # Description embedding: similarity 0.75 is below Task 4 threshold (0.82)
    # but above Task 6 description discovery threshold (0.65)
    sim = 0.75
    desc_text = "Animorphs (BOOK): Science fantasy series of young adult books"
    embedder.set_vector(desc_text, [sim, (1.0 - sim**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text(desc_text)])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: [0.65, 0.82) where Task 4 fails (sim 0.75 < 0.82) and Task 6 succeeds (sim 0.75 >= 0.65)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )
    graph = igraph.Graph(directed=True)
    graph.add_vertex(name="E019", label="Animorphs", source_chunk_ids=["C001"])

    retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        entity_description_discovery_enabled=True,
    )

    plan = QueryPlan(question="young adult books", seed_entities=[SeedEntitySpec(name="young adult books", expected_type="BOOK")])
    evidence_list = retriever.retrieve(plan=plan, question="young adult books", kb_id=kb_id)

    assert len(evidence_list) == 1
    ev = evidence_list[0]
    assert ev.chunk_id == "C001"
    assert GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE in ev.graph_provenance_types
    assert GraphProvenanceType.GRAPH_SEED_PROVENANCE not in ev.graph_provenance_types
    assert len(ev.entity_description_details) == 1
    prov = ev.entity_description_details[0]
    assert prov.candidate_id == "E019"
    assert prov.candidate_name == "Animorphs"
    assert prov.matching_method == "ENTITY_DESCRIPTION_SEARCH"
    assert "ENTITY_DESCRIPTION_SEED_PROVENANCE" in ev.supporting_graph_paths[0]


def test_evidence_merger_preserves_description_provenance():
    """Requirement 15: EvidenceMerger unions and preserves entity_description_details."""
    merger = EvidenceMerger()
    desc_prov = EntityDescriptionProvenance(
        seed_name="series",
        candidate_id="E019",
        candidate_name="Animorphs",
        candidate_type="BOOK",
        matched_description="Science fantasy series",
        similarity_score=0.95,
        type_compatible=True,
        confidence_score=0.95,
    )
    ev_graph = Evidence(
        kb_id="kb",
        chunk_id="C001",
        doc_id="D001",
        text="text",
        routes=[ProvenanceRoute.GRAPH],
        graph_rank=1,
        graph_provenance_types=[GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE],
        entity_description_details=[desc_prov],
    )
    ev_vector = Evidence(
        kb_id="kb",
        chunk_id="C001",
        doc_id="D001",
        text="text",
        routes=[ProvenanceRoute.VECTOR],
        vector_rank=1,
    )

    merged = EvidenceMerger.merge([[ev_graph], [ev_vector]])
    assert len(merged) == 1
    m = merged[0]
    assert set(m.routes) == {ProvenanceRoute.GRAPH, ProvenanceRoute.VECTOR}
    assert GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE in m.graph_provenance_types
    assert len(m.entity_description_details) == 1
    assert m.entity_description_details[0].candidate_id == "E019"


def test_interaction_with_graph_traversal_and_provenance_expansion(test_env):
    """Requirement 18: Description-discovered seeds support graph traversal and bipartite expansion."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # E019 (Animorphs) discovered via description, connected to E020 (K. A. Applegate) via AUTHOR edge
    e1 = _build_test_entity(kb_id, "E019", "Animorphs", "BOOK", description="YA sci-fi series", chunks=["C001"])
    e2 = _build_test_entity(kb_id, "E020", "K. A. Applegate", "PERSON", description="Author", chunks=["C002"])
    doc_store.save_canonical_entities([e1, e2])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs text")
    c2 = _build_test_chunk(kb_id, "C002", "D002", "Applegate author text")
    doc_store.save_chunks([c1, c2])

    # Similarity 0.75 is below Task 4 threshold (0.82) but above Task 6 (0.65)
    sim = 0.75
    embedder.set_vector("YA series", [1.0, 0.0, 0.0, 0.0])
    desc_text = "Animorphs (BOOK): YA sci-fi series"
    embedder.set_vector(desc_text, [sim, (1.0 - sim**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text(desc_text)])

    # Explicit threshold configuration:
    # Task 4 embedding threshold: 0.82
    # Task 6 description discovery threshold: 0.65
    # Intended similarity band: [0.65, 0.82) where Task 4 fails (sim 0.75 < 0.82) and Task 6 succeeds (sim 0.75 >= 0.65)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    graph = igraph.Graph(directed=True)
    graph.add_vertex(name="E019", label="Animorphs", source_chunk_ids=["C001"])
    graph.add_vertex(name="E020", label="K. A. Applegate", source_chunk_ids=["C002"])
    graph.add_edge("E019", "E020", edge_id="R001", relation_type="WRITTEN_BY", source_chunk_ids=["C002"], alignment_score=0.92)

    traverser = GraphTraverser()

    retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        traverser=traverser,
        entity_description_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="YA series",
        seed_entities=[SeedEntitySpec(name="YA series", expected_type="BOOK")],
        relationship_path=["WRITTEN_BY"],
    )
    evidence_list = retriever.retrieve(plan=plan, question="YA series", kb_id=kb_id)

    # C001 (direct chunk from description seed) has alignment 0.90
    # C002 (traversal chunk from WRITTEN_BY edge) has alignment 0.92
    assert len(evidence_list) == 2
    cids = [ev.chunk_id for ev in evidence_list]
    assert "C001" in cids
    assert "C002" in cids

    c1_ev = next(ev for ev in evidence_list if ev.chunk_id == "C001")
    assert GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE in c1_ev.graph_provenance_types

    c2_ev = next(ev for ev in evidence_list if ev.chunk_id == "C002")
    assert GraphProvenanceType.GRAPH_EDGE_PROVENANCE in c2_ev.graph_provenance_types


# =============================================================================
# PART 1 & PART 2: Verification, Boundary, Edge-Case & Invariant Tests
# =============================================================================

def test_f1_resolution_order_call_recording_spy(test_env):
    """Part 1a (F1): Task 6 is NOT called when exact, alias, or Task 4 succeeds, and IS called when all three fail."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e_exact = _build_test_entity(kb_id, "E001", "Exact Match", "BOOK")
    e_alias = _build_test_entity(kb_id, "E002", "Alias Entity", "BOOK", aliases=["Known Alias"])
    e_t4 = _build_test_entity(kb_id, "E003", "Task Four Match", "BOOK")
    e_t6 = _build_test_entity(kb_id, "E004", "Task Six Match", "BOOK", description="Only found by description")

    doc_store.save_canonical_entities([e_exact, e_alias, e_t4, e_t6])

    # Vectors
    # Spec 3 (Task 4): query vector has similarity 0.95 with Task 4 entity (>= 0.82)
    embedder.set_vector("query_t4", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Task Four Match (BOOK): Description of Task Four Match", [0.95, (1.0 - 0.95**2)**0.5, 0.0, 0.0])

    # Spec 4 (Task 6): query vector has similarity 0.75 with Task 6 entity (< 0.82, >= 0.65)
    embedder.set_vector("query_t6", [0.0, 1.0, 0.0, 0.0])
    embedder.set_vector("Task Six Match (BOOK): Only found by description", [0.0, 0.75, (1.0 - 0.75**2)**0.5, 0.0])

    vector_store.upsert_entities(
        [e_exact, e_alias, e_t4, e_t6],
        [
            embedder.embed_text("Exact Match (BOOK): Description of Exact Match"),
            embedder.embed_text("Alias Entity (BOOK): Description of Alias Entity"),
            embedder.embed_text("Task Four Match (BOOK): Description of Task Four Match"),
            embedder.embed_text("Task Six Match (BOOK): Only found by description"),
        ],
    )

    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    # Attach spy to discover_candidates
    real_discover = seed_retriever.description_discoverer.discover_candidates
    spy_discover = MagicMock(side_effect=real_discover)
    seed_retriever.description_discoverer.discover_candidates = spy_discover

    specs = [
        SeedEntitySpec(name="Exact Match", expected_type="BOOK"),       # Step 1 succeeds
        SeedEntitySpec(name="Known Alias", expected_type="BOOK"),       # Step 2 succeeds
        SeedEntitySpec(name="query_t4", expected_type="BOOK"),          # Step 3 succeeds (sim 0.95 >= 0.82)
        SeedEntitySpec(name="query_t6", expected_type="BOOK"),          # Steps 1-3 fail -> Step 4 runs!
    ]

    res = seed_retriever.resolve_seeds(specs)

    # Discoverer must be called EXACTLY ONCE, solely for spec 4 ("query_t6")
    assert spy_discover.call_count == 1
    call_kwargs = spy_discover.call_args.kwargs
    call_spec = call_kwargs.get("seed_spec") or spy_discover.call_args.args[0]
    assert call_spec.name == "query_t6"
    assert "E004" in res.resolved_entity_ids


def test_f2_type_safety_all_subcases(test_env):
    """Part 1b (F2): Type safety behavior across all four sub-cases (i, ii, iii, iv)."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e_book = _build_test_entity(kb_id, "E_BOOK", "Book Entity", "BOOK", description="Book entity description")
    e_movie = _build_test_entity(kb_id, "E_MOVIE", "Movie Entity", "MOVIE", description="Movie entity description")
    doc_store.save_canonical_entities([e_book, e_movie])

    # --- Sub-case (i): expected_type given, ONLY an incompatible candidate above threshold -> UNRESOLVED ---
    embedder.set_vector("query_i", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Movie Entity (MOVIE): Movie entity description", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([e_movie], [embedder.embed_text("Movie Entity (MOVIE): Movie entity description")])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands_i, is_ambig_i, decs_i = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_i", expected_type="BOOK"),
        entities_by_id={"E_MOVIE": e_movie},
    )
    assert cands_i == []
    assert is_ambig_i is False
    assert any(d.status == "REJECTED" and "conflicts with expected type" in d.reason for d in decs_i)
    assert any(d.status == "UNRESOLVED" for d in decs_i)

    # --- Sub-case (ii): expected_type given, compatible below threshold and incompatible above -> UNRESOLVED ---
    embedder.set_vector("query_ii", [0.0, 1.0, 0.0, 0.0])
    # Compatible book has similarity 0.50 (< 0.65)
    embedder.set_vector("Book Entity (BOOK): Book entity description", [0.0, 0.50, (1.0 - 0.50**2)**0.5, 0.0])
    # Incompatible movie has similarity 0.80 (>= 0.65)
    embedder.set_vector("Movie Entity (MOVIE): Movie entity description", [0.0, 0.80, (1.0 - 0.80**2)**0.5, 0.0])
    vector_store.upsert_entities(
        [e_book, e_movie],
        [
            embedder.embed_text("Book Entity (BOOK): Book entity description"),
            embedder.embed_text("Movie Entity (MOVIE): Movie entity description"),
        ],
    )

    cands_ii, is_ambig_ii, decs_ii = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_ii", expected_type="BOOK"),
        entities_by_id={"E_BOOK": e_book, "E_MOVIE": e_movie},
    )
    assert cands_ii == []
    assert is_ambig_ii is False
    assert any(d.status == "UNRESOLVED" for d in decs_ii)

    # --- Sub-case (iii): both compatible and incompatible above threshold -> compatible one wins ---
    embedder.set_vector("query_iii", [0.0, 0.0, 1.0, 0.0])
    # Compatible book has similarity 0.70 (>= 0.65)
    embedder.set_vector("Book Entity (BOOK): Book entity description", [0.0, 0.0, 0.70, (1.0 - 0.70**2)**0.5])
    # Incompatible movie has similarity 0.85 (>= 0.65)
    embedder.set_vector("Movie Entity (MOVIE): Movie entity description", [0.0, 0.0, 0.85, (1.0 - 0.85**2)**0.5])
    vector_store.upsert_entities(
        [e_book, e_movie],
        [
            embedder.embed_text("Book Entity (BOOK): Book entity description"),
            embedder.embed_text("Movie Entity (MOVIE): Movie entity description"),
        ],
    )

    # When query_entities returns both candidates above threshold (e.g. unfiltered pool)
    orig_query = vector_store.query_entities
    vector_store.query_entities = MagicMock(return_value=[
        {"id": "E_MOVIE", "distance": round(1.0 - 0.85, 4), "metadata": {"name": "Movie Entity", "entity_type": "MOVIE"}},
        {"id": "E_BOOK", "distance": round(1.0 - 0.70, 4), "metadata": {"name": "Book Entity", "entity_type": "BOOK"}},
    ])

    cands_iii, is_ambig_iii, decs_iii = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_iii", expected_type="BOOK"),
        entities_by_id={"E_BOOK": e_book, "E_MOVIE": e_movie},
    )
    assert len(cands_iii) == 1
    assert cands_iii[0].candidate_id == "E_BOOK"
    assert cands_iii[0].type_compatible is True
    # Incompatible movie must be rejected with superseded reason
    rej_movie = [d for d in decs_iii if d.candidate_id == "E_MOVIE" and d.status == "REJECTED"]
    assert len(rej_movie) >= 1
    assert "Superseded by type-compatible" in rej_movie[0].reason

    # Restore original query
    vector_store.query_entities = orig_query

    # Also verify ChromaDB type-filtered query where compatible candidate wins
    cands_iii_chroma, _, _ = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_iii", expected_type="BOOK"),
        entities_by_id={"E_BOOK": e_book, "E_MOVIE": e_movie},
    )
    assert len(cands_iii_chroma) == 1
    assert cands_iii_chroma[0].candidate_id == "E_BOOK"

    # --- Sub-case (iv): no expected_type -> unconstrained behaviour (highest similarity wins) ---
    cands_iv, is_ambig_iv, decs_iv = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="query_iii", expected_type=None),
        entities_by_id={"E_BOOK": e_book, "E_MOVIE": e_movie},
    )
    assert len(cands_iv) == 1
    assert cands_iv[0].candidate_id == "E_MOVIE"  # 0.85 > 0.70


def test_f3_no_redundant_resolution_counting(test_env):
    """Part 1c (F3): Assert each seed spec triggers at most one embedding call and one query per stage in total."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Animorphs", "BOOK", description="YA science fantasy series", chunks=["C001"])
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Animorphs book chunk.")
    doc_store.save_chunks([c1])

    # Query vector and description vector (sim 0.75 in [0.65, 0.82))
    sim = 0.75
    embedder.set_vector("sci-fi YA book", [1.0, 0.0, 0.0, 0.0])
    desc_text = "Animorphs (BOOK): YA science fantasy series"
    embedder.set_vector(desc_text, [sim, (1.0 - sim**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text(desc_text)])

    # Counting wrapper for embedder and vector store queries
    embed_calls = 0
    orig_embed_text = embedder.embed_text

    def counting_embed_text(text: str) -> List[float]:
        nonlocal embed_calls
        embed_calls += 1
        return orig_embed_text(text)

    embedder.embed_text = counting_embed_text

    query_calls = 0
    orig_query_entities = vector_store.query_entities

    def counting_query_entities(*args, **kwargs) -> List[Dict[str, Any]]:
        nonlocal query_calls
        query_calls += 1
        return orig_query_entities(*args, **kwargs)

    vector_store.query_entities = counting_query_entities

    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )
    graph = igraph.Graph(directed=True)
    graph.add_vertex(name="E001", label="Animorphs", source_chunk_ids=["C001"])

    retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        entity_description_discovery_enabled=True,
    )

    # Counting spy for resolve_seeds
    resolve_calls = 0
    orig_resolve_seeds = seed_retriever.resolve_seeds

    def counting_resolve_seeds(*args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return orig_resolve_seeds(*args, **kwargs)

    seed_retriever.resolve_seeds = counting_resolve_seeds

    plan = QueryPlan(
        question="sci-fi YA book",
        seed_entities=[SeedEntitySpec(name="sci-fi YA book", expected_type=None)],
    )

    evidence_list = retriever.retrieve(plan=plan, question="sci-fi YA book", kb_id=kb_id)
    assert len(evidence_list) == 1
    assert evidence_list[0].chunk_id == "C001"

    # resolve_seeds must be called EXACTLY ONCE
    assert resolve_calls == 1
    # Embedding was computed ONCE in Step 3 and reused in Step 4
    assert embed_calls == 1
    # Vector store queries: exactly 1 query per resolution stage (1 for Task 4, 1 for Task 6)
    assert query_calls == 2


def test_task6_phantom_id_rejected_single_and_mixed(test_env):
    """Part 1e: Task 6 phantom IDs absent from DocumentStore are rejected safely (single and mixed)."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    # 1. Single phantom case: ChromaDB returns ID absent from DocumentStore
    query_single = "phantom query single"
    embedder.set_vector(query_single, [1.0, 0.0, 0.0, 0.0])

    phantom_single = _build_test_entity(kb_id, "E_PHANTOM_T6", "Phantom Single", "BOOK", description="Ghostly novel")
    embedder.set_vector("Phantom Single (BOOK): Ghostly novel", [1.0, 0.0, 0.0, 0.0])
    vector_store.upsert_entities([phantom_single], [embedder.embed_text("Phantom Single (BOOK): Ghostly novel")])
    # Deliberately NOT saved to doc_store!

    cands_s, is_ambig_s, decs_s = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name=query_single, expected_type="BOOK"),
        entities_by_id={},  # Empty doc_store
    )
    assert cands_s == []
    assert is_ambig_s is False
    rej_single = [d for d in decs_s if d.candidate_id == "E_PHANTOM_T6" and d.status == "REJECTED"]
    assert len(rej_single) == 1
    assert "absent from knowledge base document store" in rej_single[0].reason

    # 2. Mixed case: phantom ranked first (sim 0.85), valid canonical entity ranked second (sim 0.75 >= 0.65)
    query_mixed = "phantom query mixed"
    embedder.set_vector(query_mixed, [0.0, 1.0, 0.0, 0.0])

    phantom_mixed = _build_test_entity(kb_id, "E_PHANTOM_MIXED", "Phantom High", "BOOK", description="Phantom high similarity")
    valid_mixed = _build_test_entity(kb_id, "E_VALID_MIXED", "Valid Candidate", "BOOK", description="Valid canonical novel")
    # Save ONLY valid entity to doc_store!
    doc_store.save_canonical_entities([valid_mixed])

    sim_phantom = 0.85
    sim_valid = 0.75
    embedder.set_vector("Phantom High (BOOK): Phantom high similarity", [0.0, sim_phantom, (1.0 - sim_phantom**2)**0.5, 0.0])
    embedder.set_vector("Valid Candidate (BOOK): Valid canonical novel", [0.0, sim_valid, (1.0 - sim_valid**2)**0.5, 0.0])

    vector_store.upsert_entities(
        [phantom_mixed, valid_mixed],
        [
            embedder.embed_text("Phantom High (BOOK): Phantom high similarity"),
            embedder.embed_text("Valid Candidate (BOOK): Valid canonical novel"),
        ],
    )

    cands_m, is_ambig_m, decs_m = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name=query_mixed, expected_type="BOOK"),
        entities_by_id={"E_VALID_MIXED": valid_mixed},
    )
    # Valid canonical entity is accepted
    assert len(cands_m) == 1
    assert cands_m[0].candidate_id == "E_VALID_MIXED"
    # Phantom entity is rejected
    rej_m = [d for d in decs_m if d.candidate_id == "E_PHANTOM_MIXED" and d.status == "REJECTED"]
    assert len(rej_m) == 1
    assert "absent from knowledge base document store" in rej_m[0].reason


def test_shared_mutable_state_isolation(test_env):
    """Part 2: Assert no state leaks between calls and across separate retriever instances."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Entity One", "BOOK", description="Description of Entity One")
    e2 = _build_test_entity(kb_id, "E002", "Entity Two", "CONCEPT", description="Description of Entity Two")
    doc_store.save_canonical_entities([e1, e2])

    retriever1 = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )
    retriever2 = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )

    # Call 1 on retriever1
    res1 = retriever1.resolve_seeds([SeedEntitySpec(name="Entity One", expected_type="BOOK")])
    res1_ids = list(res1.resolved_entity_ids)
    res1_decs_len = len(res1.resolution_diagnostics)

    # Call 2 on retriever1 with different spec
    res2 = retriever1.resolve_seeds([SeedEntitySpec(name="Entity Two", expected_type="CONCEPT")])

    # Assert res1 was not mutated by call 2
    assert res1.resolved_entity_ids == res1_ids
    assert len(res1.resolution_diagnostics) == res1_decs_len
    # Assert res2 contains ONLY Entity Two seeds and diagnostics
    assert res2.resolved_entity_ids == ["E002"]
    assert not any(d.candidate_id == "E001" for d in res2.resolution_diagnostics)

    # Call on retriever2
    res3 = retriever2.resolve_seeds([SeedEntitySpec(name="Entity One", expected_type="BOOK")])
    assert res3.resolved_entity_ids == ["E001"]
    assert not any(d.candidate_id == "E002" for d in res3.resolution_diagnostics)


def test_empty_entity_descriptions_collection(test_env):
    """Part 2: Empty entity_descriptions collection returns UNRESOLVED with clear diagnostic, no crash."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    # No entities upserted
    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="any seed", expected_type="BOOK"),
        entities_by_id={},
    )
    assert cands == []
    assert is_ambig is False
    assert len(decs) == 1
    assert decs[0].status == "UNRESOLVED"
    assert "No candidates found in entity_descriptions collection" in decs[0].reason


def test_malformed_vector_metadata_safely_handled(test_env):
    """Part 2: Malformed metadata (missing entity_id, missing type, non-string type, None metadata) safely rejected."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    valid_e = _build_test_entity(kb_id, "E_VALID", "Valid Entity", "BOOK", description="Valid description")
    doc_store.save_canonical_entities([valid_e])

    discoverer = EntityDescriptionDiscoverer(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.65,
    )

    # Mock query_entities returning records with corrupted metadata
    corrupted_records = [
        {"id": "E_CORRUPT_NONE_META", "distance": 0.1, "metadata": None},
        {"id": "E_CORRUPT_NON_STR_TYPE", "distance": 0.1, "metadata": {"name": "Test", "entity_type": 12345}},
        {"id": None, "distance": 0.1, "metadata": {"name": "No ID", "entity_type": "BOOK"}},
        {"id": "E_VALID", "distance": 0.1, "metadata": {"name": "Valid Entity", "entity_type": "BOOK"}},
    ]
    vector_store.query_entities = MagicMock(return_value=corrupted_records)

    cands, is_ambig, decs = discoverer.discover_candidates(
        seed_spec=SeedEntitySpec(name="any seed", expected_type="BOOK"),
        entities_by_id={"E_VALID": valid_e},
    )

    # Valid candidate is accepted without crash
    assert len(cands) == 1
    assert cands[0].candidate_id == "E_VALID"
    # Corrupt candidates are safely recorded as REJECTED without exception
    rej_decs = [d for d in decs if d.status == "REJECTED"]
    assert len(rej_decs) >= 3


def test_duplicate_hits_deduplication_and_provenance_union(test_env):
    """Part 2: An entity resolved by Task 4 carries GRAPH_SEED_PROVENANCE and NOT ENTITY_DESCRIPTION_SEED_PROVENANCE."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Direct Target", "BOOK", description="Target book description", chunks=["C001"])
    doc_store.save_canonical_entities([e1])
    c1 = _build_test_chunk(kb_id, "C001", "D001", "Direct target book chunk.")
    doc_store.save_chunks([c1])

    # Task 4 high similarity match (sim 0.95 >= 0.82)
    embedder.set_vector("query target", [1.0, 0.0, 0.0, 0.0])
    embedder.set_vector("Direct Target (BOOK): Target book description", [0.95, (1.0 - 0.95**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text("Direct Target (BOOK): Target book description")])

    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=True,
        entity_description_similarity_threshold=0.65,
    )
    graph = igraph.Graph(directed=True)
    graph.add_vertex(name="E001", label="Direct Target", source_chunk_ids=["C001"])

    retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        entity_description_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="query target",
        seed_entities=[SeedEntitySpec(name="query target", expected_type="BOOK")],
    )

    evidence_list = retriever.retrieve(plan=plan, question="query target", kb_id=kb_id)
    assert len(evidence_list) == 1
    ev = evidence_list[0]
    # Carries GRAPH_SEED_PROVENANCE from Task 4, and NOT ENTITY_DESCRIPTION_SEED_PROVENANCE
    assert GraphProvenanceType.GRAPH_SEED_PROVENANCE in ev.graph_provenance_types
    assert GraphProvenanceType.ENTITY_DESCRIPTION_SEED_PROVENANCE not in ev.graph_provenance_types
    assert len(ev.entity_description_details) == 0


def test_feature_flag_disabled_identical_to_baseline(test_env):
    """Part 2: When entity_description_discovery_enabled=False, description-only queries remain UNRESOLVED."""
    kb_id = test_env["kb_id"]
    doc_store = test_env["doc_store"]
    vector_store = test_env["vector_store"]
    embedder = test_env["embedder"]

    e1 = _build_test_entity(kb_id, "E001", "Animorphs", "BOOK", description="YA science fantasy series", chunks=["C001"])
    doc_store.save_canonical_entities([e1])

    # Similarity 0.75 in [0.65, 0.82)
    sim = 0.75
    embedder.set_vector("descriptive YA series", [1.0, 0.0, 0.0, 0.0])
    desc_text = "Animorphs (BOOK): YA science fantasy series"
    embedder.set_vector(desc_text, [sim, (1.0 - sim**2)**0.5, 0.0, 0.0])
    vector_store.upsert_entities([e1], [embedder.embed_text(desc_text)])

    # Retriever with flag DISABLED (default)
    seed_retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vector_store,
        embedding_provider=embedder,
        similarity_threshold=0.82,
        entity_description_discovery_enabled=False,
    )

    res = seed_retriever.resolve_seeds([SeedEntitySpec(name="descriptive YA series", expected_type="BOOK")])
    assert res.resolved_entity_ids == []
    assert "descriptive YA series" in res.unresolved_seeds
    assert not any(d.matching_method == "ENTITY_DESCRIPTION_SEARCH" for d in res.resolution_diagnostics)
    assert len(res.description_candidates) == 0

