"""Unit and regression tests for Document-Metadata-Based Entity Discovery."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import igraph
import pytest
from pydantic import ValidationError
from src.core.configuration import EngineConfig, RetrievalConfig
from src.core.models import (
    CanonicalEntity,
    Chunk,
    Document,
    Evidence,
    GraphProvenanceType,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    SeedEntitySpec,
)
from src.graph.traversal import GraphTraverser
from src.retrieval.document_title_discoverer import (
    DocumentTitleCandidate,
    DocumentTitleDiscoverer,
    is_generic_title,
    normalize_title,
    punctuation_normalized_title,
)
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedResolutionResult
from src.storage.document_store import DocumentStore


class MockSeedRetriever:
    """Mock seed retriever mapping seed names to canonical entity IDs."""

    def __init__(self, mapping: Dict[str, List[str]]) -> None:
        self.mapping = mapping

    def resolve_seeds(self, seeds: List[SeedEntitySpec]) -> SeedResolutionResult:
        resolved: List[str] = []
        for s in seeds:
            resolved.extend(self.mapping.get(s.name, []))
        return SeedResolutionResult(
            resolved_entity_ids=list(dict.fromkeys(resolved)),
            ambiguous_seeds={},
            unresolved_seeds=[s.name for s in seeds if s.name not in self.mapping],
        )

    def _resolve_single_spec(self, spec: SeedEntitySpec, entities: List[CanonicalEntity]):
        return self.mapping.get(spec.name, []), False


def _create_doc(
    kb_id: str,
    doc_id: str,
    title: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Document:
    meta = dict(metadata or {})
    if title is not None:
        meta["title"] = title
    return Document(
        kb_id=kb_id,
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        source_path=f"/data/{doc_id}.txt",
        content_hash=f"hash_{doc_id}",
        total_pages=1,
        metadata=meta,
    )


def _create_chunk(
    kb_id: str,
    cid: str,
    doc_id: str,
    text: str,
    chunk_index: int = 0,
    scope_id: Optional[str] = None,
) -> Chunk:
    return Chunk(
        kb_id=kb_id,
        chunk_id=cid,
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        page_numbers=[1],
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
        content_hash=f"hash_{cid}",
        scope_id=scope_id,
    )


# 1. Exact document-title match
def test_exact_document_title_match(tmp_path: Path):
    kb_id = "test_kb_exact"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="Esma Sultan Mansion")
    chunk = _create_chunk(kb_id, "D001_C0000", "D001", "Esma Sultan Mansion is a historic mansion.")
    store.save_all(documents=[doc], pages=[], chunks=[chunk])

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    seeds = [SeedEntitySpec(name="Esma Sultan Mansion")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.seed_name == "Esma Sultan Mansion"
    assert cand.matched_title == "Esma Sultan Mansion"
    assert cand.doc_id == "D001"
    assert cand.chunk_ids == ["D001_C0000"]
    assert cand.matching_method == "EXACT_NORMALIZED"
    assert not cand.is_ambiguous
    assert cand.candidate_doc_ids == ["D001"]


# 2. Case and whitespace normalization
def test_case_and_whitespace_normalization(tmp_path: Path):
    kb_id = "test_kb_norm"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D003", title="2014 S/S")
    chunk = _create_chunk(kb_id, "D003_C0000", "D003", "2014 S/S is an album.")
    store.save_all(documents=[doc], pages=[], chunks=[chunk])

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    seeds = [SeedEntitySpec(name="   2014    s/s   ")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 1
    assert candidates[0].doc_id == "D003"
    assert candidates[0].matching_method == "EXACT_NORMALIZED"


# 3. Punctuation normalization
def test_punctuation_normalization(tmp_path: Path):
    kb_id = "test_kb_punct"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D003", title="2014 S/S")
    chunk = _create_chunk(kb_id, "D003_C0000", "D003", "2014 S/S text.")
    store.save_all(documents=[doc], pages=[], chunks=[chunk])

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    seeds = [SeedEntitySpec(name="2014 S S")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 1
    assert candidates[0].doc_id == "D003"
    assert candidates[0].matching_method == "PUNCTUATION_NORMALIZED"


# 4. Multiple documents with identical titles (ambiguity)
def test_multiple_documents_identical_titles_ambiguity(tmp_path: Path):
    kb_id = "test_kb_ambig"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc1 = _create_doc(kb_id, "D001", title="Springfield")
    doc2 = _create_doc(kb_id, "D002", title="Springfield")
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "Springfield Illinois.")
    c2 = _create_chunk(kb_id, "D002_C0", "D002", "Springfield Massachusetts.")
    store.save_all(documents=[doc1, doc2], pages=[], chunks=[c1, c2])

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    seeds = [SeedEntitySpec(name="Springfield")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 2
    for c in candidates:
        assert c.is_ambiguous is True
        assert c.candidate_doc_ids == ["D001", "D002"]


# 5. Ambiguous title matches preserved and capped
def test_ambiguous_title_matches_preserved_and_capped(tmp_path: Path):
    kb_id = "test_kb_cap"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    docs = [
        _create_doc(kb_id, "D001", title="Metropolis"),
        _create_doc(kb_id, "D002", title="Metropolis"),
        _create_doc(kb_id, "D003", title="Metropolis"),
    ]
    chunks = [
        _create_chunk(kb_id, f"D00{i}_C0", f"D00{i}", f"Metropolis {i}")
        for i in range(1, 4)
    ]
    store.save_all(documents=docs, pages=[], chunks=chunks)

    discoverer = DocumentTitleDiscoverer(doc_store=store, max_title_matches_per_seed=2)
    seeds = [SeedEntitySpec(name="Metropolis")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 2
    assert [c.doc_id for c in candidates] == ["D001", "D002"]
    for c in candidates:
        assert c.is_ambiguous is True
        assert c.candidate_doc_ids == ["D001", "D002", "D003"]


# 6. Missing title metadata handled cleanly
def test_missing_title_metadata_handled_cleanly(tmp_path: Path):
    kb_id = "test_kb_missing"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    # Doc 1: No metadata title, but chunk has 'Title: Esma Sultan Mansion'
    doc1 = _create_doc(kb_id, "D001", title=None)
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "Title: Esma Sultan Mansion\nHistoric mansion text.")

    # Doc 2: No metadata title, no chunk header
    doc2 = _create_doc(kb_id, "D002", title=None)
    c2 = _create_chunk(kb_id, "D002_C0", "D002", "No title header here.")

    store.save_all(documents=[doc1, doc2], pages=[], chunks=[c1, c2])

    assert store.get_document_title("D001") == "Esma Sultan Mansion"
    assert store.get_document_title("D002") is None

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    candidates = discoverer.discover_title_candidates([SeedEntitySpec(name="Esma Sultan Mansion")])
    assert len(candidates) == 1
    assert candidates[0].doc_id == "D001"


# 7. Generic titles and fan-out protection
def test_generic_titles_and_fanout_protection(tmp_path: Path):
    kb_id = "test_kb_generic"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc1 = _create_doc(kb_id, "D001", title="The")
    doc2 = _create_doc(kb_id, "D002", title="Document")
    doc3 = _create_doc(kb_id, "D003", title="Esma Sultan Mansion")
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "The text")
    c2 = _create_chunk(kb_id, "D002_C0", "D002", "Doc text")
    c3 = _create_chunk(kb_id, "D003_C0", "D003", "Mansion text")
    store.save_all(documents=[doc1, doc2, doc3], pages=[], chunks=[c1, c2, c3])

    discoverer = DocumentTitleDiscoverer(doc_store=store)

    # 1. Stopword seed rejected
    assert is_generic_title("The") is True
    assert is_generic_title("Document") is True
    assert discoverer.discover_title_candidates([SeedEntitySpec(name="The")]) == []

    # 2. Substrings strictly forbidden
    res = discoverer.discover_title_candidates([SeedEntitySpec(name="Esma Sultan")])
    assert res == []  # "Esma Sultan" != "Esma Sultan Mansion"


# 8. Multiple query seeds
def test_multiple_query_seeds_balancing(tmp_path: Path):
    kb_id = "test_kb_multi"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)

    # D001: Laleli Mosque (also in graph)
    doc1 = _create_doc(kb_id, "D001", title="Laleli Mosque")
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "Laleli Mosque is an Ottoman mosque.")
    ent1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Laleli Mosque",
        normalized_name="laleli mosque",
        entity_type="LOCATION",
        source_chunk_ids=["D001_C0"],
        source_doc_ids=["D001"],
    )

    # D002: Esma Sultan Mansion (NOT in graph)
    doc2 = _create_doc(kb_id, "D002", title="Esma Sultan Mansion")
    c2 = _create_chunk(kb_id, "D002_C0", "D002", "Esma Sultan Mansion is a historical residence.")

    store.save_all(
        documents=[doc1, doc2],
        pages=[],
        chunks=[c1, c2],
        canonical_entities=[ent1],
    )

    seed_retriever = MockSeedRetriever({"Laleli Mosque": ["E001"]})
    graph_retriever = GraphEvidenceRetriever(
        doc_store=store,
        seed_retriever=seed_retriever,
        document_title_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        intent=QueryIntent.LOCAL,
        seed_entities=[
            SeedEntitySpec(name="Laleli Mosque"),
            SeedEntitySpec(name="Esma Sultan Mansion"),
        ],
    )

    evidence = graph_retriever.retrieve(question=plan.question, plan=plan, kb_id=kb_id)
    assert len(evidence) == 2
    cids = [ev.chunk_id for ev in evidence]
    assert "D001_C0" in cids
    assert "D002_C0" in cids

    ev_mansion = next(ev for ev in evidence if ev.chunk_id == "D002_C0")
    assert GraphProvenanceType.DOCUMENT_TITLE_PROVENANCE in ev_mansion.graph_provenance_types
    assert len(ev_mansion.document_title_details) == 1
    assert ev_mansion.document_title_details[0].document_title == "Esma Sultan Mansion"


# 9. Multi-chunk documents
def test_multichunk_document_capped_by_config(tmp_path: Path):
    kb_id = "test_kb_multi_chunk"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="Large Novel")
    chunks = [
        _create_chunk(kb_id, f"D001_C000{i}", "D001", f"Chapter {i} text", chunk_index=i)
        for i in range(5)
    ]
    store.save_all(documents=[doc], pages=[], chunks=chunks)

    discoverer = DocumentTitleDiscoverer(
        doc_store=store,
        max_chunks_per_title_match=3,
        max_title_candidates_per_query=6,
    )
    seeds = [SeedEntitySpec(name="Large Novel")]
    candidates = discoverer.discover_title_candidates(seeds)

    assert len(candidates) == 1
    assert candidates[0].chunk_ids == ["D001_C0000", "D001_C0001", "D001_C0002"]


# 10. Chunk deduplication and path preservation
def test_chunk_deduplication_and_path_preservation(tmp_path: Path):
    kb_id = "test_kb_dedup"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="Esma Sultan Mansion")
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "Esma Sultan Mansion text")
    ent1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Esma Sultan Mansion",
        normalized_name="esma sultan mansion",
        entity_type="LOCATION",
        source_chunk_ids=["D001_C0"],
        source_doc_ids=["D001"],
    )
    store.save_all(documents=[doc], pages=[], chunks=[c1], canonical_entities=[ent1])

    # Direct mock seed retriever resolves E001
    seed_retriever = MockSeedRetriever({"Esma Sultan Mansion": ["E001"]})
    graph_retriever = GraphEvidenceRetriever(
        doc_store=store,
        seed_retriever=seed_retriever,
        document_title_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="Where is Esma Sultan Mansion?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Esma Sultan Mansion")],
    )

    evidence = graph_retriever.retrieve(question=plan.question, plan=plan, kb_id=kb_id)
    # Must appear exactly once
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.chunk_id == "D001_C0"
    assert ev.graph_rank == 1

    # Routes include GRAPH
    assert ProvenanceRoute.GRAPH in ev.routes

    # Provenance types contain both GRAPH_SEED_PROVENANCE and DOCUMENT_TITLE_PROVENANCE
    assert GraphProvenanceType.GRAPH_SEED_PROVENANCE in ev.graph_provenance_types
    assert GraphProvenanceType.DOCUMENT_TITLE_PROVENANCE in ev.graph_provenance_types

    # Supporting paths contains both paths
    assert any("GRAPH_SEED_PROVENANCE" in p for p in ev.supporting_graph_paths)
    assert any("DOCUMENT_TITLE_PROVENANCE" in p for p in ev.supporting_graph_paths)

    # Details populated
    assert len(ev.document_title_details) == 1
    assert ev.document_title_details[0].doc_id == "D001"


# 11. Scope isolation
def test_scope_isolation(tmp_path: Path):
    kb_id = "test_kb_scope"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="Scoped Document")
    c_scope_a = _create_chunk(kb_id, "D001_C0", "D001", "Text A", scope_id="scope_A")
    c_scope_b = _create_chunk(kb_id, "D001_C1", "D001", "Text B", scope_id="scope_B")
    store.save_all(documents=[doc], pages=[], chunks=[c_scope_a, c_scope_b])

    discoverer = DocumentTitleDiscoverer(doc_store=store)
    seeds = [SeedEntitySpec(name="Scoped Document")]

    cand_a = discoverer.discover_title_candidates(seeds, scope_id="scope_A")
    assert len(cand_a) == 1
    assert cand_a[0].chunk_ids == ["D001_C0"]

    cand_b = discoverer.discover_title_candidates(seeds, scope_id="scope_B")
    assert len(cand_b) == 1
    assert cand_b[0].chunk_ids == ["D001_C1"]


# 12. Provenance attribution and EvidenceMerger
def test_provenance_attribution_and_evidence_merger(tmp_path: Path):
    kb_id = "test_kb_attr"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="The Grand Budapest Hotel")
    chunk = _create_chunk(kb_id, "D001_C0", "D001", "Hotel film details.")
    store.save_all(documents=[doc], pages=[], chunks=[chunk])

    seed_retriever = MockSeedRetriever({})
    graph_retriever = GraphEvidenceRetriever(
        doc_store=store,
        seed_retriever=seed_retriever,
        document_title_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="Who directed The Grand Budapest Hotel?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="The Grand Budapest Hotel")],
    )

    evidence = graph_retriever.retrieve(question=plan.question, plan=plan, kb_id=kb_id)
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.supporting_graph_paths == [
        "DOCUMENT_TITLE_PROVENANCE: Document Title [The Grand Budapest Hotel (D001)]"
    ]
    assert len(ev.document_title_details) == 1
    dt = ev.document_title_details[0]
    assert dt.seed_name == "The Grand Budapest Hotel"
    assert dt.document_title == "The Grand Budapest Hotel"
    assert dt.doc_id == "D001"
    assert dt.chunk_ids == ["D001_C0"]
    assert dt.matching_method == "EXACT_NORMALIZED"
    assert dt.is_ambiguous is False

    # EvidenceMerger test
    merged = EvidenceMerger.merge([[ev], [ev]])
    assert len(merged) == 1
    assert len(merged[0].document_title_details) == 1
    assert merged[0].document_title_details[0].doc_id == "D001"


# 13. Feature disabled by default
def test_feature_disabled_by_default(tmp_path: Path):
    kb_id = "test_kb_disabled"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)
    doc = _create_doc(kb_id, "D001", title="Esma Sultan Mansion")
    chunk = _create_chunk(kb_id, "D001_C0", "D001", "Mansion details.")
    store.save_all(documents=[doc], pages=[], chunks=[chunk])

    # Unresolved seed in graph
    seed_retriever = MockSeedRetriever({})
    graph_retriever = GraphEvidenceRetriever(
        doc_store=store,
        seed_retriever=seed_retriever,
        # document_title_discovery_enabled=False by default
    )

    plan = QueryPlan(
        question="Where is Esma Sultan Mansion?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Esma Sultan Mansion")],
    )

    evidence = graph_retriever.retrieve(question=plan.question, plan=plan, kb_id=kb_id)
    # When disabled, unextracted entity returns 0 chunks
    assert evidence == []
    diag = graph_retriever.last_diagnostics.get("document_title_diagnostics", {})
    assert diag.get("enabled") is False


# 14. Backward-compatible configuration loading and bounds
def test_backward_compatible_configuration_loading():
    config = RetrievalConfig()
    assert config.document_title_discovery_enabled is False
    assert config.max_title_matches_per_seed == 2
    assert config.max_chunks_per_title_match == 3
    assert config.max_title_candidates_per_query == 6

    # Validation: ge=1 bounds
    with pytest.raises(ValidationError):
        RetrievalConfig(max_title_matches_per_seed=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(max_chunks_per_title_match=-1)
    with pytest.raises(ValidationError):
        RetrievalConfig(max_title_candidates_per_query=0)


# 15. Interaction with graph and provenance expansion (precedence policy)
def test_interaction_with_graph_and_provenance_expansion(tmp_path: Path):
    kb_id = "test_kb_precedence"
    store = DocumentStore(storage_root=tmp_path, kb_id=kb_id)

    # D001: Entity E001 (Direct seed chunk)
    doc1 = _create_doc(kb_id, "D001", title="Seed Subject")
    c1 = _create_chunk(kb_id, "D001_C0", "D001", "Direct seed content.")
    ent1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Seed Subject",
        normalized_name="seed subject",
        entity_type="CONCEPT",
        source_chunk_ids=["D001_C0"],
        source_doc_ids=["D001"],
    )

    # D002: Discovered via Document Title Only
    doc2 = _create_doc(kb_id, "D002", title="Title Only Document")
    c2 = _create_chunk(kb_id, "D002_C0", "D002", "Title only content.")

    store.save_all(
        documents=[doc1, doc2],
        pages=[],
        chunks=[c1, c2],
        canonical_entities=[ent1],
    )

    seed_retriever = MockSeedRetriever({"Seed Subject": ["E001"]})
    graph_retriever = GraphEvidenceRetriever(
        doc_store=store,
        seed_retriever=seed_retriever,
        document_title_discovery_enabled=True,
    )

    plan = QueryPlan(
        question="Query both subjects",
        intent=QueryIntent.LOCAL,
        seed_entities=[
            SeedEntitySpec(name="Seed Subject"),
            SeedEntitySpec(name="Title Only Document"),
        ],
    )

    evidence = graph_retriever.retrieve(question=plan.question, plan=plan, kb_id=kb_id)
    assert len(evidence) == 2
    # Direct seed chunk (alignment 1.0, min_hop 0) strictly outranks Title Only chunk (alignment 0.80, min_hop 1)
    assert evidence[0].chunk_id == "D001_C0"
    assert evidence[0].graph_rank == 1
    assert GraphProvenanceType.GRAPH_SEED_PROVENANCE in evidence[0].graph_provenance_types

    assert evidence[1].chunk_id == "D002_C0"
    assert evidence[1].graph_rank == 2
    assert GraphProvenanceType.DOCUMENT_TITLE_PROVENANCE in evidence[1].graph_provenance_types
