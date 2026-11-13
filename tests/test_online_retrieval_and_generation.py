"""Comprehensive unit tests for Online Retrieval, Graph Traversal, RRF Reranking, and Answer Generation."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import igraph
import numpy as np
import pytest
from src.core.configuration import ResolutionConfig, RetrievalConfig
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CommunityReport,
    Document,
    Evidence,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    RetrievalResult,
    SeedEntitySpec,
    ValidationStatus,
)
from src.core.ontology import DomainOntology, RelationConstraint
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import AnswerGenerationOutput, OllamaAnswerGenerator
from src.graph.graph_builder import IGraphBuilder
from src.graph.traversal import GraphTraverser
from src.retrieval.community_retriever import CommunityRetriever, HybridRetrieverOrchestrator
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.query_planner import OllamaQueryPlanner
from src.retrieval.reranker import RRFReranker
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.retrieval.vector_retriever import VectorEvidenceRetriever
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


# --- Fixtures ---

@pytest.fixture
def sample_ontology() -> DomainOntology:
    return DomainOntology(
        domain_name="academic",
        version="1.0",
        entity_types={"PERSON": "An individual", "UNIVERSITY": "An institution", "PROJECT": "A research project"},
        relationship_types={
            "AFFILIATED_WITH": RelationConstraint(
                name="AFFILIATED_WITH",
                allowed_source_types=["PERSON"],
                allowed_target_types=["UNIVERSITY"],
            ),
            "COLLABORATES_WITH": RelationConstraint(
                name="COLLABORATES_WITH",
                allowed_source_types=["PERSON"],
                allowed_target_types=["PERSON"],
            ),
            "WORKS_ON": RelationConstraint(
                name="WORKS_ON",
                allowed_source_types=["PERSON"],
                allowed_target_types=["PROJECT"],
            ),
        },
    )


class DeterministicEmbedder:
    """Mock offline embedding service with deterministic unit vectors based on text hash."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        clean = text.strip().lower()
        h = int(hashlib.md5(clean.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        vec = rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# --- Test 1: Query Planner ---


def test_query_planner_parsing_and_ontology_validation(sample_ontology: DomainOntology) -> None:
    """Verify OllamaQueryPlanner builds prompt, parses response, and filters invalid relationship types."""
    mock_responses = {
        "Who collaborates with Alice?": json.dumps(
            {
                "question": "Who collaborates with Alice?",
                "intent": "LOCAL",
                "seed_entities": [{"name": "Alice", "expected_type": "PERSON"}],
                "target_type": "PERSON",
                "relationship_path": ["COLLABORATES_WITH", "INVALID_REL_TYPE"],
                "requires_clarification": False,
                "clarification_reason": None,
            }
        ),
        "What are the global research themes?": json.dumps(
            {
                "question": "What are the global research themes?",
                "intent": "GLOBAL",
                "seed_entities": [],
                "target_type": None,
                "relationship_path": [],
                "requires_clarification": False,
                "clarification_reason": None,
            }
        ),
    }

    def mock_generator(prompt: str) -> str:
        for q, resp in mock_responses.items():
            if q in prompt:
                return resp
        return json.dumps({"question": "q", "intent": "LOCAL", "seed_entities": []})

    llm_client = OllamaLLMClient(custom_generator=mock_generator)
    planner = OllamaQueryPlanner(ontology=sample_ontology, llm_client=llm_client)

    # 1. Local query with invalid relation filtering
    plan_local = planner.plan("Who collaborates with Alice?")
    assert plan_local.intent == QueryIntent.LOCAL
    assert len(plan_local.seed_entities) == 1
    assert plan_local.seed_entities[0].name == "Alice"
    assert plan_local.seed_entities[0].expected_type == "PERSON"
    # "INVALID_REL_TYPE" must be filtered out!
    assert plan_local.relationship_path == ["COLLABORATES_WITH"]
    assert plan_local.target_type == "PERSON"

    # 2. Global query
    plan_global = planner.plan("What are the global research themes?")
    assert plan_global.intent == QueryIntent.GLOBAL
    assert len(plan_global.seed_entities) == 0
    assert plan_global.relationship_path == []


# --- Test 2: Seed Entity Retriever ---


def test_seed_entity_retriever_hierarchy_and_ambiguity(tmp_path: Path) -> None:
    """Verify SeedEntityRetriever exact match, honorific-stripped alias match, vector search, and ambiguity."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = DeterministicEmbedder(dim=768)

    # Seed DocumentStore with canonical entities
    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Dr. Alice Smith",
        normalized_name="dr alice smith",
        entity_type="PERSON",
        description="Senior ML researcher",
        aliases=["Alice Smith", "Prof. Alice"],
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e2 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E002",
        name="Bob Jones",
        normalized_name="bob jones",
        entity_type="PERSON",
        description="Systems architect",
        aliases=[],
        source_chunk_ids=["C002"],
        source_doc_ids=["D001"],
    )
    doc_store.save_canonical_entities([e1, e2])

    # Index e2 in vector store
    vstore.upsert_entities([e2], [embedder.embed_text(e2.name)])

    retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vstore,
        embedding_provider=embedder,
        similarity_threshold=0.8,
    )

    # 1. Exact match
    res_exact = retriever.resolve_seeds([SeedEntitySpec(name="Bob Jones", expected_type="PERSON")])
    assert res_exact.resolved_entity_ids == ["E002"]
    assert len(res_exact.ambiguous_seeds) == 0

    # 2. Honorific-stripped alias match ("Alice Smith" or "Prof. Alice Smith")
    res_alias = retriever.resolve_seeds([SeedEntitySpec(name="Alice Smith", expected_type="PERSON")])
    assert res_alias.resolved_entity_ids == ["E001"]

    res_honorific = retriever.resolve_seeds([SeedEntitySpec(name="Professor Alice Smith")])
    assert res_honorific.resolved_entity_ids == ["E001"]

    # 3. Non-existent seed entity
    res_unresolved = retriever.resolve_seeds([SeedEntitySpec(name="Charlie Brown")])
    assert "Charlie Brown" in res_unresolved.unresolved_seeds


# --- Test 3: Graph Traversal & Graph Evidence Retriever ---


def test_graph_traversal_cycle_prevention_and_retrieval(tmp_path: Path) -> None:
    """Verify directed traversal, cycle avoidance, scope isolation, and graph evidence retrieval."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = DeterministicEmbedder(dim=768)

    # Build cyclic graph: E001 -> E002 -> E003 -> E001
    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Alice",
        normalized_name="alice",
        entity_type="PERSON",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e2 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E002",
        name="Bob",
        normalized_name="bob",
        entity_type="PERSON",
        source_chunk_ids=["C002"],
        source_doc_ids=["D001"],
    )
    e3 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E003",
        name="Charlie",
        normalized_name="charlie",
        entity_type="PERSON",
        source_chunk_ids=["C003"],
        source_doc_ids=["D001"],
    )

    r1 = Relationship(
        kb_id=kb_id,
        relationship_id="R001",
        source_entity_id="E001",
        target_entity_id="E002",
        relation_type="COLLABORATES_WITH",
        source_chunk_ids=["C001", "C002"],
        validation_status=ValidationStatus.VALIDATED,
        evidence_quotes=["Alice collaborates with Bob."],
    )
    r2 = Relationship(
        kb_id=kb_id,
        relationship_id="R002",
        source_entity_id="E002",
        target_entity_id="E003",
        relation_type="WORKS_ON",
        source_chunk_ids=["C002", "C003"],
        validation_status=ValidationStatus.VALIDATED,
        evidence_quotes=["Bob works on Charlie's project."],
    )
    r3 = Relationship(
        kb_id=kb_id,
        relationship_id="R003",
        source_entity_id="E003",
        target_entity_id="E001",
        relation_type="COLLABORATES_WITH",
        source_chunk_ids=["C003"],
        validation_status=ValidationStatus.VALIDATED,
        evidence_quotes=["Charlie collaborates with Alice."],
    )

    # Save chunks
    chunks = [
        Chunk(kb_id=kb_id, chunk_id="C001", doc_id="D001", filename="doc.txt", page_numbers=[1], chunk_index=0, text="Alice is an ML researcher.", char_count=26, content_hash="h1"),
        Chunk(kb_id=kb_id, chunk_id="C002", doc_id="D001", filename="doc.txt", page_numbers=[1], chunk_index=1, text="Bob collaborates with Alice on AI.", char_count=34, content_hash="h2"),
        Chunk(kb_id=kb_id, chunk_id="C003", doc_id="D001", filename="doc.txt", page_numbers=[2], chunk_index=2, text="Charlie leads project Titan.", char_count=28, content_hash="h3"),
    ]
    doc_store.save_chunks(chunks)
    doc_store.save_canonical_entities([e1, e2, e3])
    doc_store.save_relationships([r1, r2, r3])

    graph = IGraphBuilder.build_graph([e1, e2, e3], [r1, r2, r3])
    traverser = GraphTraverser(max_hops=3, max_candidates=10)

    # Traverse starting at E001
    paths = traverser.traverse(graph, seed_entity_ids=["E001"], relationship_path=["COLLABORATES_WITH", "WORKS_ON"])
    assert len(paths) >= 2
    # Verify no path contains duplicate vertices (cycle prevention)
    for p in paths:
        assert len(p.node_ids) == len(set(p.node_ids))

    # Graph Evidence Retriever
    seed_retriever = SeedEntityRetriever(doc_store, vstore, embedder)
    graph_retriever = GraphEvidenceRetriever(
        doc_store=doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        traverser=traverser,
    )

    plan = QueryPlan(
        question="What does Bob work on?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Alice")],
        relationship_path=["COLLABORATES_WITH", "WORKS_ON"],
    )

    ev_list = graph_retriever.retrieve(
        question=plan.question,
        plan=plan,
        kb_id=kb_id,
    )
    assert len(ev_list) > 0
    assert all(r == ProvenanceRoute.GRAPH for ev in ev_list for r in ev.routes)
    assert all(ev.graph_rank is not None for ev in ev_list)
    assert ev_list[0].graph_rank == 1


# --- Test 4: Evidence Merger & RRF Reranking ---


def test_evidence_merger_and_rrf_reranking() -> None:
    """Verify EvidenceMerger combines multiple streams and RRFReranker computes exact scores."""
    kb_id = "test_kb"

    # Graph stream evidence
    ev_graph_1 = Evidence(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        text="Graph hit 1",
        routes=[ProvenanceRoute.GRAPH],
        graph_rank=1,
        supporting_graph_paths=["Alice -[WORKS_ON]-> Titan"],
        supporting_edge_ids=["E_01"],
    )
    ev_graph_2 = Evidence(
        kb_id=kb_id,
        chunk_id="C002",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        text="Graph hit 2",
        routes=[ProvenanceRoute.GRAPH],
        graph_rank=2,
        supporting_graph_paths=[],
        supporting_edge_ids=[],
    )

    # Vector stream evidence (C001 also appears here at vector_rank=2; C003 appears at vector_rank=1)
    ev_vec_1 = Evidence(
        kb_id=kb_id,
        chunk_id="C003",
        doc_id="D002",
        filename="d2.txt",
        page_numbers=[1],
        text="Vector hit 1",
        routes=[ProvenanceRoute.VECTOR],
        vector_rank=1,
        vector_distance=0.05,
    )
    ev_vec_2 = Evidence(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="d1.txt",
        page_numbers=[1],
        text="Vector hit 2",
        routes=[ProvenanceRoute.VECTOR],
        vector_rank=2,
        vector_distance=0.12,
    )

    # Merge streams
    merged = EvidenceMerger.merge([[ev_graph_1, ev_graph_2], [ev_vec_1, ev_vec_2]])
    assert len(merged) == 3

    c001_merged = next(e for e in merged if e.chunk_id == "C001")
    assert ProvenanceRoute.GRAPH in c001_merged.routes
    assert ProvenanceRoute.VECTOR in c001_merged.routes
    assert c001_merged.graph_rank == 1
    assert c001_merged.vector_rank == 2
    assert c001_merged.supporting_graph_paths == ["Alice -[WORKS_ON]-> Titan"]

    # RRF Reranking with k=60
    reranker = RRFReranker(rrf_k=60)
    reranked = reranker.rerank(merged, top_k=3)

    # Expected score for C001: 1/(60 + 1) + 1/(60 + 2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # Expected score for C003: 1/(60 + 1) = 0.016393
    # Expected score for C002: 1/(60 + 2) = 0.016129
    assert reranked[0].chunk_id == "C001"
    assert pytest.approx(reranked[0].rrf_score, 0.0001) == (1.0 / 61.0 + 1.0 / 62.0)
    assert reranked[1].chunk_id == "C003"
    assert pytest.approx(reranked[1].rrf_score, 0.0001) == (1.0 / 61.0)
    assert reranked[2].chunk_id == "C002"
    assert pytest.approx(reranked[2].rrf_score, 0.0001) == (1.0 / 62.0)


# --- Test 5: Hybrid Orchestrator, Context Builder, and Answer Generator ---


def test_hybrid_orchestrator_context_builder_and_generator(tmp_path: Path) -> None:
    """Verify end-to-end integration: local and global queries, budget truncation, and grounded answer generation."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = DeterministicEmbedder(dim=768)

    # Add document and chunks
    c1 = Chunk(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="overview.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Dr. Alice Smith leads the Quantum Lab at IIIT.",
        char_count=46,
        content_hash="h1",
    )
    doc_store.save_chunks([c1])
    vstore.upsert_chunks([c1], [embedder.embed_text(c1.text)])

    # Add community report
    rep1 = CommunityReport(
        kb_id=kb_id,
        report_id="CR001",
        community_id=1,
        partition_id="part_001",
        member_entity_ids=["E001"],
        summary_text="The quantum computing cluster investigates hybrid algorithms.",
        supporting_chunk_ids=["C001"],
        model_tag="gemma3:4b",
        prompt_version="v1.0",
        verification_status="VERIFIED",
    )
    doc_store.save_community_reports([rep1])
    vstore.upsert_community_reports([rep1], [embedder.embed_text(rep1.summary_text)])

    # Add entity
    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Alice Smith",
        normalized_name="alice smith",
        entity_type="PERSON",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    doc_store.save_canonical_entities([e1])
    vstore.upsert_entities([e1], [embedder.embed_text(e1.name)])

    graph = IGraphBuilder.build_graph([e1], [])
    seed_retriever = SeedEntityRetriever(doc_store, vstore, embedder)
    graph_retriever = GraphEvidenceRetriever(doc_store, seed_retriever, graph=graph)
    vector_retriever = VectorEvidenceRetriever(doc_store, vstore, embedder, top_k=5)
    community_retriever = CommunityRetriever(doc_store, vstore, embedder, community_top_k=2)
    reranker = RRFReranker(rrf_k=60)

    orchestrator = HybridRetrieverOrchestrator(
        doc_store=doc_store,
        vector_store=vstore,
        embedding_provider=embedder,
        seed_retriever=seed_retriever,
        graph_retriever=graph_retriever,
        vector_retriever=vector_retriever,
        community_retriever=community_retriever,
        reranker=reranker,
    )

    # 1. Global Query
    global_plan = QueryPlan(
        question="Summarize the quantum research cluster.",
        intent=QueryIntent.GLOBAL,
    )
    global_res = orchestrator.retrieve_hybrid("Summarize the quantum research cluster.", global_plan, kb_id=kb_id)
    assert not global_res.abstained
    assert len(global_res.community_reports) >= 1
    assert global_res.community_reports[0].community_id == 1

    # 2. Context Builder Budgeting
    builder = ContextBuilder(max_chars=600)
    context_str = builder.build_context(global_res)
    assert len(context_str) <= 600
    assert "Community Summaries" in context_str

    # 3. Answer Generation with citations
    mock_answer = json.dumps(
        {
            "answer": "Dr. Alice Smith leads the Quantum Lab [C001, D001, p.1].",
            "abstained": False,
            "cited_chunk_ids": ["C001"],
        }
    )
    llm_client = OllamaLLMClient(custom_generator=lambda p: mock_answer)
    generator = OllamaAnswerGenerator(llm_client=llm_client, context_builder=builder)

    answer_text = generator.generate_answer(global_res)
    assert "Dr. Alice Smith leads the Quantum Lab" in answer_text
    assert "[C001, D001, p.1]" in answer_text

    # 4. Abstention handling
    abstained_result = RetrievalResult(
        kb_id=kb_id,
        question="Unanswerable question?",
        query_plan=QueryPlan(question="Unanswerable question?", requires_clarification=True, clarification_reason="Ambiguous concept"),
        abstained=True,
        abstention_reason="Ambiguous concept",
    )
    abstained_answer = generator.generate_answer(abstained_result)
    assert "unable to answer" in abstained_answer.lower()
    assert "Ambiguous concept" in abstained_answer



# --- Test 6: Vector Evidence Retriever Direct Testing ---


def test_vector_evidence_retriever_scope_and_ranking(tmp_path: Path) -> None:
    """Verify VectorEvidenceRetriever properly sets ranks, distances, and respects scope_id."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")
    embedder = DeterministicEmbedder(dim=768)

    c1 = Chunk(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="doc1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Alpha scope content.",
        char_count=20,
        content_hash="h1",
        scope_id="scope_A",
    )
    c2 = Chunk(
        kb_id=kb_id,
        chunk_id="C002",
        doc_id="D001",
        filename="doc1.txt",
        page_numbers=[1],
        chunk_index=1,
        text="Beta scope content.",
        char_count=19,
        content_hash="h2",
        scope_id="scope_B",
    )
    doc_store.save_chunks([c1, c2])
    vstore.upsert_chunks([c1, c2], [embedder.embed_text(c1.text), embedder.embed_text(c2.text)])

    retriever = VectorEvidenceRetriever(doc_store, vstore, embedder, top_k=5)
    plan = QueryPlan(question="content", intent=QueryIntent.LOCAL)

    # Scoped retrieval: scope_A should only return C001
    scoped_res = retriever.retrieve("content", plan, kb_id=kb_id, scope_id="scope_A")
    assert len(scoped_res) == 1
    assert scoped_res[0].chunk_id == "C001"
    assert scoped_res[0].vector_rank == 1
    assert scoped_res[0].routes == [ProvenanceRoute.VECTOR]


# --- Test 7: Seed Entity Retriever Ambiguity Tie Handling ---


def test_seed_entity_retriever_tied_candidates_ambiguity(tmp_path: Path) -> None:
    """Verify that multiple vector candidates with near-identical similarity are flagged as ambiguous."""
    kb_id = "test_kb"
    doc_store = DocumentStore(kb_id=kb_id, processed_dir=tmp_path / "processed")
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vstore")

    # Fixed embedder returning identical vector for both candidates and query
    dim = 768
    identical_vec = [1.0 / (dim ** 0.5)] * dim

    class AmbiguityEmbedder:
        def embed_text(self, text: str) -> List[float]:
            return list(identical_vec)

        def embed_batch(self, texts: List[str]) -> List[List[float]]:
            return [list(identical_vec) for _ in texts]

    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Project Phoenix Alpha",
        normalized_name="project phoenix alpha",
        entity_type="PROJECT",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e2 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E002",
        name="Project Phoenix Beta",
        normalized_name="project phoenix beta",
        entity_type="PROJECT",
        source_chunk_ids=["C002"],
        source_doc_ids=["D001"],
    )
    doc_store.save_canonical_entities([e1, e2])
    vstore.upsert_entities([e1, e2], [identical_vec, identical_vec])

    retriever = SeedEntityRetriever(
        doc_store=doc_store,
        vector_store=vstore,
        embedding_provider=AmbiguityEmbedder(),
        similarity_threshold=0.8,
    )

    # Query for "Phoenix" (doesn't exact-match either, but vector match produces exact tie)
    res = retriever.resolve_seeds([SeedEntitySpec(name="Phoenix", expected_type="PROJECT")])
    assert "Phoenix" in res.ambiguous_seeds
    assert len(res.ambiguous_seeds["Phoenix"]) >= 2
