from __future__ import annotations

from types import SimpleNamespace

from src.core.models import Evidence, ProvenanceRoute, QueryIntent, QueryPlan, SeedEntitySpec
from src.core.models import CanonicalEntity, Chunk, Relationship, ValidationStatus
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.graph.graph_builder import IGraphBuilder
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore
from src.retrieval.context_builder import ContextBuilder
from src.generation.answer_generator import OllamaAnswerGenerator
from src.extraction.schemas import OllamaLLMClient
from src.core.models import RetrievalResult
import hashlib
import numpy as np
from src.retrieval.community_retriever import HybridRetrieverOrchestrator
from src.retrieval.reranker import RRFReranker
from run_hotpot import select_contexts


def test_hotpot_oracle_and_full_context_selection():
    sample = {
        "context": [["support A", ["a"]], ["distractor", ["x"]], ["support B", ["b"]]],
        "supporting_facts": [["support A", 0], ["support B", 0]],
    }
    assert [x[0] for x in select_contexts(sample, "oracle")[0]] == ["support A", "support B"]
    assert [x[0] for x in select_contexts(sample, "full")[0]] == ["support A", "distractor", "support B"]
    missing = {"context": sample["context"]}
    contexts, fallback = select_contexts(missing, "oracle")
    assert len(contexts) == 2 and "missing" in fallback


def test_invalid_plan_relation_does_not_suppress_graph_or_vector():
    evidence = Evidence(kb_id="k", chunk_id="c1", doc_id="d1", text="support", routes=[ProvenanceRoute.VECTOR])

    class Seed:
        def resolve_seeds(self, _):
            return SimpleNamespace(resolved_entity_ids=["e1"], ambiguous_seeds={}, unresolved_seeds=[])

    class Graph:
        def __init__(self): self.calls = 0
        def retrieve(self, *_args, **_kwargs):
            self.calls += 1
            return []

    class Vector:
        def retrieve(self, *_args, **_kwargs): return [evidence]

    class Community:
        def retrieve_communities(self, *_args, **_kwargs): return [], []

    graph = Graph()
    orchestrator = HybridRetrieverOrchestrator(
        doc_store=object(), vector_store=object(), embedding_provider=object(), seed_retriever=Seed(),
        graph_retriever=graph, vector_retriever=Vector(), community_retriever=Community(), reranker=RRFReranker(),
    )
    plan = QueryPlan(question="q", intent=QueryIntent.LOCAL, seed_entities=[SeedEntitySpec(name="seed")], invalid_relationship_types=["ACTED_IN"])
    result = orchestrator.retrieve_hybrid("q", plan, "k")
    assert graph.calls == 1
    assert [e.chunk_id for e in result.evidence_list] == ["c1"]
    assert result.route_diagnostics["invalid_relation_count"] == 1
    assert result.route_diagnostics["vector_candidates"] == 1


def test_vector_failure_does_not_discard_graph_evidence():
    evidence = Evidence(kb_id="k", chunk_id="gc", doc_id="gd", text="graph support", routes=[ProvenanceRoute.GRAPH], graph_rank=1)

    class Seed:
        def resolve_seeds(self, _): return SimpleNamespace(resolved_entity_ids=["e1"], ambiguous_seeds={}, unresolved_seeds=[])
    class Graph:
        def retrieve(self, *_args, **_kwargs): return [evidence]
    class Vector:
        def retrieve(self, *_args, **_kwargs): raise RuntimeError("vector unavailable")
    class Community:
        def retrieve_communities(self, *_args, **_kwargs): return [], []

    orchestrator = HybridRetrieverOrchestrator(object(), object(), object(), Seed(), Graph(), Vector(), Community())
    result = orchestrator.retrieve_hybrid("q", QueryPlan(question="q", intent=QueryIntent.LOCAL, seed_entities=[SeedEntitySpec(name="seed")]), "k")
    assert len(result.evidence_list) == 1
    assert result.route_diagnostics["route_errors"]["vector"] == "vector unavailable"


def test_two_document_multi_hop_graph_provenance_and_context(tmp_path):
    class Embedder:
        def embed_text(self, text):
            rng = np.random.default_rng(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
            vector = rng.standard_normal(768)
            return (vector / np.linalg.norm(vector)).tolist()

    kb = "two_hop"
    store = DocumentStore(kb, processed_dir=tmp_path / "processed")
    vectors = ChromaVectorStore(kb, kb_vector_store_dir=tmp_path / "vectors")
    chunks = [
        Chunk(kb_id=kb, chunk_id="c1", doc_id="d1", filename="one.txt", page_numbers=[1], chunk_index=0,
              text="Asha founded Northstar Labs.", char_count=28, content_hash="h1"),
        Chunk(kb_id=kb, chunk_id="c2", doc_id="d2", filename="two.txt", page_numbers=[1], chunk_index=0,
              text="Northstar Labs developed the Orion telescope.", char_count=45, content_hash="h2"),
    ]
    entities = [
        CanonicalEntity(kb_id=kb, entity_id="e1", name="Asha", normalized_name="asha", entity_type="PERSON", source_chunk_ids=["c1"], source_doc_ids=["d1"]),
        CanonicalEntity(kb_id=kb, entity_id="e2", name="Northstar Labs", normalized_name="northstar labs", entity_type="ORGANIZATION", source_chunk_ids=["c1", "c2"], source_doc_ids=["d1", "d2"]),
        CanonicalEntity(kb_id=kb, entity_id="e3", name="Orion telescope", normalized_name="orion telescope", entity_type="TECHNOLOGY", source_chunk_ids=["c2"], source_doc_ids=["d2"]),
    ]
    relationships = [
        Relationship(kb_id=kb, relationship_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="FOUNDED", source_chunk_ids=["c1"], evidence_quotes=["Asha founded Northstar Labs."], validation_status=ValidationStatus.VALIDATED),
        Relationship(kb_id=kb, relationship_id="r2", source_entity_id="e2", target_entity_id="e3", relation_type="DEVELOPED", source_chunk_ids=["c2"], evidence_quotes=["Northstar Labs developed the Orion telescope."], validation_status=ValidationStatus.VALIDATED),
    ]
    store.save_chunks(chunks)
    store.save_canonical_entities(entities)
    graph = IGraphBuilder.build_graph(entities, relationships)
    embedder = Embedder()
    seed = SeedEntityRetriever(store, vectors, embedder)
    retriever = GraphEvidenceRetriever(store, seed, graph=graph)
    plan = QueryPlan(question="What did Asha's company develop?", intent=QueryIntent.LOCAL,
                     seed_entities=[SeedEntitySpec(name="Asha")], relationship_path=["FOUNDED", "DEVELOPED"])
    evidence = retriever.retrieve(plan.question, plan, kb)
    assert {e.doc_id for e in evidence} == {"d1", "d2"}
    result = SimpleNamespace(evidence_list=evidence, community_reports=[])
    context = ContextBuilder().build_context(result)
    assert "Asha founded Northstar Labs." in context
    assert "Northstar Labs developed the Orion telescope." in context


def test_vector_store_explicit_rebuild_clear(tmp_path):
    from src.core.models import Chunk
    store = ChromaVectorStore("rebuild", kb_vector_store_dir=tmp_path / "vectors")
    chunk = Chunk(kb_id="rebuild", chunk_id="old", doc_id="d", filename="old.txt", page_numbers=[1],
                  chunk_index=0, text="old content", char_count=11, content_hash="h")
    store.upsert_chunks([chunk], [[1.0, 0.0]])
    assert store.query_chunks([1.0, 0.0])
    store.clear()
    assert store.query_chunks([1.0, 0.0]) == []


def test_answer_marker_and_structured_status_stay_consistent():
    ev = Evidence(kb_id="k", chunk_id="c", doc_id="d", text="some context", routes=[ProvenanceRoute.VECTOR])
    result = RetrievalResult(kb_id="k", question="q", query_plan=QueryPlan(question="q"), evidence_list=[ev])
    llm = OllamaLLMClient(custom_generator=lambda _prompt: '{"answer":"[abstained] insufficient facts", "abstained":false, "cited_chunk_ids":["c"]}')
    generated = OllamaAnswerGenerator(llm).generate_structured(result)
    assert generated.abstained is True
    assert generated.cited_chunk_ids == []
    assert result.abstained is True
    assert result.answer_abstained is True
