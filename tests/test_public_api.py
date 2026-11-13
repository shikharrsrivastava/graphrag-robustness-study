"""Integration tests for the public Hybrid GraphRAG Python API."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import yaml

from hybrid_graphrag import (
    Citation,
    ConfigurationError,
    HybridGraphRAG,
    IngestionError,
    KnowledgeBaseNotFoundError,
    QueryResult,
)
from src.core.configuration import EngineConfig, OllamaConfig
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CommunityReport,
    Evidence,
    ProvenanceRoute,
    QueryPlan,
    Relationship,
    ValidationStatus,
)
from src.extraction.schemas import OllamaLLMClient
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.graph.graph_builder import IGraphBuilder
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


@pytest.fixture
def sample_config_dict(tmp_path: Path) -> dict:
    return {
        "kb_id": "test_public_kb",
        "storage_root": str(tmp_path / "kbs"),
        "ontology": {
            "domain_name": "academic",
            "entity_types": {"PERSON": "Person", "UNIVERSITY": "University"},
            "relationship_types": {"STUDIES_AT": {"name": "STUDIES_AT", "description": "Studies at"}},
            "static_aliases": {},
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "llm_model": "test-llm",
            "embedding_model": "test-embed",
            "expected_embedding_dim": 64,
        },
    }


def test_public_imports() -> None:
    """Verify that all public symbols are importable from hybrid_graphrag and src."""
    import hybrid_graphrag
    from hybrid_graphrag import (
        Citation,
        ConfigurationError,
        HybridGraphRAG,
        IngestionError,
        KnowledgeBaseNotFoundError,
        QueryResult,
    )
    import src
    from src import HybridGraphRAG as SrcHybridGraphRAG

    assert HybridGraphRAG is SrcHybridGraphRAG
    assert hasattr(hybrid_graphrag, "HybridGraphRAG")
    assert hasattr(hybrid_graphrag, "QueryResult")
    assert hasattr(hybrid_graphrag, "Citation")


def test_initialization_from_dict_and_yaml(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify initialization from dictionary and YAML file."""
    dim = 64
    mock_llm = OllamaLLMClient(custom_generator=lambda _p: "{}")
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.1] * dim for _ in texts],
    )

    # 1. Init from dict
    engine_dict = HybridGraphRAG.from_config(
        config=sample_config_dict,
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )
    assert engine_dict.kb_id == "test_public_kb"
    engine_dict.close()

    # 2. Init from YAML
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config_dict, f)

    engine_yaml = HybridGraphRAG.from_config(
        config=yaml_path,
        kb_id="overridden_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )
    assert engine_yaml.kb_id == "overridden_kb"
    engine_yaml.close()


def test_invalid_configuration_errors(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that clear exceptions are raised for invalid configuration."""
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        HybridGraphRAG.from_config(tmp_path / "non_existent.yaml")

    with pytest.raises(ConfigurationError, match="Invalid configuration dictionary"):
        HybridGraphRAG.from_config({"invalid": "missing ontology and kb_id"})

    invalid_slash = dict(sample_config_dict)
    invalid_slash["kb_id"] = "invalid/slash/id"
    with pytest.raises(ConfigurationError, match="path separators are not permitted"):
        HybridGraphRAG.from_config(invalid_slash)


def test_knowledge_base_isolation(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that two independent KBs maintain isolated documents, vectors, and graphs."""
    dim = 64
    mock_llm = OllamaLLMClient(custom_generator=lambda _p: "{}")
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.1] * dim for _ in texts],
    )

    # KB 1: research
    engine_res = HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="research_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )

    # KB 2: medical
    engine_med = HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="medical_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )

    assert engine_res.kb_id == "research_kb"
    assert engine_med.kb_id == "medical_kb"

    # Add a mock chunk only to research_kb
    chunk = Chunk(
        kb_id="research_kb",
        chunk_id="RES_C001",
        doc_id="D1",
        filename="res.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Quantum mechanics research.",
        char_count=27,
        content_hash="h1",
    )
    engine_res._engine.doc_store.save_chunks([chunk])
    engine_res._engine.vector_store.upsert_chunks([chunk], [[0.1] * dim])

    assert engine_res._engine.doc_store.get_chunk("RES_C001") is not None
    assert engine_med._engine.doc_store.get_chunk("RES_C001") is None

    # Vector store query isolation
    res_vec = engine_res._engine.vector_store.query_chunks([0.1] * dim, top_k=5)
    med_vec = engine_med._engine.vector_store.query_chunks([0.1] * dim, top_k=5)

    assert len(res_vec) == 1
    assert len(med_vec) == 0

    engine_res.close()
    engine_med.close()


def test_reopen_persisted_knowledge_base(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that an existing knowledge base can be reopened without re-ingesting documents."""
    dim = 64
    mock_llm = OllamaLLMClient(custom_generator=lambda _p: "{}")
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.1] * dim for _ in texts],
    )

    # 1. Initialize and persist data in KB
    engine = HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="persisted_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )
    kb_dir = engine.config.get_kb_root()

    e = CanonicalEntity(
        kb_id="persisted_kb",
        entity_id="E001",
        name="Alice",
        normalized_name="alice",
        entity_type="PERSON",
        description="Researcher",
        source_chunk_ids=["C1"],
        source_doc_ids=["D1"],
    )
    engine._engine.doc_store.save_canonical_entities([e])
    engine.close()

    # 2. Reopen using from_existing
    reopened = HybridGraphRAG.from_existing(
        kb_dir=kb_dir,
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )
    assert reopened.kb_id == "persisted_kb"
    stats = reopened.get_kb_stats()
    assert stats["entities"] == 1
    reopened.close()

    # Reopening non-existent directory raises KnowledgeBaseNotFoundError
    with pytest.raises(KnowledgeBaseNotFoundError):
        HybridGraphRAG.from_existing(tmp_path / "does_not_exist")


def test_query_execution_and_structured_result(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify query execution produces structured QueryResult with citations."""
    dim = 64

    # Setup pre-populated data in KB
    chunk = Chunk(
        kb_id="query_test_kb",
        chunk_id="CHK_001",
        doc_id="DOC_001",
        filename="quantum.txt",
        page_numbers=[3],
        chunk_index=0,
        text="Dr. Alice published findings on quantum coherence.",
        char_count=49,
        content_hash="h1",
    )

    def mock_llm_gen(prompt: str) -> str:
        return json.dumps({
            "reasoning": "Found published findings.",
            "canonical_answer": "Quantum coherence",
            "answer": "Dr. Alice published findings on quantum coherence [CHK_001].",
            "abstained": False,
            "cited_chunk_ids": ["CHK_001"],
        })

    mock_llm = OllamaLLMClient(custom_generator=mock_llm_gen)
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.5] * dim for _ in texts],
    )

    with HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="query_test_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    ) as engine:
        engine._engine.doc_store.save_chunks([chunk])
        engine._engine.vector_store.upsert_chunks([chunk], [[0.5] * dim])

        # Query using vector_only mode
        res = engine.query("What did Dr. Alice publish?", mode="vector_only")

        assert isinstance(res, QueryResult)
        assert res.canonical_answer == "Quantum coherence"
        assert "quantum coherence" in res.answer.lower()
        assert res.abstained is False
        assert res.cited_chunk_ids == ["CHK_001"]

        # Citation verification
        assert len(res.citations) == 1
        cit = res.citations[0]
        assert isinstance(cit, Citation)
        assert cit.chunk_id == "CHK_001"
        assert cit.doc_id == "DOC_001"
        assert cit.filename == "quantum.txt"
        assert cit.page_numbers == [3]
        assert "quantum coherence" in cit.text_snippet

        # String representation
        cit_str = str(cit)
        assert "CHK_001" in cit_str
        assert "Doc: quantum.txt" in cit_str
        assert "p.3" in cit_str


def test_ingestion_error_handling(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that IngestionError is raised when input path does not exist."""
    dim = 64
    mock_llm = OllamaLLMClient(custom_generator=lambda _p: "{}")
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.1] * dim for _ in texts],
    )

    with HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="ingest_err_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    ) as engine:
        with pytest.raises(IngestionError, match="Document input path does not exist"):
            engine.ingest(tmp_path / "missing_dir")


def test_lifecycle_and_cleanup(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that close() cleans resources and prevents queries after closure."""
    dim = 64
    mock_llm = OllamaLLMClient(custom_generator=lambda _p: "{}")
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.1] * dim for _ in texts],
    )

    engine = HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="lifecycle_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    )

    assert engine._closed is False
    engine.close()
    assert engine._closed is True

    # Operations after close raise RuntimeError
    with pytest.raises(RuntimeError, match="already been closed"):
        engine.query("Test question?")

    with pytest.raises(RuntimeError, match="already been closed"):
        engine.ingest(tmp_path)


def test_canonical_answer_propagation_in_hybrid_query(tmp_path: Path, sample_config_dict: dict) -> None:
    """Verify that canonical_answer propagates from internal structured answer to public QueryResult in hybrid mode."""
    dim = 64
    chunk1 = Chunk(
        kb_id="hybrid_prop_kb",
        chunk_id="CHK_001",
        doc_id="DOC_001",
        filename="doc1.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Scott Derrickson is an American director.",
        char_count=41,
        content_hash="h1",
    )
    chunk2 = Chunk(
        kb_id="hybrid_prop_kb",
        chunk_id="CHK_002",
        doc_id="DOC_002",
        filename="doc2.txt",
        page_numbers=[1],
        chunk_index=1,
        text="Ed Wood was an American filmmaker.",
        char_count=34,
        content_hash="h2",
    )

    def mock_llm_gen(prompt: str) -> str:
        # Planner output
        if "seed_entities" in prompt or "Query Plan" in prompt:
            return json.dumps({
                "intent": "LOCAL",
                "seed_entities": [{"name": "Scott Derrickson", "expected_type": "PERSON"}, {"name": "Ed Wood", "expected_type": "PERSON"}],
                "relationship_path": ["BORN_IN"],
            })
        # Structured answer generation output
        return json.dumps({
            "reasoning": "Both Scott Derrickson and Ed Wood were American.",
            "canonical_answer": "yes",
            "answer": "Scott Derrickson was American [CHK_001] and Ed Wood was American [CHK_002]. Therefore, they were of the same nationality.",
            "abstained": False,
            "cited_chunk_ids": ["CHK_001", "CHK_002"],
        })

    mock_llm = OllamaLLMClient(custom_generator=mock_llm_gen)
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.5] * dim for _ in texts],
    )

    with HybridGraphRAG.from_config(
        config=sample_config_dict,
        kb_id="hybrid_prop_kb",
        llm_client=mock_llm,
        embedding_service=mock_embed,
    ) as engine:
        engine._engine.doc_store.save_chunks([chunk1, chunk2])
        engine._engine.vector_store.upsert_chunks([chunk1, chunk2], [[0.5] * dim, [0.5] * dim])

        res = engine.query("Were Scott Derrickson and Ed Wood of the same nationality?", mode="hybrid")

        assert isinstance(res, QueryResult)
        assert res.canonical_answer == "yes"
        assert res.abstained is False
        assert set(res.cited_chunk_ids) == {"CHK_001", "CHK_002"}
        assert len(res.citations) == 2
        assert all(c.valid for c in res.citations)
