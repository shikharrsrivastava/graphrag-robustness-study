"""Unit and integration tests for targeted community-report refresh."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import yaml
import igraph

from experiments.refresh_community_reports import (
    load_config_from_kb,
    refresh_community_reports,
    run_preflight_check,
)
from src.core.configuration import EngineConfig
from src.core.models import CanonicalEntity, Chunk, CommunityReport, Relationship, ValidationStatus
from src.extraction.schemas import OllamaLLMClient
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.graph.graph_builder import IGraphBuilder
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


@pytest.fixture
def mock_kb(tmp_path: Path) -> Path:
    """Create a minimal synthetic knowledge base directory with graph, chunks, and vector store."""
    kb_id = "test_refresh_kb"
    kb_dir = tmp_path / "knowledge_bases" / kb_id
    graph_dir = kb_dir / "graph"
    data_dir = kb_dir / "data" / "processed"
    vstore_dir = kb_dir / "vector_store"

    graph_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    vstore_dir.mkdir(parents=True)

    # 1. Config snapshot
    config_dict = {
        "kb_id": kb_id,
        "storage_root": "knowledge_bases",
        "ontology": {
            "domain_name": "test",
            "entity_types": {"PERSON": "Person", "FILM": "Film"},
            "relationship_types": {"DIRECTED": {"name": "DIRECTED", "description": "Directed"}},
            "static_aliases": {},
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "llm_model": "test-llm",
            "embedding_model": "test-embed",
            "expected_embedding_dim": 64,
        },
    }
    with open(kb_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    # 2. Document store files
    doc_store = DocumentStore(kb_id=kb_id, storage_root="knowledge_bases", base_dir=tmp_path)

    chunk = Chunk(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="doc.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Alice directed the film Wonderland.",
        char_count=35,
        content_hash="h1",
    )
    doc_store.save_chunks([chunk])

    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Alice",
        normalized_name="alice",
        entity_type="PERSON",
        description="Director Alice",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e2 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E002",
        name="Wonderland",
        normalized_name="wonderland",
        entity_type="FILM",
        description="Fantasy film",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e3 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E003",
        name="Bob",
        normalized_name="bob",
        entity_type="PERSON",
        description="Actor Bob",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    doc_store.save_canonical_entities([e1, e2, e3])

    rel = Relationship(
        kb_id=kb_id,
        relationship_id="R001",
        source_entity_id="E001",
        target_entity_id="E002",
        relation_type="DIRECTED",
        source_chunk_ids=["C001"],
        evidence_quotes=["Alice directed the film Wonderland."],
        validation_status=ValidationStatus.VALIDATED,
    )
    doc_store.save_relationships([rel])

    # 2 communities: Comm 0 (E001, E002 with edge), Comm 1 (E003 singleton)
    comms = [
        {"kb_id": kb_id, "community_id": 0, "member_entity_ids": ["E001", "E002"], "partition_id": "p0"},
        {"kb_id": kb_id, "community_id": 1, "member_entity_ids": ["E003"], "partition_id": "p1"},
    ]
    doc_store.save_communities(comms)

    # Initial legacy reports (UNVERIFIED)
    legacy_reports = [
        CommunityReport(
            kb_id=kb_id,
            report_id="COMM_REPORT_0000",
            community_id=0,
            partition_id="p0",
            member_entity_ids=["E001", "E002"],
            summary_text="Legacy unverified summary for comm 0",
            verification_status="UNVERIFIED",
            model_tag="old",
            prompt_version="v1.0",
        ),
        CommunityReport(
            kb_id=kb_id,
            report_id="COMM_REPORT_0001",
            community_id=1,
            partition_id="p1",
            member_entity_ids=["E003"],
            summary_text="Legacy unverified summary for comm 1",
            verification_status="UNVERIFIED",
            model_tag="old",
            prompt_version="v1.0",
        ),
    ]
    doc_store.save_community_reports(legacy_reports)

    # 3. Vector store population
    vstore = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=vstore_dir)
    dim = 64
    vstore.upsert_chunks([chunk], [[0.1] * dim])
    vstore.upsert_entities([e1, e2, e3], [[0.2] * dim, [0.3] * dim, [0.4] * dim])
    vstore.upsert_community_reports(legacy_reports, [[0.5] * dim, [0.6] * dim])

    return kb_dir


def test_refresh_community_reports_dry_run(mock_kb: Path) -> None:
    """Verify that dry-run mode inspects but never writes files or alters vector collections."""
    vstore_dir = mock_kb / "vector_store"
    vstore = ChromaVectorStore(kb_id="test_refresh_kb", kb_vector_store_dir=vstore_dir)
    initial_chunks = vstore.chunk_collection.count()
    initial_entities = vstore.entity_collection.count()
    initial_comms = vstore.community_collection.count()

    res = refresh_community_reports(kb_dir=mock_kb, dry_run=True)
    assert res["status"] == "DRY_RUN_COMPLETED"
    assert res["preflight"]["total_communities"] == 2
    assert res["preflight"]["singletons_fast_path"] == 1
    assert res["preflight"]["non_singletons_llm_required"] == 1

    # Verify vector store collections are completely unchanged
    assert vstore.chunk_collection.count() == initial_chunks
    assert vstore.entity_collection.count() == initial_entities
    assert vstore.community_collection.count() == initial_comms

    # Verify no backup files were created
    backups = list((mock_kb / "graph").glob("*.bak.*"))
    assert len(backups) == 0


def test_refresh_community_reports_execution_and_idempotence(mock_kb: Path) -> None:
    """Verify targeted execution: atomic backup, status verification, vector refresh, and idempotence."""
    dim = 64

    # Mock LLM returns verified structured claim for comm 0
    def mock_gen(prompt: str) -> str:
        return json.dumps({
            "claims": [
                {
                    "subject_id": "E001",
                    "subject_type": "PERSON",
                    "relation_type": "DIRECTED",
                    "object_id": "E002",
                    "object_type": "FILM",
                    "supporting_edge_ids": ["R001"],
                }
            ]
        })

    from src.core.configuration import OllamaConfig
    mock_llm = OllamaLLMClient(custom_generator=mock_gen)
    mock_embed = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.7] * dim for _ in texts],
    )

    vstore_dir = mock_kb / "vector_store"
    vstore = ChromaVectorStore(kb_id="test_refresh_kb", kb_vector_store_dir=vstore_dir)

    initial_chunks = vstore.chunk_collection.count()
    initial_entities = vstore.entity_collection.count()

    # 1. Execute targeted refresh
    res = refresh_community_reports(
        kb_dir=mock_kb,
        dry_run=False,
        custom_llm_client=mock_llm,
        custom_embedding_service=mock_embed,
    )

    assert res["status"] == "COMPLETED"
    assert res["reports_count"] == 2
    assert res["status_breakdown"] == {"VERIFIED": 2}
    assert res["verification"]["chunks_untouched"] is True
    assert res["verification"]["entities_untouched"] is True

    # Check backup file exists
    backups = list((mock_kb / "graph").glob("community_reports.jsonl.bak.*"))
    assert len(backups) == 1

    # Check persisted reports in JSONL
    report_file = mock_kb / "graph" / "community_reports.jsonl"
    with open(report_file, "r", encoding="utf-8") as f:
        reports = [CommunityReport.model_validate_json(line) for line in f if line.strip()]
    assert len(reports) == 2
    assert all(r.verification_status == "VERIFIED" for r in reports)
    assert reports[0].report_id == "COMM_REPORT_0000"
    assert reports[1].report_id == "COMM_REPORT_0001"
    assert len(reports[0].verified_claims) == 1
    assert reports[0].verified_claims[0].relation_type == "DIRECTED"

    # Verify vector store counts
    assert vstore.chunk_collection.count() == initial_chunks
    assert vstore.entity_collection.count() == initial_entities
    assert vstore.community_collection.count() == 2

    # Query Chroma to ensure the newly embedded reports are retrievable
    query_res = vstore.query_community_reports(query_embedding=[0.7] * dim, top_k=2)
    assert len(query_res) == 2
    retrieved_ids = {r["id"] for r in query_res}
    assert retrieved_ids == {"COMM_REPORT_0000", "COMM_REPORT_0001"}

    # 2. Idempotence test: re-running produces the exact same clean state
    res2 = refresh_community_reports(
        kb_dir=mock_kb,
        dry_run=False,
        custom_llm_client=mock_llm,
        custom_embedding_service=mock_embed,
    )
    assert res2["status"] == "COMPLETED"
    assert res2["reports_count"] == 2
    assert vstore.chunk_collection.count() == initial_chunks
    assert vstore.entity_collection.count() == initial_entities
    assert vstore.community_collection.count() == 2
