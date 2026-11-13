"""Unit tests for VectorStore, IGraphBuilder, GraphValidator, Leiden Clustering, and Community Summarizer."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import igraph
import numpy as np
import pytest
from src.communities.community_summarizer import CommunitySummarizer
from src.core.configuration import LeidenConfig, OllamaConfig, load_engine_config
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CommunityReport,
    Polarity,
    Relationship,
    ValidationStatus,
)
from src.embeddings.embedding_service import OllamaEmbeddingService
from src.extraction.schemas import OllamaLLMClient
from src.graph.clustering import LeidenCommunityDetector
from src.graph.graph_builder import IGraphBuilder
from src.graph.graph_validator import GraphValidator, compute_graph_fingerprint
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore


# --- Test 1: ChromaVectorStore ---


def test_chroma_vector_store_collections_and_queries(tmp_path: Path) -> None:
    """Test ChromaVectorStore persistence, cosine distance, metadata, and top-k queries."""
    kb_id = "academic_kb"
    vstore_dir = tmp_path / "vector_store"
    store = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=vstore_dir)

    # Verify collections exist with cosine space
    assert store.entity_collection.metadata.get("hnsw:space") == "cosine"
    assert store.chunk_collection.metadata.get("hnsw:space") == "cosine"
    assert store.community_collection.metadata.get("hnsw:space") == "cosine"

    dim = 768
    # 1. Test Entity Descriptions
    e1 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E001",
        name="Rahul",
        normalized_name="rahul",
        entity_type="PERSON",
        description="Student researcher",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )
    e2 = CanonicalEntity(
        kb_id=kb_id,
        entity_id="E002",
        name="IIIT Naya Raipur",
        normalized_name="iiit naya raipur",
        entity_type="UNIVERSITY",
        description="Institute of higher education",
        source_chunk_ids=["C001"],
        source_doc_ids=["D001"],
    )

    vec_e1 = [0.1] * dim
    vec_e2 = [-0.1] * dim

    store.upsert_entities([e1, e2], [vec_e1, vec_e2])

    # Query with entity_type filter
    res_person = store.query_entities(vec_e1, top_k=2, entity_type="PERSON")
    assert len(res_person) == 1
    assert res_person[0]["id"] == "E001"
    assert res_person[0]["metadata"]["entity_type"] == "PERSON"

    res_all = store.query_entities(vec_e1, top_k=2)
    assert len(res_all) == 2

    # 2. Test Text Chunks with scope_id
    c1 = Chunk(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="p.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Text chunk 1 content.",
        char_count=21,
        content_hash="h1",
        scope_id="scope_alpha",
    )
    c2 = Chunk(
        kb_id=kb_id,
        chunk_id="C002",
        doc_id="D001",
        filename="p.txt",
        page_numbers=[1],
        chunk_index=1,
        text="Text chunk 2 content.",
        char_count=21,
        content_hash="h2",
        scope_id="scope_beta",
    )

    vec_c1 = [0.2] * dim
    vec_c2 = [-0.2] * dim

    store.upsert_chunks([c1, c2], [vec_c1, vec_c2])

    res_scoped = store.query_chunks(vec_c1, top_k=5, scope_id="scope_alpha")
    assert len(res_scoped) == 1
    assert res_scoped[0]["id"] == "C001"
    assert res_scoped[0]["metadata"]["scope_id"] == "scope_alpha"

    # 3. Test Community Reports
    r1 = CommunityReport(
        kb_id=kb_id,
        report_id="COMM_REPORT_0000",
        community_id=0,
        partition_id="part_1",
        member_entity_ids=["E001", "E002"],
        supporting_chunk_ids=["C001"],
        summary_text="Academic community summary text.",
        model_tag="gemma3:4b",
        prompt_version="v1.0",
    )

    vec_r1 = [0.3] * dim
    store.upsert_community_reports([r1], [vec_r1])

    res_report = store.query_community_reports(vec_r1, top_k=2)
    assert len(res_report) == 1
    assert res_report[0]["id"] == "COMM_REPORT_0000"
    assert res_report[0]["metadata"]["community_id"] == 0

    # Cross-KB rejection
    bad_entity = CanonicalEntity(
        kb_id="wrong_kb",
        entity_id="E999",
        name="Bad",
        normalized_name="bad",
        entity_type="PERSON",
        source_chunk_ids=["C1"],
        source_doc_ids=["D1"],
    )
    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.upsert_entities([bad_entity], [[0.1] * dim])


# --- Test 2: IGraphBuilder & GraphValidator ---


def test_graph_builder_and_validator(tmp_path: Path) -> None:
    """Test building teaching graph, preserving distinct multigraph edges, tracking isolated nodes, and manifest."""
    kb_id = "academic_kb"
    cfg = load_engine_config(Path("configs/domains/academic.yaml"))

    entities = [
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E001",
            name="Rahul",
            normalized_name="rahul",
            entity_type="PERSON",
            description="Graduate student",
            source_chunk_ids=["C001", "C002"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E002",
            name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            description="Faculty advisor",
            source_chunk_ids=["C001", "C002", "C003"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E003",
            name="Medical AI",
            normalized_name="medical ai",
            entity_type="PROJECT",
            description="Healthcare diagnostic lab",
            source_chunk_ids=["C004"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E004",
            name="Machine Learning",
            normalized_name="machine learning",
            entity_type="TECHNOLOGY",
            description="Computational statistical learning",
            source_chunk_ids=["C005"],
            source_doc_ids=["D001"],
        ),
        # Isolated node with no edges
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E005",
            name="Unrelated Concept",
            normalized_name="unrelated concept",
            entity_type="CONCEPT",
            description="Stand-alone theoretical concept",
            source_chunk_ids=["C006"],
            source_doc_ids=["D001"],
        ),
    ]

    relationships = [
        # Distinct Edge 1 between E001 and E002 (WORKS_UNDER)
        Relationship(
            kb_id=kb_id,
            relationship_id="R001",
            source_entity_id="E001",
            target_entity_id="E002",
            relation_type="WORKS_UNDER",
            source_chunk_ids=["C001", "C002"],
            source_doc_ids=["D001"],
            evidence_quotes=["Rahul works under Professor Sharma."],
            validation_status=ValidationStatus.VALIDATED,
            support_count=2,
        ),
        # Distinct Edge 2 between E001 and E002 (AUTHORED - Rahul authored a paper with Sharma)
        Relationship(
            kb_id=kb_id,
            relationship_id="R002",
            source_entity_id="E001",
            target_entity_id="E003",
            relation_type="WORKS_ON",
            source_chunk_ids=["C003"],
            source_doc_ids=["D001"],
            evidence_quotes=["Rahul works on Medical AI."],
            validation_status=ValidationStatus.VALIDATED,
            support_count=1,
        ),
        # Edge between E002 and E003
        Relationship(
            kb_id=kb_id,
            relationship_id="R003",
            source_entity_id="E002",
            target_entity_id="E003",
            relation_type="SUPERVISES",
            source_chunk_ids=["C004"],
            source_doc_ids=["D001"],
            evidence_quotes=["Professor Sharma supervises Medical AI."],
            validation_status=ValidationStatus.VALIDATED,
            support_count=1,
        ),
        # Edge between E003 and E004
        Relationship(
            kb_id=kb_id,
            relationship_id="R004",
            source_entity_id="E003",
            target_entity_id="E004",
            relation_type="USES",
            source_chunk_ids=["C005"],
            source_doc_ids=["D001"],
            evidence_quotes=["Medical AI uses Machine Learning."],
            validation_status=ValidationStatus.VALIDATED,
            support_count=1,
        ),
    ]

    graph = IGraphBuilder.build_graph(entities, relationships)

    assert graph.is_directed() is True
    assert graph.vcount() == 5
    assert graph.ecount() == 4
    assert set(graph.vs["entity_id"]) == {"E001", "E002", "E003", "E004", "E005"}

    manifest_file = tmp_path / "manifest.json"
    valid_chunks = {"C001", "C002", "C003", "C004", "C005", "C006"}
    report = GraphValidator.validate(
        graph=graph,
        entities=entities,
        relationships=relationships,
        valid_chunk_ids=valid_chunks,
        ontology=cfg.ontology,
        manifest_path=manifest_file,
    )

    assert report["is_valid"] is True
    assert report["node_count"] == 5
    assert report["edge_count"] == 4
    assert report["isolated_nodes"] == ["E005"]
    assert report["relation_type_distribution"]["WORKS_UNDER"] == 1
    assert report["relation_type_distribution"]["SUPERVISES"] == 1
    assert report["relation_type_distribution"]["USES"] == 1
    assert len(report["graph_fingerprint"]) == 64
    assert manifest_file.exists()

    # DocumentStore reconstruction test
    doc_store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    doc_store.save_canonical_entities(entities)
    doc_store.save_relationships(relationships)

    reconstructed = IGraphBuilder.reconstruct_from_store(doc_store)
    assert reconstructed.vcount() == 5
    assert reconstructed.ecount() == 4
    assert compute_graph_fingerprint(reconstructed) == report["graph_fingerprint"]


# --- Test 3: Leiden Community Detection ---


def test_leiden_community_detection_and_edge_weights(tmp_path: Path) -> None:
    """Test Leiden clustering on undirected projection with distinct supporting chunk weights."""
    kb_id = "academic_kb"
    entities = [
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E001",
            name="Rahul",
            normalized_name="rahul",
            entity_type="PERSON",
            source_chunk_ids=["C001"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E002",
            name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            source_chunk_ids=["C001"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E003",
            name="Medical AI",
            normalized_name="medical ai",
            entity_type="PROJECT",
            source_chunk_ids=["C003"],
            source_doc_ids=["D001"],
        ),
    ]

    relationships = [
        # Two directed edges between E001 and E002 with chunks C001, C002 and C002, C003
        Relationship(
            kb_id=kb_id,
            relationship_id="R001",
            source_entity_id="E001",
            target_entity_id="E002",
            relation_type="WORKS_UNDER",
            source_chunk_ids=["C001", "C002"],
            evidence_quotes=["Quote 1"],
            validation_status=ValidationStatus.VALIDATED,
        ),
        Relationship(
            kb_id=kb_id,
            relationship_id="R002",
            source_entity_id="E002",
            target_entity_id="E001",
            relation_type="MENTORS",
            source_chunk_ids=["C002", "C003"],
            evidence_quotes=["Quote 2"],
            validation_status=ValidationStatus.VALIDATED,
        ),
        # Edge to E003
        Relationship(
            kb_id=kb_id,
            relationship_id="R003",
            source_entity_id="E002",
            target_entity_id="E003",
            relation_type="SUPERVISES",
            source_chunk_ids=["C004"],
            evidence_quotes=["Quote 3"],
            validation_status=ValidationStatus.VALIDATED,
        ),
    ]

    graph = IGraphBuilder.build_graph(entities, relationships)
    assert graph.is_directed() is True

    doc_store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    detector = LeidenCommunityDetector(
        config=LeidenConfig(random_seed=42, resolution_parameter=1.0),
        doc_store=doc_store,
    )

    communities = detector.detect_communities(graph, kb_id=kb_id)

    # Verify original directed graph is unchanged
    assert graph.is_directed() is True

    assert len(communities) >= 1
    assert "community_id" in communities[0]
    assert "member_entity_ids" in communities[0]
    assert "partition_id" in communities[0]
    assert communities[0]["seed"] == 42
    assert communities[0]["resolution"] == 1.0

    # Verify DocumentStore saved communities.jsonl
    loaded_comm = doc_store.load_communities()
    assert len(loaded_comm) == len(communities)
    assert loaded_comm[0]["community_id"] == communities[0]["community_id"]


# --- Test 4: Community Summarizer & ChromaDB Vector Indexing ---


def test_community_summarizer_and_chroma_indexing(tmp_path: Path) -> None:
    """Test CommunitySummarizer generates grounded reports and indexes them into ChromaDB."""
    kb_id = "academic_kb"
    dim = 768

    doc_store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    chunk = Chunk(
        kb_id=kb_id,
        chunk_id="C001",
        doc_id="D001",
        filename="p.txt",
        page_numbers=[1],
        chunk_index=0,
        text="Rahul conducts research under Professor Sharma in the AI lab.",
        char_count=62,
        content_hash="h1",
    )
    doc_store.save_chunks([chunk])

    entities = [
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E001",
            name="Rahul",
            normalized_name="rahul",
            entity_type="PERSON",
            description="Student researcher",
            source_chunk_ids=["C001"],
            source_doc_ids=["D001"],
        ),
        CanonicalEntity(
            kb_id=kb_id,
            entity_id="E002",
            name="Professor Sharma",
            normalized_name="professor sharma",
            entity_type="PERSON",
            description="Faculty supervisor",
            source_chunk_ids=["C001"],
            source_doc_ids=["D001"],
        ),
    ]

    relationships = [
        Relationship(
            kb_id=kb_id,
            relationship_id="R001",
            source_entity_id="E001",
            target_entity_id="E002",
            relation_type="WORKS_UNDER",
            source_chunk_ids=["C001"],
            evidence_quotes=["Rahul conducts research under Professor Sharma."],
            validation_status=ValidationStatus.VALIDATED,
        )
    ]

    graph = IGraphBuilder.build_graph(entities, relationships)

    communities = [
        {
            "kb_id": kb_id,
            "community_id": 0,
            "member_entity_ids": ["E001", "E002"],
            "size": 2,
            "partition_id": "test_part_001",
            "resolution": 1.0,
            "seed": 42,
            "objective_value": 0.0,
            "graph_fingerprint": "abc",
        }
    ]

    # Mock LLM returns grounded summary
    mock_response = {
        "summary_text": "This academic community consists of student Rahul working under the supervision of Professor Sharma in the AI lab."
    }
    llm_client = OllamaLLMClient(custom_generator=lambda prompt: json.dumps(mock_response))

    # Mock embedding provider
    embed_service = OllamaEmbeddingService(
        config=OllamaConfig(expected_embedding_dim=dim),
        custom_embedder=lambda texts: [[0.5] * dim for _ in texts],
    )

    vector_store = ChromaVectorStore(kb_id=kb_id, kb_vector_store_dir=tmp_path / "vector_store")

    summarizer = CommunitySummarizer(
        llm_client=llm_client,
        embedding_provider=embed_service,
        vector_store=vector_store,
        doc_store=doc_store,
    )

    reports = summarizer.summarize_communities(communities, graph, doc_store=doc_store)

    assert len(reports) == 1
    rep = reports[0]
    assert rep.community_id == 0
    assert rep.report_id == "COMM_REPORT_0000"
    assert "Rahul" in rep.summary_text
    assert "Professor Sharma" in rep.summary_text
    assert rep.member_entity_ids == ["E001", "E002"]
    assert "C001" in rep.supporting_chunk_ids

    # Verify JSONL persistence in doc_store
    loaded_reports = doc_store.load_community_reports()
    assert len(loaded_reports) == 1
    assert loaded_reports[0].report_id == rep.report_id

    # Verify ChromaDB indexing and querying
    query_vec = [0.5] * dim
    results = vector_store.query_community_reports(query_vec, top_k=2)
    assert len(results) == 1
    assert results[0]["id"] == "COMM_REPORT_0000"
    assert results[0]["metadata"]["community_id"] == 0
