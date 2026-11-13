"""Unit tests for Phase 1, Step 1: Configuration, Ontology, and Core Pydantic Models."""

from pathlib import Path
import pytest
from pydantic import ValidationError
from src.core.configuration import ChunkingConfig, load_engine_config
from src.core.models import (
    CanonicalEntity,
    Chunk,
    EntityMention,
    Polarity,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    ValidationStatus,
)


def test_load_all_four_domain_configs(tmp_path: Path) -> None:
    domain_files = [
        "configs/domains/academic.yaml",
        "configs/domains/company.yaml",
        "configs/domains/medical_research.yaml",
        "configs/domains/legal.yaml",
    ]
    seen_hashes = set()
    seen_kb_ids = set()

    for file_path in domain_files:
        cfg = load_engine_config(Path(file_path))
        assert cfg.chunking.chunk_size == 1100
        assert cfg.chunking.chunk_overlap == 180
        assert cfg.ollama.expected_embedding_dim == 768
        assert cfg.retrieval.rrf_k == 60
        assert cfg.leiden.resolution_parameter == 1.0

        cfg_hash = cfg.compute_config_hash()
        assert len(cfg_hash) == 64
        seen_hashes.add(cfg_hash)
        seen_kb_ids.add(cfg.kb_id)

        # Verify isolated directory creation in tmp_path
        paths = cfg.initialize_kb_directories(base_dir=tmp_path)
        assert (paths["kb_root"] / "config_snapshot.yaml").exists()
        assert paths["processed"].exists()
        assert paths["graph"].exists()
        assert paths["vector_store"].exists()

    assert len(seen_hashes) == 4
    assert len(seen_kb_ids) == 4


def test_chunking_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=500, chunk_overlap=500)


def test_academic_ontology_triple_validation() -> None:
    cfg = load_engine_config(Path("configs/domains/academic.yaml"))
    ont = cfg.ontology

    # Valid triples from our teaching example
    ok, err = ont.validate_triple_types("person", "works_under", "person")
    assert ok is True and err is None

    ok, err = ont.validate_triple_types("PERSON", "SUPERVISES", "PROJECT")
    assert ok is True and err is None

    ok, err = ont.validate_triple_types("PROJECT", "USES", "TECHNOLOGY")
    assert ok is True and err is None

    # Invalid endpoint direction: TECHNOLOGY cannot SUPERVISE a PROJECT
    ok, err = ont.validate_triple_types("TECHNOLOGY", "SUPERVISES", "PROJECT")
    assert ok is False
    assert "not permitted" in (err or "")

    # Unknown relation label
    ok, err = ont.validate_triple_types("PERSON", "INVENTED_BY_ALIENS", "PROJECT")
    assert ok is False
    assert "Unknown relationship type" in (err or "")


def test_models_normalize_and_preserve_provenance() -> None:
    chunk = Chunk(
        kb_id="academic_kb",
        chunk_id="C001",
        doc_id="D001",
        filename="sample.pdf",
        page_numbers=[1, 2],
        chunk_index=0,
        text="Rahul studies at IIIT Naya Raipur.",
        char_count=34,
        content_hash="abc123hash",
    )
    assert chunk.page_numbers == [1, 2]

    mention = EntityMention(
        kb_id="academic_kb",
        mention_id="M001",
        surface_name="Rahul",
        normalized_name="rahul",
        entity_type="person",
        source_chunk_id="C001",
        source_doc_id="D001",
        page_numbers=[1],
    )
    assert mention.entity_type == "PERSON"

    rel = Relationship(
        kb_id="academic_kb",
        relationship_id="R001",
        source_entity_id="E001",
        target_entity_id="E002",
        relation_type="works under",
        source_chunk_ids=["C001"],
        evidence_quotes=["Rahul works under Professor Sharma."],
        polarity=Polarity.POSITIVE,
        validation_status=ValidationStatus.VALIDATED,
    )
    assert rel.relation_type == "WORKS_UNDER"
