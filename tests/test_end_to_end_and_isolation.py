"""End-to-End Engine Orchestrator and Multi-Domain Isolation Unit Tests."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pytest
from src.core.configuration import load_engine_config
from src.core.models import (
    ProvenanceRoute,
    QueryIntent,
    RetrievalResult,
)
from src.engine import HybridGraphRAGEngine
from src.extraction.schemas import OllamaLLMClient


class DeterministicEmbedder:
    """Offline unit vector embedder seeded by text MD5 for deterministic semantic similarity."""

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


def create_mock_llm_generator() -> OllamaLLMClient:
    """Mock LLM generator handling entity extraction, relationship extraction, query planning, summarization, and QA."""

    def mock_generate(prompt: str) -> str:
        # 1. Entity Extraction
        if "ALLOWED ENTITY TYPES" in prompt or "Extract all named entities" in prompt:
            if "Rahul studies at IIIT Naya Raipur" in prompt:
                return json.dumps(
                    {
                        "entities": [
                            {"surface_name": "Rahul", "entity_type": "PERSON", "description": "Student researcher"},
                            {"surface_name": "IIIT Naya Raipur", "entity_type": "UNIVERSITY", "description": "Higher education institute"},
                            {"surface_name": "Professor Sharma", "entity_type": "PERSON", "description": "Academic supervisor"},
                        ]
                    }
                )
            elif "Professor Sharma supervises the Medical AI project" in prompt:
                return json.dumps(
                    {
                        "entities": [
                            {"surface_name": "Professor Sharma", "entity_type": "PERSON", "description": "Professor supervising AI"},
                            {"surface_name": "Medical AI", "entity_type": "PROJECT", "description": "Medical artificial intelligence initiative"},
                        ]
                    }
                )
            elif "Medical AI uses machine learning" in prompt:
                return json.dumps(
                    {
                        "entities": [
                            {"surface_name": "Medical AI", "entity_type": "PROJECT", "description": "Healthcare intelligence project"},
                            {"surface_name": "machine learning", "entity_type": "TECHNOLOGY", "description": "ML algorithms"},
                            {"surface_name": "diseases", "entity_type": "CONCEPT", "description": "Target health conditions"},
                        ]
                    }
                )
            elif "Priya manages the CloudSec product" in prompt:
                return json.dumps(
                    {
                        "entities": [
                            {"surface_name": "Priya", "entity_type": "EMPLOYEE", "description": "Product engineering lead"},
                            {"surface_name": "CloudSec", "entity_type": "PRODUCT", "description": "Cloud security platform"},
                            {"surface_name": "Platform Department", "entity_type": "DEPARTMENT", "description": "Core infrastructure group"},
                        ]
                    }
                )
            elif "CloudSec is built with Rust" in prompt:
                return json.dumps(
                    {
                        "entities": [
                            {"surface_name": "CloudSec", "entity_type": "PRODUCT", "description": "Security software"},
                            {"surface_name": "Rust", "entity_type": "TECHNOLOGY", "description": "Systems programming language"},
                            {"surface_name": "Acme Corp", "entity_type": "CLIENT", "description": "Enterprise customer"},
                        ]
                    }
                )
            return json.dumps({"entities": []})

        # 2. Relationship Extraction
        if "ALLOWED RELATIONSHIP TYPES" in prompt or "IDENTIFIED ENTITY MENTIONS" in prompt:
            if "Rahul studies at IIIT Naya Raipur" in prompt:
                return json.dumps(
                    {
                        "relationships": [
                            {
                                "source_mention_id": "Rahul",
                                "target_mention_id": "IIIT Naya Raipur",
                                "relation_type": "STUDIES_AT",
                                "evidence_quote": "Rahul studies at IIIT Naya Raipur.",
                                "polarity": "POSITIVE",
                            },
                            {
                                "source_mention_id": "Rahul",
                                "target_mention_id": "Professor Sharma",
                                "relation_type": "WORKS_UNDER",
                                "evidence_quote": "Rahul works under Professor Sharma.",
                                "polarity": "POSITIVE",
                            },
                        ]
                    }
                )
            elif "Professor Sharma supervises the Medical AI project" in prompt:
                return json.dumps(
                    {
                        "relationships": [
                            {
                                "source_mention_id": "Professor Sharma",
                                "target_mention_id": "Medical AI",
                                "relation_type": "SUPERVISES",
                                "evidence_quote": "Professor Sharma supervises the Medical AI project.",
                                "polarity": "POSITIVE",
                            }
                        ]
                    }
                )
            elif "Medical AI uses machine learning" in prompt:
                return json.dumps(
                    {
                        "relationships": [
                            {
                                "source_mention_id": "Medical AI",
                                "target_mention_id": "machine learning",
                                "relation_type": "USES",
                                "evidence_quote": "Medical AI uses machine learning to detect diseases.",
                                "polarity": "POSITIVE",
                            }
                        ]
                    }
                )
            elif "Priya manages the CloudSec product" in prompt:
                return json.dumps(
                    {
                        "relationships": [
                            {
                                "source_mention_id": "Priya",
                                "target_mention_id": "CloudSec",
                                "relation_type": "MANAGES",
                                "evidence_quote": "Priya manages the CloudSec product in the Platform Department.",
                                "polarity": "POSITIVE",
                            },
                            {
                                "source_mention_id": "CloudSec",
                                "target_mention_id": "Platform Department",
                                "relation_type": "BELONGS_TO",
                                "evidence_quote": "Priya manages the CloudSec product in the Platform Department.",
                                "polarity": "POSITIVE",
                            },
                        ]
                    }
                )
            elif "CloudSec is built with Rust" in prompt:
                return json.dumps(
                    {
                        "relationships": [
                            {
                                "source_mention_id": "CloudSec",
                                "target_mention_id": "Rust",
                                "relation_type": "BUILT_WITH",
                                "evidence_quote": "CloudSec is built with Rust and serves Acme Corp.",
                                "polarity": "POSITIVE",
                            },
                            {
                                "source_mention_id": "CloudSec",
                                "target_mention_id": "Acme Corp",
                                "relation_type": "SERVES",
                                "evidence_quote": "CloudSec is built with Rust and serves Acme Corp.",
                                "polarity": "POSITIVE",
                            },
                        ]
                    }
                )
            return json.dumps({"relationships": []})

        # 3. Community Summary
        if "COMMUNITY ID:" in prompt or "community of related entities" in prompt:
            if "company_knowledge" in prompt or "CloudSec" in prompt:
                return json.dumps({"summary_text": "Platform Engineering community developing CloudSec security platform."})
            return json.dumps({"summary_text": "Medical AI and Healthcare Machine Learning research group at IIIT Naya Raipur."})

        # 4. Query Planning
        if "DOMAIN ONTOLOGY CONSTRAINTS" in prompt:
            if "Who supervises the Medical AI project?" in prompt:
                return json.dumps(
                    {
                        "question": "Who supervises the Medical AI project?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "Medical AI", "expected_type": "PROJECT"}],
                        "target_type": "PERSON",
                        "relationship_path": ["SUPERVISES"],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "What technology does the project supervised by Rahul's mentor use?" in prompt:
                return json.dumps(
                    {
                        "question": "What technology does the project supervised by Rahul's mentor use?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "Rahul", "expected_type": "PERSON"}],
                        "target_type": "TECHNOLOGY",
                        "relationship_path": ["WORKS_UNDER", "SUPERVISES", "USES"],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "Summarize the primary research themes across the projects." in prompt:
                return json.dumps(
                    {
                        "question": "Summarize the primary research themes across the projects.",
                        "intent": "GLOBAL",
                        "seed_entities": [],
                        "target_type": None,
                        "relationship_path": [],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "Who works under Prof. Sharma?" in prompt:
                return json.dumps(
                    {
                        "question": "Who works under Prof. Sharma?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "Prof. Sharma", "expected_type": "PERSON"}],
                        "target_type": "PERSON",
                        "relationship_path": ["WORKS_UNDER"],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "What is the capital of France?" in prompt:
                return json.dumps(
                    {
                        "question": "What is the capital of France?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "France", "expected_type": None}],
                        "target_type": None,
                        "relationship_path": [],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "What is CloudSec built with?" in prompt:
                return json.dumps(
                    {
                        "question": "What is CloudSec built with?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "CloudSec", "expected_type": "PRODUCT"}],
                        "target_type": "TECHNOLOGY",
                        "relationship_path": ["BUILT_WITH"],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            elif "Who is Priya?" in prompt:
                return json.dumps(
                    {
                        "question": "Who is Priya?",
                        "intent": "LOCAL",
                        "seed_entities": [{"name": "Priya", "expected_type": "EMPLOYEE"}],
                        "target_type": None,
                        "relationship_path": ["MANAGES"],
                        "requires_clarification": False,
                        "clarification_reason": None,
                    }
                )
            return json.dumps({"question": "q", "intent": "LOCAL", "seed_entities": []})

        # 5. Answer Generation
        if "EVIDENCE CONTEXT:" in prompt:
            if "capital of France" in prompt or "France" in prompt:
                return json.dumps(
                    {
                        "answer": "The context does not contain any information about France.",
                        "abstained": True,
                        "cited_chunk_ids": [],
                    }
                )
            elif "What technology" in prompt or "mentor use" in prompt:
                return json.dumps(
                    {
                        "answer": "Medical AI uses machine learning to detect diseases [D003_C0000, D003, p.1].",
                        "abstained": False,
                        "cited_chunk_ids": ["D003_C0000"],
                    }
                )
            elif "Who supervises the Medical AI project?" in prompt:
                return json.dumps(
                    {
                        "answer": "Professor Sharma supervises the Medical AI project [D002_C0000, D002, p.1].",
                        "abstained": False,
                        "cited_chunk_ids": ["D002_C0000"],
                    }
                )
            elif "Summarize the primary research themes" in prompt:
                return json.dumps(
                    {
                        "answer": "The primary research theme is Medical AI and Healthcare Machine Learning [D003_C0000, D003, p.1].",
                        "abstained": False,
                        "cited_chunk_ids": ["D003_C0000"],
                    }
                )
            elif "Who works under Prof. Sharma?" in prompt:
                return json.dumps(
                    {
                        "answer": "Rahul works under Professor Sharma [D001_C0000, D001, p.1].",
                        "abstained": False,
                        "cited_chunk_ids": ["D001_C0000"],
                    }
                )
            elif "What is CloudSec built with?" in prompt:
                return json.dumps(
                    {
                        "answer": "CloudSec is built with Rust [D002_C0000, D002, p.1].",
                        "abstained": False,
                        "cited_chunk_ids": ["D002_C0000"],
                    }
                )
            return json.dumps(
                {
                    "answer": "Grounded factual answer [D001_C0000, D001, p.1].",
                    "abstained": False,
                    "cited_chunk_ids": ["D001_C0000"],
                }
            )

        return json.dumps({})

    return OllamaLLMClient(custom_generator=mock_generate)


# --- End-to-End & Multi-Domain Isolation Test ---


def test_end_to_end_orchestration_and_domain_isolation(tmp_path: Path) -> None:
    """Test full 19-stage pipeline on academic_kb and company_kb, testing 5 query types and strict isolation."""
    # 1. Prepare raw document directories
    academic_docs_dir = tmp_path / "raw_academic"
    academic_docs_dir.mkdir()
    (academic_docs_dir / "doc1.txt").write_text(
        "Rahul studies at IIIT Naya Raipur. Rahul works under Professor Sharma.",
        encoding="utf-8",
    )
    (academic_docs_dir / "doc2.txt").write_text(
        "Professor Sharma supervises the Medical AI project.",
        encoding="utf-8",
    )
    (academic_docs_dir / "doc3.txt").write_text(
        "Medical AI uses machine learning to detect diseases.",
        encoding="utf-8",
    )

    company_docs_dir = tmp_path / "raw_company"
    company_docs_dir.mkdir()
    (company_docs_dir / "comp_doc1.txt").write_text(
        "Priya manages the CloudSec product in the Platform Department.",
        encoding="utf-8",
    )
    (company_docs_dir / "comp_doc2.txt").write_text(
        "CloudSec is built with Rust and serves Acme Corp.",
        encoding="utf-8",
    )

    # 2. Setup engines with shared embedder & mock LLM
    embedder = DeterministicEmbedder(dim=768)
    mock_llm = create_mock_llm_generator()

    academic_config = load_engine_config(
        domain_config_path=Path("configs/domains/academic.yaml"),
        kb_id_override="academic_kb",
    )
    company_config = load_engine_config(
        domain_config_path=Path("configs/domains/company.yaml"),
        kb_id_override="company_kb",
    )

    # Configure custom storage base_dir to tmp_path
    academic_engine = HybridGraphRAGEngine(
        config=academic_config,
        llm_client=mock_llm,
        embedding_service=embedder,
        base_dir=tmp_path,
    )
    company_engine = HybridGraphRAGEngine(
        config=company_config,
        llm_client=mock_llm,
        embedding_service=embedder,
        base_dir=tmp_path,
    )

    # 3. Index academic_kb (Stages 1-10)
    acad_stats = academic_engine.index_directory(academic_docs_dir)
    assert acad_stats["documents"] == 3
    assert acad_stats["chunks"] == 3
    assert acad_stats["entities"] >= 4  # Rahul, IIIT Naya Raipur, Professor Sharma, Medical AI, machine learning, diseases
    assert acad_stats["relationships"] >= 3
    assert acad_stats["manifest_fingerprint"] is not None

    # 4. Index company_kb (Stages 1-10)
    comp_stats = company_engine.index_directory(company_docs_dir)
    assert comp_stats["documents"] == 2
    assert comp_stats["chunks"] == 2
    assert comp_stats["entities"] >= 4  # Priya, CloudSec, Platform Department, Rust, Acme Corp
    assert comp_stats["relationships"] >= 3
    assert comp_stats["manifest_fingerprint"] is not None

    # 5. Verify Academic Queries (5 required question types)

    # Query Type 1: Single-hop traversal query
    ans_1, res_1 = academic_engine.query("Who supervises the Medical AI project?")
    assert not res_1.abstained
    assert "Professor Sharma" in ans_1
    assert any("D002" in ev.doc_id for ev in res_1.evidence_list)

    # Query Type 2: Multi-hop (3-hop) traversal: Rahul -> Professor Sharma -> Medical AI -> machine learning
    ans_2, res_2 = academic_engine.query("What technology does the project supervised by Rahul's mentor use?")
    assert not res_2.abstained
    assert "machine learning" in ans_2.lower()
    assert any("D003" in ev.doc_id for ev in res_2.evidence_list)

    # Query Type 3: Broad global research theme query
    ans_3, res_3 = academic_engine.query("Summarize the primary research themes across the projects.")
    assert not res_3.abstained
    assert len(res_3.community_reports) >= 1 or len(res_3.evidence_list) >= 1
    assert "Medical AI" in ans_3

    # Query Type 4: Alias query with honorific prefix ("Prof. Sharma" resolves to "Professor Sharma")
    ans_4, res_4 = academic_engine.query("Who works under Prof. Sharma?")
    assert not res_4.abstained
    assert "Rahul" in ans_4
    assert any("D001" in ev.doc_id for ev in res_4.evidence_list)

    # Query Type 5: Unanswerable query triggering clean abstention
    ans_5, res_5 = academic_engine.query("What is the capital of France?")
    assert res_5.abstained or "does not contain" in ans_5.lower() or "unable to answer" in ans_5.lower()

    # 6. Verify Company Queries
    ans_comp, res_comp = company_engine.query("What is CloudSec built with?")
    assert not res_comp.abstained
    assert "Rust" in ans_comp
    assert any("comp_doc2" in ev.filename or "D002" in ev.doc_id for ev in res_comp.evidence_list)

    # 7. Strict Multi-Domain Isolation Verification (Zero Cross-KB Leakage)

    # Academic KB must contain zero company entities or chunks
    academic_chunks = academic_engine.doc_store.list_all_chunks()
    assert all(c.kb_id == "academic_kb" for c in academic_chunks)
    assert not any("CloudSec" in c.text or "Priya" in c.text for c in academic_chunks)

    academic_entities = academic_engine.doc_store.load_canonical_entities()
    assert all(e.kb_id == "academic_kb" for e in academic_entities)
    assert not any("CloudSec" == e.name or "Priya" == e.name for e in academic_entities)

    # Company KB must contain zero academic entities or chunks
    company_chunks = company_engine.doc_store.list_all_chunks()
    assert all(c.kb_id == "company_kb" for c in company_chunks)
    assert not any("Rahul" in c.text or "Professor Sharma" in c.text for c in company_chunks)

    company_entities = company_engine.doc_store.load_canonical_entities()
    assert all(e.kb_id == "company_kb" for e in company_entities)
    assert not any("Rahul" == e.name or "Professor Sharma" == e.name for e in company_entities)

    # Querying academic KB for company-specific terms yields zero company evidence
    _, leak_check_1 = academic_engine.query("Who is Priya?", mode="vector_only")
    assert all(ev.kb_id == "academic_kb" for ev in leak_check_1.evidence_list)
    assert not any("Platform Department" in ev.text for ev in leak_check_1.evidence_list)

    # Querying company KB for academic-specific terms yields zero academic evidence
    _, leak_check_2 = company_engine.query("Where does Rahul study?", mode="vector_only")
    assert all(ev.kb_id == "company_kb" for ev in leak_check_2.evidence_list)
    assert not any("IIIT Naya Raipur" in ev.text for ev in leak_check_2.evidence_list)
