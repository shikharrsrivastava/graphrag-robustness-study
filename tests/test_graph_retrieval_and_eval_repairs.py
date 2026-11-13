"""Comprehensive regression tests for targeted repairs in Hybrid GraphRAG.

Tests:
1. Seed entity retrieval and person name variation resolution (diminutives, affixes).
2. Seed entities returning source chunks without accepted incident edges.
3. Multi-seed balanced graph retrieval (preventing seed starvation).
4. Fast-path deterministic summarization for singleton communities.
5. Canonical answer extraction and benchmark evaluation (EM and F1).
6. Expanded ontology relationships (WORKS_ON, AUTHORED, MEMBER_OF on films/works, NATIONALITY).
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import igraph
from src.core.configuration import EngineConfig, LeidenConfig, ResolutionConfig
from src.core.models import (
    CanonicalEntity,
    Chunk,
    EntityMention,
    Evidence,
    ProvenanceRoute,
    QueryPlan,
    SeedEntitySpec,
    ValidationStatus,
)
from src.core.ontology import DomainOntology, RelationConstraint
from src.communities.community_summarizer import CommunitySummarizer
from src.extraction.schemas import OllamaLLMClient
from src.generation.answer_generator import AnswerGenerationOutput, OllamaAnswerGenerator
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.resolution.entity_resolver import (
    ConservativeEntityResolver,
    person_names_compatible,
    strip_name_affixes,
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore
from run_hotpot import extract_canonical_candidate, normalize_answer, token_f1


def test_person_names_compatible_variations() -> None:
    """Verify that diminutive, middle-name, and suffix variations match while father/son are distinguished."""
    # Diminutives and middle name / suffix variations
    assert person_names_compatible("Ed Wood", "Edward Davis Wood Jr.") is True
    assert person_names_compatible("Edward Wood", "Ed Wood") is True
    assert person_names_compatible("Eddie Wood", "Edward Davis Wood Jr.") is True
    assert person_names_compatible("Bob Smith", "Robert J. Smith") is True
    assert person_names_compatible("Bill Murray", "William James Murray") is True

    # Distinction between Sr. and Jr. / non-Sr.
    assert person_names_compatible("Ed Wood", "Ed Wood Sr.") is False
    assert person_names_compatible("Ed Wood Sr.", "Edward Davis Wood Jr.") is False
    assert person_names_compatible("Ed Wood Sr.", "Ed Wood Sr.") is True

    # Non-matching individuals
    assert person_names_compatible("Scott Derrickson", "Ed Wood") is False
    assert person_names_compatible("Conrad Brooks", "Ed Wood") is False


def test_seed_retriever_resolves_compatible_name_variations(tmp_path: Path) -> None:
    """Verify that SeedEntityRetriever resolves candidate entities matching person name variations."""
    ds = DocumentStore(kb_id="test_seed_kb", storage_root=tmp_path / "kb")
    entities = [
        CanonicalEntity(
            kb_id="test_seed_kb",
            entity_id="E001",
            name="Ed Wood",
            normalized_name="ed wood",
            entity_type="FILM",
            description="A film",
            source_chunk_ids=["C_FILM"],
            source_doc_ids=["D_FILM"],
        ),
        CanonicalEntity(
            kb_id="test_seed_kb",
            entity_id="E040",
            name="Edward Davis Wood Jr.",
            normalized_name="edward davis wood jr",
            entity_type="PERSON",
            description="Director and filmmaker",
            source_chunk_ids=["C_BIO"],
            source_doc_ids=["D_BIO"],
        ),
        CanonicalEntity(
            kb_id="test_seed_kb",
            entity_id="E069",
            name="Ed Wood",
            normalized_name="ed wood",
            entity_type="PERSON",
            description="Director mentioned in passing",
            source_chunk_ids=["C_PASSING"],
            source_doc_ids=["D_PASSING"],
        ),
    ]
    ds.save_canonical_entities(entities)

    class MockEmbeddingService:
        def embed_text(self, text: str) -> list[float]:
            return [0.1] * 768

    class MockVectorStore:
        def query_entities(self, **kwargs) -> list[dict]:
            return []

    retriever = SeedEntityRetriever(
        doc_store=ds,
        vector_store=MockVectorStore(),  # type: ignore
        embedding_provider=MockEmbeddingService(),  # type: ignore
    )

    spec = SeedEntitySpec(name="Ed Wood", expected_type="PERSON")
    res = retriever.resolve_seeds([spec])

    # Both E069 (exact PERSON match) and E040 (compatible name variation) must be returned
    assert "E069" in res.resolved_entity_ids
    assert "E040" in res.resolved_entity_ids
    assert "E001" not in res.resolved_entity_ids  # FILM excluded by PERSON type


def test_isolated_seed_entity_returns_direct_chunks(tmp_path: Path) -> None:
    """Verify that an isolated seed entity with zero incident edges returns its source chunk."""
    ds = DocumentStore(kb_id="isolated_kb", storage_root=tmp_path / "kb")
    text = "Edward Davis Wood Jr. was an American filmmaker."
    chunk = Chunk(
        kb_id="isolated_kb",
        chunk_id="C_ED_WOOD",
        doc_id="D_ED_WOOD",
        filename="ed_wood.txt",
        text=text,
        page_numbers=[1],
        chunk_index=0,
        char_count=len(text),
        content_hash="h1",
    )
    ds.save_chunks([chunk])
    entity = CanonicalEntity(
        kb_id="isolated_kb",
        entity_id="E040",
        name="Edward Davis Wood Jr.",
        normalized_name="edward davis wood jr",
        entity_type="PERSON",
        description="Filmmaker",
        source_chunk_ids=["C_ED_WOOD"],
        source_doc_ids=["D_ED_WOOD"],
    )
    ds.save_canonical_entities([entity])

    # Construct graph with isolated node
    g = igraph.Graph(directed=True)
    g.add_vertex(
        name="E040",
        entity_id="E040",
        label="Edward Davis Wood Jr.",
        entity_type="PERSON",
        source_chunk_ids=["C_ED_WOOD"],
    )
    assert g.ecount() == 0  # Zero edges in graph

    class MockSeedRetriever:
        def resolve_seeds(self, seeds):
            return SeedResolutionResult(resolved_entity_ids=["E040"], ambiguous_seeds={}, unresolved_seeds=[])
        def _resolve_single_spec(self, spec, entities):
            return ["E040"], False

    graph_retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=MockSeedRetriever(),  # type: ignore
        graph=g,
    )
    plan = QueryPlan(
        question="What was Ed Wood's nationality?",
        intent="LOCAL",
        seed_entities=[SeedEntitySpec(name="Ed Wood", expected_type="PERSON")],
    )
    evidence = graph_retriever.retrieve(plan.question, plan, ds.kb_id)

    assert len(evidence) == 1
    assert evidence[0].chunk_id == "C_ED_WOOD"
    assert evidence[0].graph_rank == 1
    assert any("Seed Entity" in path for path in evidence[0].supporting_graph_paths)


def test_multi_seed_balanced_graph_retrieval(tmp_path: Path) -> None:
    """Verify that multi-seed queries balance evidence across seeds and do not starve seeds with fewer chunks."""
    ds = DocumentStore(kb_id="balanced_kb", storage_root=tmp_path / "kb")
    chunks = [
        Chunk(
            kb_id="balanced_kb",
            chunk_id=f"C_SCOTT_{i}",
            doc_id=f"D_S{i}",
            filename=f"s{i}.txt",
            text=f"Scott text {i}",
            page_numbers=[1],
            chunk_index=0,
            char_count=len(f"Scott text {i}"),
            content_hash=f"h_s{i}",
        )
        for i in range(5)
    ]
    ed_text = "Ed Wood text"
    chunks.append(
        Chunk(
            kb_id="balanced_kb",
            chunk_id="C_ED_WOOD",
            doc_id="D_ED",
            filename="ed.txt",
            text=ed_text,
            page_numbers=[1],
            chunk_index=0,
            char_count=len(ed_text),
            content_hash="h_ed",
        )
    )
    ds.save_chunks(chunks)

    e_scott = CanonicalEntity(
        kb_id="balanced_kb",
        entity_id="E_SCOTT",
        name="Scott Derrickson",
        normalized_name="scott derrickson",
        entity_type="PERSON",
        description="Director",
        source_chunk_ids=[f"C_SCOTT_{i}" for i in range(5)],
        source_doc_ids=[f"D_S{i}" for i in range(5)],
    )
    e_ed = CanonicalEntity(
        kb_id="balanced_kb",
        entity_id="E_ED",
        name="Ed Wood",
        normalized_name="ed wood",
        entity_type="PERSON",
        description="Director",
        source_chunk_ids=["C_ED_WOOD"],
        source_doc_ids=["D_ED"],
    )
    ds.save_canonical_entities([e_scott, e_ed])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_SCOTT", entity_id="E_SCOTT", label="Scott Derrickson", entity_type="PERSON", source_chunk_ids=[f"C_SCOTT_{i}" for i in range(5)])
    g.add_vertex(name="E_ED", entity_id="E_ED", label="Ed Wood", entity_type="PERSON", source_chunk_ids=["C_ED_WOOD"])

    class MockSeedRetriever:
        def resolve_seeds(self, seeds):
            return SeedResolutionResult(resolved_entity_ids=["E_SCOTT", "E_ED"], ambiguous_seeds={}, unresolved_seeds=[])
        def _resolve_single_spec(self, spec, entities):
            if "Scott" in spec.name:
                return ["E_SCOTT"], False
            return ["E_ED"], False

    graph_retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=MockSeedRetriever(),  # type: ignore
        graph=g,
    )
    plan = QueryPlan(
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        intent="LOCAL",
        seed_entities=[
            SeedEntitySpec(name="Scott Derrickson", expected_type="PERSON"),
            SeedEntitySpec(name="Ed Wood", expected_type="PERSON"),
        ],
    )
    evidence = graph_retriever.retrieve(plan.question, plan, ds.kb_id)

    # Top 2 must contain both Scott Derrickson's top chunk and Ed Wood's top chunk!
    top_2_chunks = {evidence[0].chunk_id, evidence[1].chunk_id}
    assert "C_ED_WOOD" in top_2_chunks
    assert any(c.startswith("C_SCOTT") for c in top_2_chunks)


def test_singleton_community_summarizer_fast_path(tmp_path: Path) -> None:
    """Verify that singleton communities bypass LLM summarisation and produce grounded deterministic reports."""
    g = igraph.Graph(directed=True)
    g.add_vertex(
        name="E040",
        entity_id="E040",
        label="Edward Davis Wood Jr.",
        entity_type="PERSON",
        description="American filmmaker, actor, and director",
        source_chunk_ids=["C005"],
    )

    llm_call_count = 0

    def mock_generator(prompt: str) -> str:
        nonlocal llm_call_count
        llm_call_count += 1
        return json.dumps({"summary_text": "LLM generated summary"})

    mock_client = OllamaLLMClient(custom_generator=mock_generator)
    summarizer = CommunitySummarizer(llm_client=mock_client)

    communities = [
        {
            "community_id": 0,
            "member_entity_ids": ["E040"],
            "partition_id": "test_part",
        }
    ]

    reports = summarizer.summarize_communities(communities, g)
    assert len(reports) == 1
    # LLM must NOT have been called for singleton community!
    assert llm_call_count == 0
    assert "Edward Davis Wood Jr." in reports[0].summary_text
    assert "American filmmaker" in reports[0].summary_text
    assert reports[0].member_entity_ids == ["E040"]


def test_canonical_answer_extraction_and_evaluation() -> None:
    """Verify canonical answer extraction for exact match and token F1 against gold benchmarks."""
    gold = "yes"
    explanatory_answer = (
        "Scott Derrickson was born in 1966 [C002, D002, p.1], while Ed Wood was born in 1924 [C005, D005, p.1]. "
        "Therefore, they were of the same nationality, which is American [C002] and [C005]."
    )

    # 1. When model outputs explicit canonical_answer:
    cand_1 = extract_canonical_candidate(explanatory_answer, canonical_answer="yes", question="Were they of the same nationality?")
    assert normalize_answer(cand_1) == normalize_answer(gold)
    assert token_f1(cand_1, gold) == 1.0

    # 2. When canonical_answer is extracted from conclusion pattern:
    cand_2 = extract_canonical_candidate(explanatory_answer, canonical_answer=None, question="Were Scott Derrickson and Ed Wood of the same nationality?")
    assert cand_2 == "yes"
    assert normalize_answer(cand_2) == normalize_answer(gold)
    assert token_f1(cand_2, gold) == 1.0

    # 3. Negative question check:
    neg_answer = "Scott Derrickson is American, while Akira Kurosawa was Japanese. Therefore, they were not of the same nationality."
    cand_3 = extract_canonical_candidate(neg_answer, canonical_answer=None, question="Were they of the same nationality?")
    assert cand_3 == "no"


def test_academic_ontology_expanded_types() -> None:
    """Verify that academic ontology permits creative works for WORKS_ON, AUTHORED, and supports NATIONALITY."""
    from src.core.configuration import load_engine_config
    config = load_engine_config(Path("configs/domains/academic.yaml"))
    ontology = config.ontology

    # WORKS_ON on FILM and WORK
    ok, err = ontology.validate_triple_types("PERSON", "WORKS_ON", "FILM")
    assert ok is True, err
    ok, err = ontology.validate_triple_types("PERSON", "WORKS_ON", "WORK")
    assert ok is True, err

    # AUTHORED on FILM, BOOK, WORK
    ok, err = ontology.validate_triple_types("PERSON", "AUTHORED", "FILM")
    assert ok is True, err
    ok, err = ontology.validate_triple_types("PERSON", "AUTHORED", "BOOK")
    assert ok is True, err

    # MEMBER_OF on FILM (cast)
    ok, err = ontology.validate_triple_types("PERSON", "MEMBER_OF", "FILM")
    assert ok is True, err

    # NATIONALITY
    ok, err = ontology.validate_triple_types("PERSON", "NATIONALITY", "COUNTRY")
    assert ok is True, err
    ok, err = ontology.validate_triple_types("PERSON", "NATIONALITY", "LOCATION")
    assert ok is True, err


def test_conservative_entity_resolver_false_positive_protection() -> None:
    """Verify that distinct people sharing first/last names or middle initial omissions do NOT auto-merge."""
    ontology = DomainOntology(
        domain_name="test",
        entity_types={"PERSON": "People"},
        relationship_types={
            "WORKS_WITH": RelationConstraint(name="WORKS_WITH", description="Colleague")
        },
        static_aliases={},
    )
    resolver = ConservativeEntityResolver(
        resolution_config=ResolutionConfig(require_type_match=True),
        ontology=ontology,
    )

    # 1. Michael Jordan vs Michael B. Jordan
    m_mj = EntityMention(
        kb_id="test_kb",
        mention_id="M_MJ",
        surface_name="Michael Jordan",
        normalized_name="michael jordan",
        entity_type="PERSON",
        source_chunk_id="C1",
        source_doc_id="D1",
        page_numbers=[1],
    )
    m_mbj = EntityMention(
        kb_id="test_kb",
        mention_id="M_MBJ",
        surface_name="Michael B. Jordan",
        normalized_name="michael b. jordan",
        entity_type="PERSON",
        source_chunk_id="C2",
        source_doc_id="D2",
        page_numbers=[1],
    )
    canonicals_mj, _ = resolver.resolve_mentions([m_mj, m_mbj])
    assert len(canonicals_mj) == 2, "Michael Jordan and Michael B. Jordan must not auto-merge"
    assert {c.name for c in canonicals_mj} == {"Michael Jordan", "Michael B. Jordan"}

    # 2. George Bush vs George W. Bush
    m_gb = EntityMention(
        kb_id="test_kb",
        mention_id="M_GB",
        surface_name="George Bush",
        normalized_name="george bush",
        entity_type="PERSON",
        source_chunk_id="C3",
        source_doc_id="D3",
        page_numbers=[1],
    )
    m_gwb = EntityMention(
        kb_id="test_kb",
        mention_id="M_GWB",
        surface_name="George W. Bush",
        normalized_name="george w. bush",
        entity_type="PERSON",
        source_chunk_id="C4",
        source_doc_id="D4",
        page_numbers=[1],
    )
    canonicals_gb, _ = resolver.resolve_mentions([m_gb, m_gwb])
    assert len(canonicals_gb) == 2, "George Bush and George W. Bush must not auto-merge"
    assert {c.name for c in canonicals_gb} == {"George Bush", "George W. Bush"}

    # 3. John Smith vs John F. Smith
    m_js = EntityMention(
        kb_id="test_kb",
        mention_id="M_JS",
        surface_name="John Smith",
        normalized_name="john smith",
        entity_type="PERSON",
        source_chunk_id="C5",
        source_doc_id="D5",
        page_numbers=[1],
    )
    m_jfs = EntityMention(
        kb_id="test_kb",
        mention_id="M_JFS",
        surface_name="John F. Smith",
        normalized_name="john f. smith",
        entity_type="PERSON",
        source_chunk_id="C6",
        source_doc_id="D6",
        page_numbers=[1],
    )
    canonicals_js, _ = resolver.resolve_mentions([m_js, m_jfs])
    assert len(canonicals_js) == 2, "John Smith and John F. Smith must not auto-merge"
    assert {c.name for c in canonicals_js} == {"John Smith", "John F. Smith"}


def test_conservative_entity_resolver_generational_protection() -> None:
    """Verify that generational distinctions are preserved and suffix stripping never bypasses them."""
    ontology = DomainOntology(
        domain_name="test",
        entity_types={"PERSON": "People"},
        relationship_types={
            "WORKS_WITH": RelationConstraint(name="WORKS_WITH", description="Colleague")
        },
        static_aliases={},
    )
    resolver = ConservativeEntityResolver(
        resolution_config=ResolutionConfig(require_type_match=True),
        ontology=ontology,
    )

    # 1. Ed Wood Sr. vs Ed Wood Jr.
    m_sr = EntityMention(
        kb_id="test_kb",
        mention_id="M_SR",
        surface_name="Ed Wood Sr.",
        normalized_name="ed wood sr",
        entity_type="PERSON",
        source_chunk_id="C1",
        source_doc_id="D1",
        page_numbers=[1],
    )
    m_jr = EntityMention(
        kb_id="test_kb",
        mention_id="M_JR",
        surface_name="Ed Wood Jr.",
        normalized_name="ed wood jr",
        entity_type="PERSON",
        source_chunk_id="C2",
        source_doc_id="D2",
        page_numbers=[1],
    )
    canonicals_sr_jr, _ = resolver.resolve_mentions([m_sr, m_jr])
    assert len(canonicals_sr_jr) == 2, "Ed Wood Sr. and Ed Wood Jr. must not merge"
    assert {c.name for c in canonicals_sr_jr} == {"Ed Wood Sr.", "Ed Wood Jr."}

    # 2. Ed Wood vs Ed Wood Sr.
    m_plain = EntityMention(
        kb_id="test_kb",
        mention_id="M_PLAIN",
        surface_name="Ed Wood",
        normalized_name="ed wood",
        entity_type="PERSON",
        source_chunk_id="C3",
        source_doc_id="D3",
        page_numbers=[1],
    )
    canonicals_plain_sr, _ = resolver.resolve_mentions([m_plain, m_sr])
    assert len(canonicals_plain_sr) == 2, "Ed Wood and Ed Wood Sr. must not auto-merge via suffix stripping"
    assert {c.name for c in canonicals_plain_sr} == {"Ed Wood", "Ed Wood Sr."}

    # 3. Ed Wood vs Ed Wood Jr.
    canonicals_plain_jr, _ = resolver.resolve_mentions([m_plain, m_jr])
    assert len(canonicals_plain_jr) == 2, "Ed Wood and Ed Wood Jr. must not auto-merge via suffix stripping"
    assert {c.name for c in canonicals_plain_jr} == {"Ed Wood", "Ed Wood Jr."}

    # 4. Verify embedding similarity & LLM verification CANNOT bypass generational protection
    class MockHighSimilarityEmbedding:
        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    def mock_affirmative_llm(prompt: str) -> str:
        # LLM erroneously attempts to confirm merge between Sr. and Jr.
        return json.dumps({"is_same_entity": True, "reason": "Both are named Ed Wood"})

    mock_client = OllamaLLMClient(custom_generator=mock_affirmative_llm)
    bypass_resistant_resolver = ConservativeEntityResolver(
        resolution_config=ResolutionConfig(
            require_type_match=True,
            auto_merge_threshold=0.8,
            similarity_candidate_threshold=0.5,
            use_llm_verification_for_ambiguous=True,
        ),
        ontology=ontology,
        embedding_provider=MockHighSimilarityEmbedding(),  # type: ignore
        llm_client=mock_client,
    )

    # Even with 1.0 embedding similarity and affirmative LLM verdict, Ed Wood Sr. and Ed Wood Jr. MUST NOT MERGE (both directions)
    canonicals_sr_jr_emb, _ = bypass_resistant_resolver.resolve_mentions([m_sr, m_jr])
    assert len(canonicals_sr_jr_emb) == 2, "Generational conflict between Sr. and Jr. must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_sr_jr_emb} == {"Ed Wood Sr.", "Ed Wood Jr."}

    canonicals_jr_sr_emb, _ = bypass_resistant_resolver.resolve_mentions([m_jr, m_sr])
    assert len(canonicals_jr_sr_emb) == 2, "Generational conflict between Jr. and Sr. must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_jr_sr_emb} == {"Ed Wood Sr.", "Ed Wood Jr."}

    # Even with 1.0 embedding similarity and affirmative LLM verdict, Ed Wood and Ed Wood Sr. MUST NOT MERGE (both directions)
    canonicals_plain_sr_emb, _ = bypass_resistant_resolver.resolve_mentions([m_plain, m_sr])
    assert len(canonicals_plain_sr_emb) == 2, "Generational conflict between plain and Sr. must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_plain_sr_emb} == {"Ed Wood", "Ed Wood Sr."}

    canonicals_sr_plain_emb, _ = bypass_resistant_resolver.resolve_mentions([m_sr, m_plain])
    assert len(canonicals_sr_plain_emb) == 2, "Generational conflict between Sr. and plain must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_sr_plain_emb} == {"Ed Wood", "Ed Wood Sr."}

    # Even with 1.0 embedding similarity and affirmative LLM verdict, Ed Wood and Ed Wood Jr. MUST NOT MERGE (both directions)
    canonicals_plain_jr_emb, _ = bypass_resistant_resolver.resolve_mentions([m_plain, m_jr])
    assert len(canonicals_plain_jr_emb) == 2, "Generational conflict between plain and Jr. must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_plain_jr_emb} == {"Ed Wood", "Ed Wood Jr."}

    canonicals_jr_plain_emb, _ = bypass_resistant_resolver.resolve_mentions([m_jr, m_plain])
    assert len(canonicals_jr_plain_emb) == 2, "Generational conflict between Jr. and plain must never merge via embeddings or LLM"
    assert {c.name for c in canonicals_jr_plain_emb} == {"Ed Wood", "Ed Wood Jr."}


def test_conservative_entity_resolver_legitimate_variants_and_query_discovery(tmp_path: Path) -> None:
    """Verify that legitimate suffix/honorific variants merge during ingestion while query discovery finds candidates without mutating them."""
    ontology = DomainOntology(
        domain_name="test",
        entity_types={"PERSON": "People"},
        relationship_types={
            "WORKS_WITH": RelationConstraint(name="WORKS_WITH", description="Colleague")
        },
        static_aliases={},
    )
    resolver = ConservativeEntityResolver(
        resolution_config=ResolutionConfig(require_type_match=True),
        ontology=ontology,
    )

    # 1. Legitimate suffix formatting variants (same generational suffix, punctuation differences)
    m_jr1 = EntityMention(
        kb_id="test_kb",
        mention_id="M_JR1",
        surface_name="Edward Davis Wood Jr.",
        normalized_name="edward davis wood jr",
        entity_type="PERSON",
        source_chunk_id="C1",
        source_doc_id="D1",
        page_numbers=[1],
    )
    m_jr2 = EntityMention(
        kb_id="test_kb",
        mention_id="M_JR2",
        surface_name="Edward Davis Wood, Jr.",
        normalized_name="edward davis wood, jr",
        entity_type="PERSON",
        source_chunk_id="C2",
        source_doc_id="D2",
        page_numbers=[1],
    )
    canonicals_jr, _ = resolver.resolve_mentions([m_jr1, m_jr2])
    assert len(canonicals_jr) == 1, "Punctuation variants of the same suffix must merge"
    assert canonicals_jr[0].source_mention_ids == ["M_JR1", "M_JR2"]

    # 2. Surname-only honorific variants: Professor Sharma vs Dr. Sharma MUST REMAIN SEPARATE
    # Surnames alone after honorific stripping are not proof of identity without given names or static aliases.
    m_prof = EntityMention(
        kb_id="test_kb",
        mention_id="M_PROF",
        surface_name="Professor Sharma",
        normalized_name="professor sharma",
        entity_type="PERSON",
        source_chunk_id="C3",
        source_doc_id="D3",
        page_numbers=[1],
    )
    m_dr = EntityMention(
        kb_id="test_kb",
        mention_id="M_DR",
        surface_name="Dr. Sharma",
        normalized_name="dr. sharma",
        entity_type="PERSON",
        source_chunk_id="C4",
        source_doc_id="D4",
        page_numbers=[1],
    )
    canonicals_sharma, _ = resolver.resolve_mentions([m_prof, m_dr])
    assert len(canonicals_sharma) == 2, "Surname-only honorific matches must not auto-merge without given name or static aliases"
    assert {c.name for c in canonicals_sharma} == {"Professor Sharma", "Dr. Sharma"}

    # 3. Sufficiently informative honorific variants: Professor Ananya Sharma vs Dr. Ananya Sharma
    # When full given name + surname is present ('Ananya Sharma'), title variants DO merge safely.
    m_prof_ananya = EntityMention(
        kb_id="test_kb",
        mention_id="M_PROF_ANANYA",
        surface_name="Professor Ananya Sharma",
        normalized_name="professor ananya sharma",
        entity_type="PERSON",
        source_chunk_id="C5",
        source_doc_id="D5",
        page_numbers=[1],
    )
    m_dr_ananya = EntityMention(
        kb_id="test_kb",
        mention_id="M_DR_ANANYA",
        surface_name="Dr. Ananya Sharma",
        normalized_name="dr. ananya sharma",
        entity_type="PERSON",
        source_chunk_id="C6",
        source_doc_id="D6",
        page_numbers=[1],
    )
    canonicals_ananya, _ = resolver.resolve_mentions([m_prof_ananya, m_dr_ananya])
    assert len(canonicals_ananya) == 1, "Informative personal names across title variants must merge"
    assert canonicals_ananya[0].source_mention_ids == ["M_PROF_ANANYA", "M_DR_ANANYA"]

    # 3. Query-time seed candidate discovery: both E069 and E040 remain distinct and discoverable
    ds = DocumentStore(kb_id="query_discovery_kb", storage_root=tmp_path / "kb")
    entities = [
        CanonicalEntity(
            kb_id="query_discovery_kb",
            entity_id="E040",
            name="Edward Davis Wood Jr.",
            normalized_name="edward davis wood jr",
            entity_type="PERSON",
            description="American filmmaker",
            source_chunk_ids=["C005"],
            source_doc_ids=["D005"],
        ),
        CanonicalEntity(
            kb_id="query_discovery_kb",
            entity_id="E069",
            name="Ed Wood",
            normalized_name="ed wood",
            entity_type="PERSON",
            description="Director",
            source_chunk_ids=["C009"],
            source_doc_ids=["D009"],
        ),
    ]
    ds.save_canonical_entities(entities)

    class MockEmbeddingService:
        def embed_text(self, text: str) -> list[float]:
            return [0.1] * 768

    class MockVectorStore:
        def query_entities(self, **kwargs) -> list[dict]:
            return []

    seed_retriever = SeedEntityRetriever(
        doc_store=ds,
        vector_store=MockVectorStore(),  # type: ignore
        embedding_provider=MockEmbeddingService(),  # type: ignore
    )
    spec = SeedEntitySpec(name="Ed Wood", expected_type="PERSON")
    res = seed_retriever.resolve_seeds([spec])

    # Must discover both candidates without mutating or merging them
    assert set(res.resolved_entity_ids) == {"E040", "E069"}
    assert res.ambiguous_seeds.get("Ed Wood") == ["E069", "E040"] or "Ed Wood" in res.ambiguous_seeds
    # Ensure entities on disk remain untouched
    reloaded_entities = ds.load_canonical_entities()
    assert len(reloaded_entities) == 2
    assert {e.entity_id for e in reloaded_entities} == {"E040", "E069"}
