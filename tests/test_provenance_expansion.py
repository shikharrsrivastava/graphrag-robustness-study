"""Unit and regression tests for 1-hop Entity-Chunk-Entity bipartite provenance expansion."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import igraph
import pytest
from src.core.configuration import EngineConfig, RetrievalConfig, load_engine_config
from src.core.models import (
    CanonicalEntity,
    Chunk,
    CooccurrenceProvenance,
    Evidence,
    GraphProvenanceType,
    Polarity,
    ProvenanceRoute,
    QueryIntent,
    QueryPlan,
    Relationship,
    SeedEntitySpec,
    ValidationStatus,
)
from src.graph.traversal import GraphTraverser
from src.retrieval.evidence_merger import EvidenceMerger
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.retrieval.seed_retriever import SeedEntityRetriever, SeedResolutionResult
from src.storage.document_store import DocumentStore


class DirectMockSeedRetriever:
    """Mock seed retriever returning specified seed IDs directly."""

    def __init__(self, mapping: Dict[str, List[str]]) -> None:
        self.mapping = mapping

    def resolve_seeds(self, seeds: List[SeedEntitySpec]) -> SeedResolutionResult:
        resolved: List[str] = []
        for s in seeds:
            resolved.extend(self.mapping.get(s.name, []))
        return SeedResolutionResult(
            resolved_entity_ids=list(dict.fromkeys(resolved)),
            ambiguous_seeds={},
            unresolved_seeds=[],
        )

    def _resolve_single_spec(self, spec: SeedEntitySpec, entities: List[CanonicalEntity]):
        return self.mapping.get(spec.name, []), False


def _create_chunk(kb_id: str, cid: str, doc_id: str, text: str, scope_id: Optional[str] = None) -> Chunk:
    return Chunk(
        kb_id=kb_id,
        chunk_id=cid,
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        text=text,
        page_numbers=[1],
        chunk_index=0,
        char_count=len(text),
        content_hash=f"hash_{cid}",
        scope_id=scope_id,
    )


def _create_entity(kb_id: str, eid: str, name: str, chunk_ids: List[str], doc_ids: List[str]) -> CanonicalEntity:
    return CanonicalEntity(
        kb_id=kb_id,
        entity_id=eid,
        name=name,
        normalized_name=name.lower(),
        entity_type="CONCEPT",
        description=f"Description of {name}",
        source_chunk_ids=chunk_ids,
        source_doc_ids=doc_ids,
    )


# --- Tests ---

def test_default_disabled(tmp_path: Path) -> None:
    """By default, provenance expansion is disabled and only direct/traversed chunks are returned."""
    kb_id = "test_default_disabled"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    # Seed entity E1 in C1. E2 co-occurs in C1 and also appears in C2.
    c1 = _create_chunk(kb_id, "C1", "D1", "Seed entity E1 is mentioned alongside co-occurring E2.")
    c2 = _create_chunk(kb_id, "C2", "D2", "Entity E2 appears here alone.")
    ds.save_chunks([c1, c2])

    e1 = _create_entity(kb_id, "E1", "Seed Entity Alpha", ["C1"], ["D1"])
    e2 = _create_entity(kb_id, "E2", "Co-occur Beta", ["C1", "C2"], ["D1", "D2"])
    ds.save_canonical_entities([e1, e2])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E1", entity_id="E1", label=e1.name, source_chunk_ids=["C1"])
    g.add_vertex(name="E2", entity_id="E2", label=e2.name, source_chunk_ids=["C1", "C2"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity Alpha": ["E1"]}),
        graph=g,
        # Default config: provenance_expansion_enabled=False
    )
    assert retriever.provenance_expansion_enabled is False

    plan = QueryPlan(
        question="What about Seed Entity Alpha?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity Alpha")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # Only direct seed chunk C1 retrieved, C2 is NOT retrieved
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "C1"
    assert evidence[0].graph_provenance_types == [GraphProvenanceType.GRAPH_SEED_PROVENANCE]
    assert retriever.last_diagnostics["provenance_expansion_enabled"] is False
    assert retriever.last_diagnostics["expanded_chunks_count"] == 0
    assert retriever.last_diagnostics["candidate_fan_out"] == 0


def test_bipartite_expansion_recovers_cooccurring_chunk(tmp_path: Path) -> None:
    """When enabled, 1-hop Entity-Chunk-Entity bipartite expansion retrieves co-occurring chunk."""
    kb_id = "test_expansion_success"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    c1 = _create_chunk(kb_id, "C1", "D1", "Nora Ephron co-wrote Hanging Up with Delia Ephron.")
    c2 = _create_chunk(kb_id, "C2", "D2", "Delia Ephron wrote You've Got Mail.")
    ds.save_chunks([c1, c2])

    e_nora = _create_entity(kb_id, "E_NORA", "Nora Ephron", ["C1"], ["D1"])
    e_delia = _create_entity(kb_id, "E_DELIA", "Delia Ephron", ["C1", "C2"], ["D1", "D2"])
    ds.save_canonical_entities([e_nora, e_delia])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_NORA", entity_id="E_NORA", label=e_nora.name, source_chunk_ids=["C1"])
    g.add_vertex(name="E_DELIA", entity_id="E_DELIA", label=e_delia.name, source_chunk_ids=["C1", "C2"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Nora Ephron": ["E_NORA"]}),
        graph=g,
        provenance_expansion_enabled=True,
    )

    plan = QueryPlan(
        question="Nora Ephron co-wrote what film with Delia?",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Nora Ephron")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # Both C1 (direct seed) and C2 (bipartite expanded) must be retrieved
    assert len(evidence) == 2
    cids = [ev.chunk_id for ev in evidence]
    assert cids == ["C1", "C2"]

    # Verify C1 provenance
    ev_c1 = evidence[0]
    assert ev_c1.graph_rank == 1
    assert GraphProvenanceType.GRAPH_SEED_PROVENANCE in ev_c1.graph_provenance_types
    assert any("GRAPH_SEED_PROVENANCE: Seed Entity" in p for p in ev_c1.supporting_graph_paths)

    # Verify C2 provenance
    ev_c2 = evidence[1]
    assert ev_c2.graph_rank == 2
    assert GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE in ev_c2.graph_provenance_types
    assert len(ev_c2.cooccurrence_details) == 1

    detail = ev_c2.cooccurrence_details[0]
    assert detail.seed_entity_id == "E_NORA"
    assert detail.direct_chunk_id == "C1"
    assert detail.cooccurring_entity_id == "E_DELIA"
    assert detail.expanded_chunk_id == "C2"
    assert detail.expansion_depth == 1
    assert detail.fan_out_pruned is False

    assert any(
        "ENTITY_CHUNK_COOCCURRENCE: Seed [Nora Ephron (E_NORA)] -> Chunk [C1] -> Co-occur [Delia Ephron (E_DELIA)] -> Chunk [C2]"
        in p
        for p in ev_c2.supporting_graph_paths
    )

    # Verify diagnostics
    diag = retriever.last_diagnostics
    assert diag["provenance_expansion_enabled"] is True
    assert diag["cooccurring_entities_count"] == 1
    assert diag["expanded_chunks_count"] == 1
    assert diag["candidate_fan_out"] == 1
    assert diag["skipped_hub_entities"] == []
    assert diag["fan_out_pruned"] is False


def test_direct_seed_chunks_strictly_outrank_expanded(tmp_path: Path) -> None:
    """Direct seed chunks (alignment=1.0) strictly outrank expanded chunks (alignment=0.85)."""
    kb_id = "test_ranking_precedence"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    # C_SEED has chunk_id "Z_SEED" (lexicographically later than "A_EXP")
    c_seed = _create_chunk(kb_id, "Z_SEED", "D1", "Seed entity text.")
    c_exp = _create_chunk(kb_id, "A_EXP", "D2", "Expanded entity text.")
    ds.save_chunks([c_seed, c_exp])

    e1 = _create_entity(kb_id, "E1", "Seed Entity", ["Z_SEED"], ["D1"])
    e2 = _create_entity(kb_id, "E2", "Co-occur Entity", ["Z_SEED", "A_EXP"], ["D1", "D2"])
    ds.save_canonical_entities([e1, e2])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E1", entity_id="E1", label=e1.name, source_chunk_ids=["Z_SEED"])
    g.add_vertex(name="E2", entity_id="E2", label=e2.name, source_chunk_ids=["Z_SEED", "A_EXP"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E1"]}),
        graph=g,
        provenance_expansion_enabled=True,
        direct_seed_outranks_expanded=True,
    )

    plan = QueryPlan(
        question="Query for seed",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # Z_SEED must be rank 1 despite being lexicographically after A_EXP
    assert evidence[0].chunk_id == "Z_SEED"
    assert evidence[0].graph_rank == 1
    assert evidence[1].chunk_id == "A_EXP"
    assert evidence[1].graph_rank == 2


def test_traversed_graph_edge_chunks_outrank_expanded(tmp_path: Path) -> None:
    """Traversed graph-edge chunks (alignment=1.0) outrank bipartite expanded chunks (alignment=0.85)."""
    kb_id = "test_traversal_precedence"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    c_seed = _create_chunk(kb_id, "C_SEED", "D0", "Seed chunk.")
    c_trav = _create_chunk(kb_id, "C_TRAV", "D1", "Traversed edge chunk.")
    c_exp = _create_chunk(kb_id, "C_EXP", "D2", "Co-occurring expanded chunk.")
    ds.save_chunks([c_seed, c_trav, c_exp])

    e1 = _create_entity(kb_id, "E1", "Seed Entity", ["C_SEED"], ["D0"])
    e_trav = _create_entity(kb_id, "E_TRAV", "Traversed Target", ["C_TRAV"], ["D1"])
    e_cooccur = _create_entity(kb_id, "E_CO", "Co-occurring", ["C_SEED", "C_EXP"], ["D0", "D2"])
    ds.save_canonical_entities([e1, e_trav, e_cooccur])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E1", entity_id="E1", label=e1.name, source_chunk_ids=["C_SEED"])
    g.add_vertex(name="E_TRAV", entity_id="E_TRAV", label=e_trav.name, source_chunk_ids=["C_TRAV"])
    g.add_vertex(name="E_CO", entity_id="E_CO", label=e_cooccur.name, source_chunk_ids=["C_SEED", "C_EXP"])
    g.add_edge(
        "E1",
        "E_TRAV",
        edge_id="R1",
        relation_type="DIRECTED_BY",
        source_chunk_ids=["C_TRAV"],
        support_count=1,
    )

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E1"]}),
        graph=g,
        provenance_expansion_enabled=True,
    )

    plan = QueryPlan(
        question="Query with edge and co-occurrence",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
        relationship_path=["DIRECTED_BY"],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    cids = [ev.chunk_id for ev in evidence]
    # Direct seed and traversed chunks (alignment 1.0) both precede expanded chunk (alignment 0.85)
    assert cids.index("C_SEED") < cids.index("C_EXP")
    assert cids.index("C_TRAV") < cids.index("C_EXP")
    assert GraphProvenanceType.GRAPH_EDGE_PROVENANCE in evidence[cids.index("C_TRAV")].graph_provenance_types
    assert GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE in evidence[cids.index("C_EXP")].graph_provenance_types


def test_hub_entity_suppression(tmp_path: Path) -> None:
    """Entities appearing in > max_chunks_per_cooccurring_entity chunks are suppressed as hubs."""
    kb_id = "test_hub_suppression"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    # Seed in C1. E_HUB in C1, C_HUB1, C_HUB2 (3 chunks > threshold 2).
    # E_NONHUB in C1, C_VALID (2 chunks <= threshold 2).
    c1 = _create_chunk(kb_id, "C1", "D1", "Seed chunk with Hub and NonHub.")
    c_hub1 = _create_chunk(kb_id, "C_HUB1", "D2", "Hub chunk 1.")
    c_hub2 = _create_chunk(kb_id, "C_HUB2", "D3", "Hub chunk 2.")
    c_valid = _create_chunk(kb_id, "C_VALID", "D4", "Non-hub chunk.")
    ds.save_chunks([c1, c_hub1, c_hub2, c_valid])

    e_seed = _create_entity(kb_id, "E_SEED", "Seed Entity", ["C1"], ["D1"])
    e_hub = _create_entity(kb_id, "E_HUB", "Generic Hub Entity", ["C1", "C_HUB1", "C_HUB2"], ["D1", "D2", "D3"])
    e_nonhub = _create_entity(kb_id, "E_NONHUB", "Specific Entity", ["C1", "C_VALID"], ["D1", "D4"])
    ds.save_canonical_entities([e_seed, e_hub, e_nonhub])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_SEED", entity_id="E_SEED", label=e_seed.name, source_chunk_ids=["C1"])
    g.add_vertex(name="E_HUB", entity_id="E_HUB", label=e_hub.name, source_chunk_ids=["C1", "C_HUB1", "C_HUB2"])
    g.add_vertex(name="E_NONHUB", entity_id="E_NONHUB", label=e_nonhub.name, source_chunk_ids=["C1", "C_VALID"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E_SEED"]}),
        graph=g,
        provenance_expansion_enabled=True,
        max_chunks_per_cooccurring_entity=2,
    )

    plan = QueryPlan(
        question="Query seed",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    retrieved_cids = {ev.chunk_id for ev in evidence}
    assert "C1" in retrieved_cids
    assert "C_VALID" in retrieved_cids
    # Hub chunks MUST NOT be retrieved
    assert "C_HUB1" not in retrieved_cids
    assert "C_HUB2" not in retrieved_cids

    diag = retriever.last_diagnostics
    assert "E_HUB" in diag["skipped_hub_entities"]
    assert "E_NONHUB" not in diag["skipped_hub_entities"]


def test_max_cooccurring_entities_per_chunk_cap(tmp_path: Path) -> None:
    """Per-chunk fan-out cap limits how many co-occurring entities are expanded per direct chunk."""
    kb_id = "test_chunk_fanout_cap"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    c1 = _create_chunk(kb_id, "C1", "D0", "Seed chunk containing 4 entities.")
    exp_chunks = [_create_chunk(kb_id, f"C_EXP_{i}", f"D{i}", f"Chunk {i}") for i in range(1, 5)]
    ds.save_chunks([c1] + exp_chunks)

    e_seed = _create_entity(kb_id, "E_SEED", "Seed Entity", ["C1"], ["D0"])
    co_ents = [
        _create_entity(kb_id, f"E_CO_{i}", f"Entity {i}", ["C1", f"C_EXP_{i}"], ["D0", f"D{i}"])
        for i in range(1, 5)
    ]
    ds.save_canonical_entities([e_seed] + co_ents)

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_SEED", entity_id="E_SEED", label=e_seed.name, source_chunk_ids=["C1"])
    for ent in co_ents:
        g.add_vertex(name=ent.entity_id, entity_id=ent.entity_id, label=ent.name, source_chunk_ids=ent.source_chunk_ids)

    # Cap at 2 co-occurring entities per direct chunk
    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E_SEED"]}),
        graph=g,
        provenance_expansion_enabled=True,
        max_cooccurring_entities_per_chunk=2,
    )

    plan = QueryPlan(
        question="Query",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # C1 + 2 expanded chunks = 3 total evidence items
    assert len(evidence) == 3
    assert retriever.last_diagnostics["cooccurring_entities_count"] == 2
    assert retriever.last_diagnostics["expanded_chunks_count"] == 2


def test_max_expanded_candidates_per_query_cap(tmp_path: Path) -> None:
    """Query-level cap limits total expanded candidates and sets fan_out_pruned=True."""
    kb_id = "test_query_fanout_cap"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    c1 = _create_chunk(kb_id, "C1", "D0", "Seed chunk.")
    exp_chunks = [_create_chunk(kb_id, f"C_EXP_{i}", f"D{i}", f"Chunk {i}") for i in range(1, 6)]
    ds.save_chunks([c1] + exp_chunks)

    e_seed = _create_entity(kb_id, "E_SEED", "Seed Entity", ["C1"], ["D0"])
    co_ents = [
        _create_entity(kb_id, f"E_CO_{i}", f"Entity {i}", ["C1", f"C_EXP_{i}"], ["D0", f"D{i}"])
        for i in range(1, 6)
    ]
    ds.save_canonical_entities([e_seed] + co_ents)

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_SEED", entity_id="E_SEED", label=e_seed.name, source_chunk_ids=["C1"])
    for ent in co_ents:
        g.add_vertex(name=ent.entity_id, entity_id=ent.entity_id, label=ent.name, source_chunk_ids=ent.source_chunk_ids)

    # Set query cap to 2 expanded candidates
    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E_SEED"]}),
        graph=g,
        provenance_expansion_enabled=True,
        max_expanded_candidates_per_query=2,
    )

    plan = QueryPlan(
        question="Query",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # C1 + 2 expanded chunks = 3 total evidence items
    assert len(evidence) == 3
    diag = retriever.last_diagnostics
    assert diag["candidate_fan_out"] == 2
    assert diag["fan_out_pruned"] is True

    # At least one expanded evidence item records fan_out_pruned=True
    exp_evidence = [ev for ev in evidence if GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE in ev.graph_provenance_types]
    assert len(exp_evidence) == 2
    assert any(d.fan_out_pruned is True for ev in exp_evidence for d in ev.cooccurrence_details)


def test_scope_isolation(tmp_path: Path) -> None:
    """Chunks belonging to a different scope_id are filtered out during expansion."""
    kb_id = "test_scope_isolation"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    c1 = _create_chunk(kb_id, "C1", "D1", "Seed chunk in scope_A.", scope_id="scope_A")
    c_same_scope = _create_chunk(kb_id, "C_SAME", "D2", "Expanded chunk in scope_A.", scope_id="scope_A")
    c_other_scope = _create_chunk(kb_id, "C_DIFF", "D3", "Expanded chunk in scope_B.", scope_id="scope_B")
    ds.save_chunks([c1, c_same_scope, c_other_scope])

    e_seed = _create_entity(kb_id, "E_SEED", "Seed Entity", ["C1"], ["D1"])
    e_same = _create_entity(kb_id, "E_SAME", "Same Scope Entity", ["C1", "C_SAME"], ["D1", "D2"])
    e_diff = _create_entity(kb_id, "E_DIFF", "Diff Scope Entity", ["C1", "C_DIFF"], ["D1", "D3"])
    ds.save_canonical_entities([e_seed, e_same, e_diff])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E_SEED", entity_id="E_SEED", label=e_seed.name, source_chunk_ids=["C1"])
    g.add_vertex(name="E_SAME", entity_id="E_SAME", label=e_same.name, source_chunk_ids=["C1", "C_SAME"])
    g.add_vertex(name="E_DIFF", entity_id="E_DIFF", label=e_diff.name, source_chunk_ids=["C1", "C_DIFF"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E_SEED"]}),
        graph=g,
        provenance_expansion_enabled=True,
    )

    plan = QueryPlan(
        question="Query",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id, scope_id="scope_A")

    retrieved_cids = {ev.chunk_id for ev in evidence}
    assert "C1" in retrieved_cids
    assert "C_SAME" in retrieved_cids
    assert "C_DIFF" not in retrieved_cids


def test_multi_seed_fair_representation_with_expansion(tmp_path: Path) -> None:
    """Multi-seed query interleaving ensures direct seed chunks of all seeds outrank expanded chunks."""
    kb_id = "test_multi_seed_expansion"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    # Seed A has direct chunk C_A_DIRECT and expanded chunk C_A_EXP
    c_a_dir = _create_chunk(kb_id, "C_A_DIRECT", "DA1", "Seed A direct chunk.")
    c_a_exp = _create_chunk(kb_id, "C_A_EXP", "DA2", "Seed A expanded chunk.")
    # Seed B has direct chunk C_B_DIRECT and expanded chunk C_B_EXP
    c_b_dir = _create_chunk(kb_id, "C_B_DIRECT", "DB1", "Seed B direct chunk.")
    c_b_exp = _create_chunk(kb_id, "C_B_EXP", "DB2", "Seed B expanded chunk.")
    ds.save_chunks([c_a_dir, c_a_exp, c_b_dir, c_b_exp])

    ea_seed = _create_entity(kb_id, "EA_SEED", "Entity A", ["C_A_DIRECT"], ["DA1"])
    ea_co = _create_entity(kb_id, "EA_CO", "Entity A Co", ["C_A_DIRECT", "C_A_EXP"], ["DA1", "DA2"])
    eb_seed = _create_entity(kb_id, "EB_SEED", "Entity B", ["C_B_DIRECT"], ["DB1"])
    eb_co = _create_entity(kb_id, "EB_CO", "Entity B Co", ["C_B_DIRECT", "C_B_EXP"], ["DB1", "DB2"])
    ds.save_canonical_entities([ea_seed, ea_co, eb_seed, eb_co])

    g = igraph.Graph(directed=True)
    for ent in [ea_seed, ea_co, eb_seed, eb_co]:
        g.add_vertex(name=ent.entity_id, entity_id=ent.entity_id, label=ent.name, source_chunk_ids=ent.source_chunk_ids)

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Entity A": ["EA_SEED"], "Entity B": ["EB_SEED"]}),
        graph=g,
        provenance_expansion_enabled=True,
    )

    plan = QueryPlan(
        question="Compare Entity A and Entity B",
        intent=QueryIntent.LOCAL,
        seed_entities=[
            SeedEntitySpec(name="Entity A"),
            SeedEntitySpec(name="Entity B"),
        ],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    top_2_cids = {evidence[0].chunk_id, evidence[1].chunk_id}
    # Direct seed chunks of BOTH entities must be in top 2! Neither seed is starved.
    assert "C_A_DIRECT" in top_2_cids
    assert "C_B_DIRECT" in top_2_cids

    # Expanded chunks come after the direct chunks
    remaining_cids = {evidence[2].chunk_id, evidence[3].chunk_id}
    assert "C_A_EXP" in remaining_cids
    assert "C_B_EXP" in remaining_cids


def test_evidence_merger_preserves_provenance() -> None:
    """EvidenceMerger preserves graph_provenance_types and cooccurrence_details when merging streams."""
    prov_detail = CooccurrenceProvenance(
        seed_entity_id="E1",
        direct_chunk_id="C1",
        cooccurring_entity_id="E2",
        expanded_chunk_id="C2",
        expansion_depth=1,
    )
    ev_graph = Evidence(
        kb_id="test_kb",
        chunk_id="C2",
        doc_id="D2",
        text="Sample text",
        routes=[ProvenanceRoute.GRAPH],
        graph_rank=2,
        supporting_graph_paths=["ENTITY_CHUNK_COOCCURRENCE: Seed [E1] -> Chunk [C1] -> Co-occur [E2] -> Chunk [C2]"],
        graph_provenance_types=[GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE],
        cooccurrence_details=[prov_detail],
    )
    ev_vector = Evidence(
        kb_id="test_kb",
        chunk_id="C2",
        doc_id="D2",
        text="Sample text",
        routes=[ProvenanceRoute.VECTOR],
        vector_rank=1,
        vector_distance=0.15,
    )

    merged = EvidenceMerger.merge([[ev_graph], [ev_vector]])
    assert len(merged) == 1
    m = merged[0]

    assert set(m.routes) == {ProvenanceRoute.GRAPH, ProvenanceRoute.VECTOR}
    assert m.graph_rank == 2
    assert m.vector_rank == 1
    assert m.graph_provenance_types == [GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE]
    assert len(m.cooccurrence_details) == 1
    assert m.cooccurrence_details[0].seed_entity_id == "E1"


def test_backward_compatibility_config_deserialization() -> None:
    """Configurations omitting the new provenance expansion fields deserialize safely with defaults."""
    old_retrieval_data = {
        "max_traversal_hops": 3,
        "max_graph_candidates": 20,
        "vector_top_k": 10,
        "community_top_k": 3,
        "final_context_chunks": 8,
        "rrf_k": 60,
        "max_context_chars": 12000,
    }
    cfg = RetrievalConfig(**old_retrieval_data)
    assert cfg.provenance_expansion_enabled is False
    assert cfg.max_chunks_per_cooccurring_entity == 2
    assert cfg.max_cooccurring_entities_per_chunk == 5
    assert cfg.max_expanded_candidates_per_query == 10
    assert cfg.direct_seed_outranks_expanded is True

    # Loading domain configs still succeeds seamlessly
    domain_cfg = load_engine_config(Path("configs/domains/academic.yaml"))
    assert domain_cfg.retrieval.provenance_expansion_enabled is False
    assert domain_cfg.retrieval.max_chunks_per_cooccurring_entity == 2


def test_deduplication_of_multi_path_chunks(tmp_path: Path) -> None:
    """A chunk reached via multiple co-occurring paths accumulates paths without duplicating Evidence items."""
    kb_id = "test_dedup"
    ds = DocumentStore(kb_id=kb_id, storage_root=tmp_path / "kb")

    # Seed in C1. E2 and E3 both co-occur in C1, and both also appear in C_TARGET.
    c1 = _create_chunk(kb_id, "C1", "D1", "Chunk 1.")
    c_target = _create_chunk(kb_id, "C_TARGET", "D2", "Target Chunk.")
    ds.save_chunks([c1, c_target])

    e1 = _create_entity(kb_id, "E1", "Seed Entity", ["C1"], ["D1"])
    e2 = _create_entity(kb_id, "E2", "Co Entity 1", ["C1", "C_TARGET"], ["D1", "D2"])
    e3 = _create_entity(kb_id, "E3", "Co Entity 2", ["C1", "C_TARGET"], ["D1", "D2"])
    ds.save_canonical_entities([e1, e2, e3])

    g = igraph.Graph(directed=True)
    g.add_vertex(name="E1", entity_id="E1", label=e1.name, source_chunk_ids=["C1"])
    g.add_vertex(name="E2", entity_id="E2", label=e2.name, source_chunk_ids=["C1", "C_TARGET"])
    g.add_vertex(name="E3", entity_id="E3", label=e3.name, source_chunk_ids=["C1", "C_TARGET"])

    retriever = GraphEvidenceRetriever(
        doc_store=ds,
        seed_retriever=DirectMockSeedRetriever({"Seed Entity": ["E1"]}),
        graph=g,
        provenance_expansion_enabled=True,
    )

    plan = QueryPlan(
        question="Query",
        intent=QueryIntent.LOCAL,
        seed_entities=[SeedEntitySpec(name="Seed Entity")],
    )
    evidence = retriever.retrieve(plan.question, plan, kb_id)

    # Exactly 2 evidence items: C1 (direct seed) and C_TARGET (expanded via 2 paths)
    assert len(evidence) == 2
    assert evidence[0].chunk_id == "C1"
    assert evidence[1].chunk_id == "C_TARGET"
    assert len(evidence[1].supporting_graph_paths) == 2
    assert len(evidence[1].cooccurrence_details) == 2
    assert GraphProvenanceType.ENTITY_CHUNK_COOCCURRENCE in evidence[1].graph_provenance_types
