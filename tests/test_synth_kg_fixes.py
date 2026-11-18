"""Tests for the synthetic-KG fixes.

Fix 1: person-like entity types come from the ontology (no literal type names), initial + surname matching.
Fix 2: exact-name merge across types with plurality type vote, and ontology re-validation after resolution.
Fix 3: relationship-extraction prompt file selected by prompt_versions.relationship_extraction_version.
Fix 4: graph_context_mode "off" omits the graph path block (provenance marker text) from the LLM context.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import yaml

from src.core.configuration import EngineConfig, PromptVersions, ResolutionConfig, load_engine_config
from src.core.models import (
    CanonicalEntity,
    Chunk,
    EntityMention,
    Evidence,
    Polarity,
    ProvenanceRoute,
    QueryPlan,
    Relationship,
    RetrievalResult,
    SeedEntitySpec,
    ValidationStatus,
)
from src.core.ontology import DomainOntology, RelationConstraint
from src.engine import HybridGraphRAGEngine
from src.extraction.relation_extractor import OllamaRelationshipExtractor
from src.extraction.schemas import RawRelationshipExtractionOutput
from src.resolution.entity_resolver import (
    ConservativeEntityResolver,
    is_initial_surname_form,
    person_names_compatible,
)
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

KB = "fix_kb"
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- helpers
def _ontology(person_type: str = "EMPLOYEE", declare_person_like: bool = True) -> DomainOntology:
    """Company-shaped ontology whose person type name is configurable."""
    kwargs = {}
    if declare_person_like:
        kwargs["person_like_types"] = [person_type]
    return DomainOntology(
        domain_name="test_company",
        entity_types={
            person_type: "People",
            "DEPARTMENT": "Units",
            "PRODUCT": "Products",
            "TECHNOLOGY": "Tech",
            "CLIENT": "Clients",
        },
        relationship_types={
            "MANAGES": RelationConstraint(
                name="MANAGES", allowed_source_types=[person_type],
                allowed_target_types=[person_type, "DEPARTMENT", "PRODUCT"],
            ),
            "BELONGS_TO": RelationConstraint(
                name="BELONGS_TO", allowed_source_types=[person_type, "PRODUCT"], allowed_target_types=["DEPARTMENT"],
            ),
            "DEVELOPS": RelationConstraint(
                name="DEVELOPS", allowed_source_types=[person_type, "DEPARTMENT"], allowed_target_types=["PRODUCT"],
            ),
            "BUILT_WITH": RelationConstraint(
                name="BUILT_WITH", allowed_source_types=["PRODUCT"], allowed_target_types=["TECHNOLOGY"],
            ),
            "SERVES": RelationConstraint(
                name="SERVES", allowed_source_types=["PRODUCT", "DEPARTMENT"], allowed_target_types=["CLIENT"],
            ),
        },
        **kwargs,
    )


def _mention(mid: str, name: str, etype: str, chunk: str = "C1", doc: str = "D1") -> EntityMention:
    return EntityMention(
        kb_id=KB, mention_id=mid, surface_name=name, normalized_name=name.lower(), entity_type=etype,
        description="", source_chunk_id=chunk, source_doc_id=doc,
    )


def _triple(rid: str, src: str, rel: str, tgt: str, status: ValidationStatus = ValidationStatus.VALIDATED,
            chunk: str = "C1") -> Relationship:
    return Relationship(
        kb_id=KB, relationship_id=rid, source_entity_id=src, target_entity_id=tgt, relation_type=rel,
        source_chunk_ids=[chunk], source_doc_ids=["D1"], evidence_quotes=["q"], support_count=1,
        polarity=Polarity.POSITIVE, validation_status=status,
    )


def _resolver(ontology: DomainOntology, **flags) -> ConservativeEntityResolver:
    return ConservativeEntityResolver(resolution_config=ResolutionConfig(**flags), ontology=ontology)


def _entity(eid: str, name: str, etype: str) -> CanonicalEntity:
    return CanonicalEntity(
        kb_id=KB, entity_id=eid, name=name, normalized_name=name.lower(), entity_type=etype,
        description=f"Description of {name}", source_chunk_ids=[f"C_{eid}"], source_doc_ids=[f"D_{eid}"],
    )


class _Embedder:
    def embed_text(self, text: str) -> List[float]:
        return [1.0 / (768 ** 0.5)] * 768

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def _seed_retriever(tmp_path: Path, entities: List[CanonicalEntity], person_like: Optional[List[str]]) -> SeedEntityRetriever:
    ds = DocumentStore(kb_id=KB, processed_dir=tmp_path / "processed")
    vs = ChromaVectorStore(kb_id=KB, kb_vector_store_dir=tmp_path / "vstore")
    ds.save_canonical_entities(entities)
    kwargs = {} if person_like is None else {"person_like_types": person_like}
    return SeedEntityRetriever(ds, vs, _Embedder(), **kwargs)


# =========================================================================== Fix 1: ontology field
def test_person_like_types_default_and_validation() -> None:
    default_onto = _ontology(person_type="PERSON", declare_person_like=False)
    assert default_onto.person_like_types == ["PERSON"]
    assert default_onto.is_person_like_type("person") is True

    # Default list is exempt from the existence check (ontology without any person type still loads).
    no_person = _ontology(person_type="EMPLOYEE", declare_person_like=False)
    assert no_person.person_like_types == ["PERSON"]
    assert no_person.is_person_like_type("EMPLOYEE") is False

    custom = _ontology(person_type="EMPLOYEE")
    assert custom.is_person_like_type("employee") is True
    assert custom.is_person_like_type("PRODUCT") is False
    assert custom.is_person_like_type(None) is False

    with pytest.raises(ValueError, match="person_like_types"):
        DomainOntology(
            domain_name="bad",
            entity_types={"EMPLOYEE": "People"},
            relationship_types={"MANAGES": RelationConstraint(name="MANAGES")},
            person_like_types=["STAFF"],
        )


def test_no_quoted_person_type_literal_in_resolution_retrieval_extraction() -> None:
    """Source scan: the engine must not hardcode a person entity-type literal in these packages."""
    pattern = re.compile(r"""["']person["']""", re.IGNORECASE)
    offenders: List[str] = []
    for pkg in ("resolution", "retrieval", "extraction"):
        for path in sorted((REPO_ROOT / "src" / pkg).rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []


# =========================================================================== Fix 1: name compatibility
def test_person_names_compatible_initial_plus_surname() -> None:
    assert person_names_compatible("G. Nebikvane", "Gryxethvath Nebikvane") is True
    assert person_names_compatible("Gryxethvath Nebikvane", "G Nebikvane") is True
    assert person_names_compatible("G. Nebikvane", "Gryxethvath Kren") is False       # different surname
    assert person_names_compatible("L. Nebikvane", "Gryxethvath Nebikvane") is False   # different initial
    assert person_names_compatible("G. Nebikvane", "Gryxethvath Orin Nebikvane") is True  # middle name on full form
    assert is_initial_surname_form("G. Nebikvane") is True
    assert is_initial_surname_form("Gryxethvath Nebikvane") is False
    assert is_initial_surname_form("Nebikvane") is False


def test_person_names_compatible_previous_behaviour_unchanged() -> None:
    assert person_names_compatible("Ed Wood", "Edward Davis Wood Jr.") is True
    assert person_names_compatible("Bob Smith", "Robert J. Smith") is True
    assert person_names_compatible("John Smith", "Jane Smith") is False
    assert person_names_compatible("Edward Wood", "Edward Wood Sr.") is False
    assert person_names_compatible("Smith", "John Smith") is False


# =========================================================================== Fix 1: seed retrieval
def test_seed_employee_initial_surname_resolves(tmp_path: Path) -> None:
    entities = [
        _entity("E001", "Gryxethvath Nebikvane", "EMPLOYEE"),
        _entity("E002", "Gryxethvath Kren", "EMPLOYEE"),      # same given name, different surname
        _entity("E003", "Lorosmir Nebikvane", "EMPLOYEE"),    # same surname, different initial
        _entity("E004", "Nebikvane Gateway", "PRODUCT"),      # non-person entity sharing a token
    ]
    retriever = _seed_retriever(tmp_path, entities, ["EMPLOYEE"])

    typed = retriever.resolve_seeds([SeedEntitySpec(name="G. Nebikvane", expected_type="EMPLOYEE")])
    assert typed.resolved_entity_ids == ["E001"]
    assert typed.ambiguous_seeds == {}

    # No planner type: the person-name branch must still run for person-like entities.
    untyped = retriever.resolve_seeds([SeedEntitySpec(name="G. Nebikvane", expected_type=None)])
    assert untyped.resolved_entity_ids == ["E001"]
    assert untyped.resolution_diagnostics[0].matching_method == "PERSON_NAME_COMPATIBLE"

    # Planner assigned a wrong type: still resolved (type-conflict tier), not lost.
    wrong = retriever.resolve_seeds([SeedEntitySpec(name="G. Nebikvane", expected_type="DEPARTMENT")])
    assert wrong.resolved_entity_ids == ["E001"]


def test_seed_employee_initial_surname_ambiguous_when_two_match(tmp_path: Path) -> None:
    entities = [
        _entity("E002", "Galthor Nebikvane", "EMPLOYEE"),
        _entity("E001", "Gryxethvath Nebikvane", "EMPLOYEE"),
    ]
    retriever = _seed_retriever(tmp_path, entities, ["EMPLOYEE"])
    result = retriever.resolve_seeds([SeedEntitySpec(name="G. Nebikvane", expected_type="EMPLOYEE")])
    # Existing policy: candidates sorted by entity_id, capped at max_candidates_per_seed, flagged ambiguous.
    assert result.resolved_entity_ids == ["E001", "E002"]
    assert result.ambiguous_seeds == {"G. Nebikvane": ["E001", "E002"]}
    assert all(d.status == "AMBIGUOUS" for d in result.resolution_diagnostics)


def test_seed_employee_not_person_like_without_ontology_setting(tmp_path: Path) -> None:
    """With the default person-like set, EMPLOYEE entities get no person-name matching (old behaviour)."""
    entities = [_entity("E001", "Gryxethvath Nebikvane", "EMPLOYEE")]
    retriever = _seed_retriever(tmp_path, entities, None)
    result = retriever.resolve_seeds([SeedEntitySpec(name="G. Nebikvane", expected_type="EMPLOYEE")])
    assert result.resolved_entity_ids == []
    assert result.unresolved_seeds == ["G. Nebikvane"]


@pytest.mark.parametrize("spec_name,spec_type", [
    ("Ed Wood", "PERSON"), ("Edward Wood", None), ("Wood Pictures", "ORG"), ("Edward Davis Wood Jr.", "PERSON"),
    ("Jane Wood", "PERSON"),
])
def test_seed_person_ontology_matches_explicit_person_setting(tmp_path: Path, spec_name: str, spec_type: Optional[str]) -> None:
    """A PERSON ontology behaves the same whether person_like_types is left at its default or passed explicitly,
    and reproduces the pre-change expectations for name variations."""
    entities = [
        _entity("E001", "Edward Davis Wood Jr.", "PERSON"),
        _entity("E002", "Wood Pictures", "ORG"),
        _entity("E003", "John Wood", "PERSON"),
    ]
    default_ret = _seed_retriever(tmp_path / "a", entities, None)
    explicit_ret = _seed_retriever(tmp_path / "b", entities, ["PERSON"])
    spec = SeedEntitySpec(name=spec_name, expected_type=spec_type)
    r1, r2 = default_ret.resolve_seeds([spec]), explicit_ret.resolve_seeds([spec])
    assert r1.resolved_entity_ids == r2.resolved_entity_ids
    assert [(d.candidate_id, d.matching_method, d.status) for d in r1.resolution_diagnostics] == \
           [(d.candidate_id, d.matching_method, d.status) for d in r2.resolution_diagnostics]
    expected = {
        "Ed Wood": ["E001"], "Edward Wood": ["E001"], "Wood Pictures": ["E002"],
        "Edward Davis Wood Jr.": ["E001"], "Jane Wood": [],
    }[spec_name]
    assert r1.resolved_entity_ids == expected


# =========================================================================== Fix 1: resolution
def _person_scenario(person_type: str) -> List[EntityMention]:
    return [
        _mention("M1", "Ananya Sharma", person_type, "C1"),
        _mention("M2", "Sharma", person_type, "C2"),          # Rule C2: single-token surname
        _mention("M3", "Dr. Ananya Sharma", person_type, "C3"),  # honorific stripping
        _mention("M4", "Rahul Verma Jr.", person_type, "C4"),
        _mention("M5", "Rahul Verma", person_type, "C5"),     # generational suffix must keep these apart
        _mention("M6", "Platform Unit", "DEPARTMENT", "C6"),
    ]


def _partition(resolver: ConservativeEntityResolver, mentions: List[EntityMention]) -> List[Tuple[str, ...]]:
    _, mapping = resolver.resolve_mentions(mentions)
    groups: Dict[str, List[str]] = {}
    for mid, eid in mapping.items():
        groups.setdefault(eid, []).append(mid)
    return sorted(tuple(sorted(v)) for v in groups.values())


def test_resolution_person_rules_follow_ontology_person_like_types() -> None:
    expected = [("M1", "M2", "M3"), ("M4",), ("M5",), ("M6",)]
    # PERSON ontology (default person_like_types): identical to the behaviour before the change.
    assert _partition(_resolver(_ontology("PERSON", declare_person_like=False)), _person_scenario("PERSON")) == expected
    # EMPLOYEE ontology that declares EMPLOYEE person-like: the same rules now apply.
    assert _partition(_resolver(_ontology("EMPLOYEE")), _person_scenario("EMPLOYEE")) == expected
    # EMPLOYEE ontology without the declaration: person rules do not run (surname and honorific forms stay apart).
    undeclared = _partition(_resolver(_ontology("EMPLOYEE", declare_person_like=False)), _person_scenario("EMPLOYEE"))
    assert ("M1", "M2", "M3") not in undeclared
    assert ("M2",) in undeclared


def test_resolution_initial_surname_merges_only_with_unique_candidate() -> None:
    onto = _ontology("EMPLOYEE")
    unique = [
        _mention("M1", "Gryxethvath Nebikvane", "EMPLOYEE", "C1"),
        _mention("M2", "G. Nebikvane", "EMPLOYEE", "C2"),
        _mention("M3", "L. Nebikvane", "EMPLOYEE", "C3"),     # different initial: stays separate
        _mention("M4", "G. Kren", "EMPLOYEE", "C4"),          # different surname: stays separate
    ]
    on = _resolver(onto, enable_initial_surname_person_merge=True)
    assert _partition(on, unique) == [("M1", "M2"), ("M3",), ("M4",)]
    assert on.last_resolution_report["initial_surname_merges"][0]["into"] == "Gryxethvath Nebikvane"

    # Initial form seen before the full name: handled by the post-pass.
    assert _partition(_resolver(onto, enable_initial_surname_person_merge=True), [unique[1], unique[0]]) == [("M1", "M2")]

    # Two candidates share initial and surname: no merge.
    ambiguous = [
        _mention("M1", "Gryxethvath Nebikvane", "EMPLOYEE", "C1"),
        _mention("M5", "Galthor Nebikvane", "EMPLOYEE", "C5"),
        _mention("M2", "G. Nebikvane", "EMPLOYEE", "C2"),
    ]
    assert _partition(_resolver(onto, enable_initial_surname_person_merge=True), ambiguous) == [("M1",), ("M2",), ("M5",)]

    # Flag off (default): initial forms are never merged.
    assert _partition(_resolver(onto), unique) == [("M1",), ("M2",), ("M3",), ("M4",)]


# =========================================================================== Fix 1: relation extractor
class _FixedLLM:
    def __init__(self, payload: Dict) -> None:
        self.payload = payload
        self.prompts: List[str] = []

    def generate_json(self, prompt: str, schema_cls, max_retries: Optional[int] = None):
        self.prompts.append(prompt)
        return schema_cls.model_validate(self.payload)


def _surname_extraction(person_type: str, declare: bool) -> List[Relationship]:
    text = "Ananya Sharma leads the Platform Unit. Sharma manages the Platform Unit day to day."
    chunk = Chunk(kb_id=KB, chunk_id="D1_C0", doc_id="D1", filename="d.txt", page_numbers=[1], chunk_index=0,
                  text=text, char_count=len(text), content_hash="h")
    mentions = [
        _mention("D1_C0_M001", "Ananya Sharma", person_type, "D1_C0"),
        _mention("D1_C0_M002", "Platform Unit", "DEPARTMENT", "D1_C0"),
    ]
    payload = {"relationships": [{
        "source_mention_id": "Sharma",  # surname only: resolvable only through the person-like surname rule
        "target_mention_id": "D1_C0_M002",
        "relation_type": "MANAGES",
        "evidence_quote": "Sharma manages the Platform Unit day to day.",
        "polarity": "POSITIVE",
        "confidence": 1.0,
    }]}
    extractor = OllamaRelationshipExtractor(
        ontology=_ontology(person_type, declare_person_like=declare), llm_client=_FixedLLM(payload), prompt_template="{chunk_text}",
    )
    return extractor.extract_relationships(chunk, mentions)


def test_relation_extractor_surname_rule_uses_person_like_types() -> None:
    person = _surname_extraction("PERSON", declare=False)      # PERSON ontology: unchanged behaviour
    assert [(r.source_entity_id, r.validation_status) for r in person] == [("D1_C0_M001", ValidationStatus.VALIDATED)]
    employee = _surname_extraction("EMPLOYEE", declare=True)   # declared person-like: same behaviour
    assert [(r.source_entity_id, r.validation_status) for r in employee] == [("D1_C0_M001", ValidationStatus.VALIDATED)]
    undeclared = _surname_extraction("EMPLOYEE", declare=False)  # not person-like: surname is not resolved
    assert undeclared == []


# =========================================================================== Fix 2: merge + type vote
def _typed_name_scenario() -> Tuple[List[EntityMention], List[Relationship]]:
    mentions = [
        _mention("A1", "Taelan Wyrus", "EMPLOYEE", "C1"),
        _mention("A2", "Taelan Wyrus", "EMPLOYEE", "C2"),
        _mention("A3", "Taelan Wyrus", "PRODUCT", "C3"),   # mistyped once
        _mention("A4", "Taelan Wyrus", "EMPLOYEE", "C4"),
        _mention("B1", "Fennik Quen", "EMPLOYEE", "C3"),
        _mention("B2", "Fennik Quen", "EMPLOYEE", "C5"),
        _mention("D1", "Core Unit", "DEPARTMENT", "C5"),
        _mention("K1", "Acme Corp", "CLIENT", "C6"),
        _mention("B3", "Fennik Quen", "EMPLOYEE", "C6"),
        _mention("A5", "Taelan Wyrus", "CLIENT", "C6"),    # mistyped as CLIENT in this chunk
    ]
    triples = [
        # Valid at mention level only because A3 was typed PRODUCT; employee -DEVELOPS-> employee is not allowed.
        _triple("R1", "B1", "DEVELOPS", "A3", ValidationStatus.VALIDATED, "C3"),
        # Rejected at mention level only because A5 was typed CLIENT; employee -MANAGES-> employee is allowed.
        _triple("R2", "B3", "MANAGES", "A5", ValidationStatus.REJECTED_ONTOLOGY, "C6"),
        _triple("R3", "B2", "BELONGS_TO", "D1", ValidationStatus.VALIDATED, "C5"),
        # Evidence-level rejections must never be revived, whatever the canonical types say.
        _triple("R4", "B2", "MANAGES", "D1", ValidationStatus.REJECTED_QUOTE_MISMATCH, "C5"),
        _triple("R5", "B2", "MANAGES", "D1", ValidationStatus.REJECTED_NEGATION, "C5"),
        _triple("R6", "B2", "MANAGES", "D1", ValidationStatus.REJECTED_UNCERTAIN, "C5"),
        _triple("R7", "B2", "MANAGES", "D1", ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE, "C5"),
    ]
    return mentions, triples


def test_merge_exact_name_across_types_plurality_vote() -> None:
    mentions, triples = _typed_name_scenario()
    resolver = _resolver(_ontology(), merge_exact_name_across_types=True)
    entities, mapping, _ = resolver.resolve_and_remap(mentions, triples)
    by_name = {e.name: e for e in entities}
    assert len(entities) == 4
    taelan = by_name["Taelan Wyrus"]
    assert taelan.entity_type == "EMPLOYEE"
    assert taelan.type_vote_counts == {"CLIENT": 1, "EMPLOYEE": 3, "PRODUCT": 1}
    assert {mapping[m] for m in ("A1", "A2", "A3", "A4", "A5")} == {taelan.entity_id}
    report = {r["name"]: r for r in resolver.last_resolution_report["type_reconciliation"]}
    assert report["Taelan Wyrus"]["tie_break"] == "plurality"


def test_revalidation_after_resolution_changes_only_ontology_statuses() -> None:
    mentions, triples = _typed_name_scenario()
    resolver = _resolver(
        _ontology(), merge_exact_name_across_types=True, revalidate_relationships_after_resolution=True
    )
    entities, mapping, graph_rels = resolver.resolve_and_remap(mentions, triples)
    out = {r.relationship_id: r for r in resolver.last_revalidated_triples}

    # employee -DEVELOPS-> (entity now EMPLOYEE): rejected, original status kept.
    assert out["R1"].validation_status == ValidationStatus.REJECTED_ONTOLOGY
    assert out["R1"].mention_level_status == ValidationStatus.VALIDATED
    # mention-level type error only: becomes valid with canonical types.
    assert out["R2"].validation_status == ValidationStatus.VALIDATED
    assert out["R2"].mention_level_status == ValidationStatus.REJECTED_ONTOLOGY
    # unchanged but recomputed
    assert out["R3"].validation_status == ValidationStatus.VALIDATED
    assert out["R3"].mention_level_status == ValidationStatus.VALIDATED
    # evidence-level rejections untouched and never revived
    for rid, status in [("R4", ValidationStatus.REJECTED_QUOTE_MISMATCH), ("R5", ValidationStatus.REJECTED_NEGATION),
                        ("R6", ValidationStatus.REJECTED_UNCERTAIN), ("R7", ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE)]:
        assert out[rid].validation_status == status
        assert out[rid].mention_level_status is None

    graph = {(r.source_entity_id, r.relation_type, r.target_entity_id): r for r in graph_rels}
    fennik, taelan, core = mapping["B1"], mapping["A1"], mapping["D1"]
    assert set(graph) == {(fennik, "MANAGES", taelan), (fennik, "BELONGS_TO", core)}
    assert graph[(fennik, "MANAGES", taelan)].mention_level_status == ValidationStatus.REJECTED_ONTOLOGY
    summary = resolver.last_resolution_report["revalidation"]
    assert summary["validated_to_rejected_ontology"] == 1
    assert summary["rejected_ontology_to_validated"] == 1
    # inputs are not mutated
    assert all(t.mention_level_status is None for t in triples)
    assert triples[0].validation_status == ValidationStatus.VALIDATED


def test_fix2_flags_off_output_identical_to_current_resolver() -> None:
    mentions, triples = _typed_name_scenario()
    resolver = _resolver(_ontology())  # defaults: both flags False
    entities, mapping, graph_rels = resolver.resolve_and_remap(mentions, triples)
    # require_type_match keeps differently typed mentions of one name apart, exactly as before.
    assert sorted((e.name, e.entity_type) for e in entities) == [
        ("Acme Corp", "CLIENT"), ("Core Unit", "DEPARTMENT"), ("Fennik Quen", "EMPLOYEE"),
        ("Taelan Wyrus", "CLIENT"), ("Taelan Wyrus", "EMPLOYEE"), ("Taelan Wyrus", "PRODUCT"),
    ]
    assert all(e.type_vote_counts == {} for e in entities)
    assert resolver.last_revalidated_triples is None
    assert resolver.last_resolution_report == {}
    # Only mention-level VALIDATED triples reach the graph, with their original relation types.
    assert sorted((r.relation_type, r.mention_level_status) for r in graph_rels) == [("BELONGS_TO", None), ("DEVELOPS", None)]
    assert mapping["A3"] != mapping["A1"]


def test_type_vote_tie_broken_by_relationship_validity() -> None:
    mentions = [
        _mention("X1", "Alpha", "EMPLOYEE", "C1"),
        _mention("X2", "Alpha", "PRODUCT", "C2"),
        _mention("T1", "TechX", "TECHNOLOGY", "C2"),
    ]
    triples = [_triple("R1", "X2", "BUILT_WITH", "T1", ValidationStatus.VALIDATED, "C2")]
    resolver = _resolver(_ontology(), merge_exact_name_across_types=True)
    entities, _, _ = resolver.resolve_and_remap(mentions, triples)
    alpha = next(e for e in entities if e.name == "Alpha")
    # EMPLOYEE comes first in declaration order, but only PRODUCT keeps the BUILT_WITH edge valid.
    assert alpha.entity_type == "PRODUCT"
    rep = resolver.last_resolution_report["type_reconciliation"][0]
    assert rep["tie_break"] == "relationship_validity"
    assert rep["valid_relationship_counts"] == {"EMPLOYEE": 0, "PRODUCT": 1}


def test_type_vote_tie_broken_by_ontology_declaration_order() -> None:
    mentions = [_mention("X1", "Beta", "PRODUCT", "C1"), _mention("X2", "Beta", "DEPARTMENT", "C2")]
    resolver = _resolver(_ontology(), merge_exact_name_across_types=True)
    entities, _, _ = resolver.resolve_and_remap(mentions, [])
    assert entities[0].entity_type == "DEPARTMENT"  # declared before PRODUCT
    assert resolver.last_resolution_report["type_reconciliation"][0]["tie_break"] == "ontology_declaration_order"


def test_type_vote_tied_neighbour_treated_as_first_tied_type() -> None:
    mentions = [
        _mention("A1", "Alpha", "EMPLOYEE", "C1"), _mention("A2", "Alpha", "PRODUCT", "C2"),
        _mention("B1", "Beta", "CLIENT", "C1"), _mention("B2", "Beta", "DEPARTMENT", "C2"),
    ]
    # Alpha -SERVES-> Beta is valid only for (PRODUCT, CLIENT). Beta is tied and is treated as DEPARTMENT
    # (its first tied type in declaration order) while Alpha is resolved, so no type of Alpha keeps the edge
    # valid and declaration order decides; the same holds for Beta with Alpha treated as EMPLOYEE.
    triples = [_triple("R1", "A2", "SERVES", "B1", ValidationStatus.VALIDATED, "C1")]
    resolver = _resolver(_ontology(), merge_exact_name_across_types=True)
    entities, _, _ = resolver.resolve_and_remap(mentions, triples)
    types = {e.name: e.entity_type for e in entities}
    assert types == {"Alpha": "EMPLOYEE", "Beta": "DEPARTMENT"}
    assert {r["name"]: r["tie_break"] for r in resolver.last_resolution_report["type_reconciliation"]} == {
        "Alpha": "ontology_declaration_order", "Beta": "ontology_declaration_order",
    }


def test_type_reconciliation_and_revalidation_deterministic_under_shuffle() -> None:
    mentions, triples = _typed_name_scenario()
    mentions = mentions + [
        _mention("X1", "Alpha", "EMPLOYEE", "C7"), _mention("X2", "Alpha", "PRODUCT", "C8"),
        _mention("T1", "TechX", "TECHNOLOGY", "C8"),
        _mention("Y1", "Beta", "PRODUCT", "C9"), _mention("Y2", "Beta", "DEPARTMENT", "C9"),
    ]
    triples = triples + [_triple("R8", "X2", "BUILT_WITH", "T1", ValidationStatus.VALIDATED, "C8")]

    def run(ms: List[EntityMention], ts: List[Relationship]):
        resolver = _resolver(
            _ontology(), merge_exact_name_across_types=True, revalidate_relationships_after_resolution=True
        )
        entities, mapping, graph_rels = resolver.resolve_and_remap(ms, ts)
        name_of = {e.entity_id: e.normalized_name for e in entities}
        ent_view = sorted((e.normalized_name, e.entity_type, tuple(sorted(e.type_vote_counts.items()))) for e in entities)
        status_view = sorted((r.relationship_id, r.validation_status.value) for r in resolver.last_revalidated_triples)
        graph_view = sorted((name_of[r.source_entity_id], r.relation_type, name_of[r.target_entity_id]) for r in graph_rels)
        return ent_view, status_view, graph_view

    reference = run(mentions, triples)
    rng = random.Random(0)
    for _ in range(20):
        ms, ts = list(mentions), list(triples)
        rng.shuffle(ms)
        rng.shuffle(ts)
        assert run(ms, ts) == reference


# =========================================================================== config files
def test_default_config_keeps_fix_flags_off_and_company_enables_them() -> None:
    default_cfg = load_engine_config(REPO_ROOT / "configs/domains/academic.yaml", REPO_ROOT / "configs/default.yaml")
    raw_default = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    assert raw_default["resolution"]["merge_exact_name_across_types"] is False
    assert raw_default["resolution"]["revalidate_relationships_after_resolution"] is False
    assert raw_default["resolution"]["enable_initial_surname_person_merge"] is False
    assert default_cfg.resolution.merge_exact_name_across_types is False
    assert default_cfg.resolution.revalidate_relationships_after_resolution is False
    assert default_cfg.resolution.enable_initial_surname_person_merge is False
    assert default_cfg.prompt_versions.relationship_extraction_version == "v1.0"
    assert ResolutionConfig().merge_exact_name_across_types is False
    assert ResolutionConfig().revalidate_relationships_after_resolution is False

    company = load_engine_config(REPO_ROOT / "configs/domains/company.yaml", REPO_ROOT / "configs/default.yaml")
    assert company.resolution.merge_exact_name_across_types is True
    assert company.resolution.revalidate_relationships_after_resolution is True
    assert company.ontology.person_like_types == ["EMPLOYEE"]
    assert company.ontology.is_person_like_type("EMPLOYEE") is True


def test_engine_config_has_no_duplicate_top_level_revalidation_flag() -> None:
    assert "revalidate_relationships_after_resolution" not in EngineConfig.model_fields
    assert "revalidate_relationships_after_resolution" in ResolutionConfig.model_fields


# =========================================================================== Fix 3: prompt selection
def test_relationship_prompt_version_validation() -> None:
    assert PromptVersions().relationship_extraction_version == "v1.0"
    assert PromptVersions(relationship_extraction_version="V1.1").relationship_extraction_version == "v1.1"
    with pytest.raises(ValueError):
        PromptVersions(relationship_extraction_version="v9.9")


def test_relationship_prompt_files_differ_only_by_multi_relation_instruction() -> None:
    v10 = (REPO_ROOT / "prompts/relationship_extraction.txt").read_text(encoding="utf-8")
    v11 = (REPO_ROOT / "prompts/relationship_extraction_v1_1.txt").read_text(encoding="utf-8")
    added = [l for l in v11.splitlines() if l not in v10.splitlines()]
    removed = [l for l in v10.splitlines() if l not in v11.splitlines()]
    assert removed == []
    assert len(added) == 1
    assert "several different relations between the same two entities, output one object per relation" in added[0]


class _StubEngine:
    """Minimal stand-in exposing what HybridGraphRAGEngine._relationship_prompt_kwargs reads."""
    RELATIONSHIP_PROMPT_FILES = HybridGraphRAGEngine.RELATIONSHIP_PROMPT_FILES

    def __init__(self, version: str, prompts_dir: str) -> None:
        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.prompt_versions = PromptVersions(relationship_extraction_version=version)
        self.config.prompts_dir = prompts_dir


def test_engine_selects_relationship_prompt_file_by_version(tmp_path: Path) -> None:
    prompts_dir = str(REPO_ROOT / "prompts")
    # Default version: no override, the extractor keeps its historical default path.
    assert HybridGraphRAGEngine._relationship_prompt_kwargs(_StubEngine("v1.0", prompts_dir)) == {}
    kwargs = HybridGraphRAGEngine._relationship_prompt_kwargs(_StubEngine("v1.1", prompts_dir))
    assert Path(kwargs["prompt_template_path"]).name == "relationship_extraction_v1_1.txt"
    extractor = OllamaRelationshipExtractor(ontology=_ontology(), llm_client=_FixedLLM({"relationships": []}), **kwargs)
    assert "output one object per relation" in extractor.prompt_template
    default_extractor = OllamaRelationshipExtractor(ontology=_ontology(), llm_client=_FixedLLM({"relationships": []}))
    if Path("prompts/relationship_extraction.txt").exists():  # default path is cwd-relative
        assert "output one object per relation" not in default_extractor.prompt_template
    # A selected version whose file is missing is an error, never a silent fallback.
    with pytest.raises(FileNotFoundError):
        HybridGraphRAGEngine._relationship_prompt_kwargs(_StubEngine("v1.1", str(tmp_path)))


def test_parser_keeps_two_relation_types_for_one_mention_pair() -> None:
    text = "Pryx Iskor formally belongs to Zanis Group. Pryx Iskor serves as the primary manager of Zanis Group."
    chunk = Chunk(kb_id=KB, chunk_id="D1_C0", doc_id="D1", filename="d.txt", page_numbers=[1], chunk_index=0,
                  text=text, char_count=len(text), content_hash="h")
    mentions = [
        _mention("D1_C0_M001", "Pryx Iskor", "EMPLOYEE", "D1_C0"),
        _mention("D1_C0_M002", "Zanis Group", "DEPARTMENT", "D1_C0"),
    ]
    payload = {"relationships": [
        {"source_mention_id": "D1_C0_M001", "target_mention_id": "D1_C0_M002", "relation_type": "BELONGS_TO",
         "evidence_quote": "Pryx Iskor formally belongs to Zanis Group.", "polarity": "POSITIVE", "confidence": 1.0},
        {"source_mention_id": "D1_C0_M001", "target_mention_id": "D1_C0_M002", "relation_type": "MANAGES",
         "evidence_quote": "Pryx Iskor serves as the primary manager of Zanis Group.", "polarity": "POSITIVE", "confidence": 1.0},
    ]}
    assert len(RawRelationshipExtractionOutput.model_validate(payload).relationships) == 2
    extractor = OllamaRelationshipExtractor(ontology=_ontology(), llm_client=_FixedLLM(payload), prompt_template="{chunk_text}")
    triples = extractor.extract_relationships(chunk, mentions)
    assert sorted((t.relation_type, t.validation_status) for t in triples) == [
        ("BELONGS_TO", ValidationStatus.VALIDATED), ("MANAGES", ValidationStatus.VALIDATED),
    ]
    # Resolution and aggregation keep both relation types between the same canonical pair.
    _, _, graph_rels = _resolver(_ontology()).resolve_and_remap(mentions, triples)
    assert sorted(r.relation_type for r in graph_rels) == ["BELONGS_TO", "MANAGES"]


# =========================================================================== Fix 4: graph context "off"
def _retrieval_result_with_graph_path() -> RetrievalResult:
    ev = Evidence(
        kb_id=KB, chunk_id="D1_C0", doc_id="D1", filename="d.txt", page_numbers=[1],
        text="Pryx Iskor manages Zanis Group.", routes=[ProvenanceRoute.GRAPH],
        supporting_graph_paths=["GRAPH_EDGE_PROVENANCE: Pryx Iskor -[MANAGES]-> Zanis Group"],
    )
    return RetrievalResult(kb_id=KB, question="q", query_plan=QueryPlan(question="q"), evidence_list=[ev])


def test_graph_context_mode_off_string_omits_provenance_marker() -> None:
    result = _retrieval_result_with_graph_path()
    off_ctx = ContextBuilder(graph_context_mode="off").build_context(result)
    assert "GRAPH_EDGE_PROVENANCE" not in off_ctx
    assert "### Supporting Graph Traversal Paths" not in off_ctx
    assert "Pryx Iskor manages Zanis Group." in off_ctx
    assert "GRAPH_EDGE_PROVENANCE" not in ContextBuilder(graph_context_mode="OFF").build_context(result)
    # Boolean False keeps the legacy block (backward compatible default).
    legacy_ctx = ContextBuilder(graph_context_mode=False).build_context(result)
    assert "### Supporting Graph Traversal Paths" in legacy_ctx
    assert "GRAPH_EDGE_PROVENANCE" in legacy_ctx
