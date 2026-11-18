"""Conservative entity resolution and relationship remapping pipeline."""

from __future__ import annotations
import json
import logging
from pathlib import Path
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from src.core.configuration import EngineConfig, ResolutionConfig
from src.core.interfaces import EmbeddingProvider
from src.core.models import CanonicalEntity, EntityMention, Polarity, Relationship, ValidationStatus
from src.core.ontology import DomainOntology, RelationConstraint
from src.embeddings.embedding_service import cosine_similarity_matrix
from src.extraction.entity_extractor import normalize_entity_name
from src.extraction.schemas import OllamaLLMClient

logger = logging.getLogger(__name__)

HONORIFIC_PREFIX_REGEX = re.compile(
    r"^(prof\.|professor|dr\.|doctor|mr\.|mrs\.|ms\.|assoc\.\s*prof\.|asst\.\s*prof\.)\s+",
    re.IGNORECASE,
)

DEFAULT_ENTITY_RESOLUTION_PROMPT = """You are a conservative entity resolution judge for an enterprise knowledge graph. Your task is to determine whether two entity records refer to the exact same real-world entity or concept.

ENTITY A:
{entity_a_json}

ENTITY B:
{entity_b_json}

INSTRUCTIONS:
1. Compare the names, entity types, descriptions, aliases, and source contexts of Entity A and Entity B.
2. Be CONSERVATIVE:
   - Return true ONLY when there is clear, unambiguous evidence that both records refer to the exact same real-world individual, institution, project, or concept.
   - If there is ambiguity, uncertainty, or if they could be two different entities with similar or shared names, return false.
   - Distinct individuals, separate departments, or related but non-identical concepts must NEVER be merged.
3. Return ONLY a valid JSON object matching the following schema:

{
  "is_same_entity": true,
  "reason": "Brief explanation of the decision"
}
"""


def strip_honorifics(name: str) -> str:
    """Strip academic and professional honorific prefixes from a name string."""
    cleaned = name.strip()
    return HONORIFIC_PREFIX_REGEX.sub("", cleaned).strip()


SUFFIX_REGEX = re.compile(r",?\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.IGNORECASE)

DIMINUTIVES: Dict[str, str] = {
    "ed": "edward", "eddie": "edward", "ted": "edward",
    "bob": "robert", "bobby": "robert", "rob": "robert",
    "bill": "william", "billy": "william", "will": "william",
    "jim": "james", "jimmy": "james",
    "tom": "thomas", "tommy": "thomas",
    "dick": "richard", "rick": "richard", "rich": "richard",
    "mike": "michael", "dan": "daniel", "danny": "daniel",
    "dave": "david", "tony": "anthony", "chris": "christopher",
    "alex": "alexander", "matt": "matthew", "joe": "joseph",
    "ben": "benjamin", "benny": "benjamin", "steve": "stephen",
}


def extract_name_suffix(name: str) -> Optional[str]:
    """Extract generational suffix (e.g. jr, sr, iii) normalized without punctuation."""
    m = SUFFIX_REGEX.search(name.strip().lower())
    return m.group(1).replace(".", "") if m else None


def strip_name_affixes(name: str) -> str:
    """Strip honorific prefixes and generational suffixes from a name."""
    cleaned = strip_honorifics(name.strip())
    cleaned = SUFFIX_REGEX.sub("", cleaned).strip()
    return cleaned


def _person_name_tokens(name: str) -> List[str]:
    """Lowercased alphanumeric tokens of a person name with honorifics and generational suffix removed."""
    cleaned = strip_name_affixes(name.lower().strip())
    return re.sub(r"[^a-z0-9\s]", " ", cleaned).split()


def is_initial_surname_form(name: str) -> bool:
    """True for a name written as one initial plus a surname, e.g. 'G. Nebikvane' or 'G Nebikvane'."""
    toks = _person_name_tokens(name)
    return len(toks) == 2 and len(toks[0]) == 1 and toks[0].isalpha() and len(toks[1]) > 1


def person_names_compatible(name1: str, name2: str) -> bool:
    """Determine whether two person names are compatible variations (nicknames, affixes, middle names, initials).

    Initial + surname rule: a name consisting of a single-letter initial (optional period) plus a surname is
    compatible with a name whose given name starts with that letter and whose last token equals the surname.
    """
    s1 = extract_name_suffix(name1)
    s2 = extract_name_suffix(name2)
    # If both have suffixes and they differ, not compatible
    if s1 and s2 and s1 != s2:
        return False
    # If one is specifically Sr. and the other is not Sr., not compatible
    if (s1 == "sr" and s2 != "sr") or (s2 == "sr" and s1 != "sr"):
        return False

    def norm_tokens(n: str) -> List[str]:
        c = strip_name_affixes(n.lower().strip())
        return re.sub(r"[^a-z0-9\s]", " ", c).split()

    t1, t2 = norm_tokens(name1), norm_tokens(name2)
    if not t1 or not t2:
        return False
    if t1 == t2:
        return True
    # Last name must match
    if t1[-1] != t2[-1]:
        return False
    # First name check (exact or diminutive match)
    fn1, fn2 = t1[0], t2[0]
    first_match = (
        fn1 == fn2
        or DIMINUTIVES.get(fn1) == fn2
        or DIMINUTIVES.get(fn2) == fn1
        or (DIMINUTIVES.get(fn1) is not None and DIMINUTIVES.get(fn1) == DIMINUTIVES.get(fn2))
    )
    if not first_match:
        # Initial + surname form (surname equality was already required above).
        if len(t1) == 2 and len(fn1) == 1 and fn1.isalpha() and len(t2) >= 2 and fn2.startswith(fn1):
            first_match = True
        elif len(t2) == 2 and len(fn2) == 1 and fn2.isalpha() and len(t1) >= 2 and fn1.startswith(fn2):
            first_match = True
    if not first_match:
        return False
    # Middle names check: if one or both omit middle names, they are compatible
    mid1, mid2 = t1[1:-1], t2[1:-1]
    if not mid1 or not mid2:
        return True
    return mid1 == mid2


class EntityResolutionVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_same_entity: bool
    reason: str = Field(default="")


class ConservativeEntityResolver:
    """Executes the 5-step entity resolution pipeline with conservative ambiguity verification."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        resolution_config: Optional[ResolutionConfig] = None,
        ontology: Optional[DomainOntology] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_client: Optional[OllamaLLMClient] = None,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/entity_resolution.txt",
        prompt_template: Optional[str] = None,
    ) -> None:
        if config is not None:
            self.resolution_config = config.resolution
            self.ontology = config.ontology
        else:
            self.resolution_config = resolution_config or ResolutionConfig()
            self.ontology = ontology or DomainOntology(
                domain_name="general",
                entity_types={"ANY": "General entity"},
                relationship_types={
                    "RELATED_TO": RelationConstraint(
                        name="RELATED_TO",
                        description="General relation",
                        allowed_source_types=[],
                        allowed_target_types=[],
                    )
                },
            )

        self.embedding_provider = embedding_provider
        self.llm_client = llm_client
        # Populated by resolve_mentions / resolve_and_remap for persistence and reporting.
        self.last_resolution_report: Dict[str, Any] = {}
        self.last_revalidated_triples: Optional[List[Relationship]] = None

        if prompt_template is not None:
            self.prompt_template = prompt_template
        elif prompt_template_path is not None and Path(prompt_template_path).exists():
            self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        else:
            self.prompt_template = DEFAULT_ENTITY_RESOLUTION_PROMPT

    # ------------------------------------------------------------------ helpers
    def _is_person(self, entity_type: Optional[str]) -> bool:
        """Whether an entity type is person-like according to the ontology (never a literal type name)."""
        return self.ontology.is_person_like_type(entity_type)

    def _flag(self, name: str) -> bool:
        return bool(getattr(self.resolution_config, name, False))

    def resolve_mentions(
        self,
        mentions: List[EntityMention],
        relationships: Optional[List[Relationship]] = None,
    ) -> Tuple[List[CanonicalEntity], Dict[str, str]]:
        """Resolve a list of EntityMention objects into CanonicalEntity records.

        `relationships` (mention-level triples) are only used by the optional type-reconciliation
        tie-break when merge_exact_name_across_types is enabled.
        """
        canonical_entities: List[CanonicalEntity] = []
        mention_to_entity: Dict[str, str] = {}
        canonical_counter = 1
        merge_across_types = self._flag("merge_exact_name_across_types")
        self.last_resolution_report = {}

        for mention in mentions:
            matched_canonical = self._find_match(mention, canonical_entities)

            if matched_canonical is not None:
                # Merge mention into existing canonical entity
                self._merge_into_canonical(matched_canonical, mention)
                mention_to_entity[mention.mention_id] = matched_canonical.entity_id
            else:
                # Create a new canonical entity
                new_id = f"E{canonical_counter:03d}"
                canonical = CanonicalEntity(
                    kb_id=mention.kb_id,
                    entity_id=new_id,
                    name=mention.surface_name,
                    normalized_name=mention.normalized_name,
                    entity_type=mention.entity_type,
                    description=mention.description,
                    aliases=[],
                    source_mention_ids=[mention.mention_id],
                    source_chunk_ids=[mention.source_chunk_id],
                    source_doc_ids=[mention.source_doc_id],
                )
                canonical_entities.append(canonical)
                mention_to_entity[mention.mention_id] = new_id
                canonical_counter += 1

        # Type reconciliation (optional): plurality vote over the mention types of each canonical entity.
        # Runs before the person post-passes so they see the reconciled types.
        if merge_across_types:
            self._reconcile_entity_types(canonical_entities, mentions, mention_to_entity, relationships)

        # Rule C2 Post-pass: Handle single-token person-like canonicals created before multi-token ones
        if getattr(self.resolution_config, "enable_single_token_person_merge", True):
            single_token_persons = [
                c for c in canonical_entities
                if self._is_person(c.entity_type)
                and len(normalize_entity_name(strip_honorifics(c.name)).split()) == 1
            ]
            multi_token_persons = [
                c for c in canonical_entities
                if self._is_person(c.entity_type)
                and len(normalize_entity_name(strip_honorifics(c.name)).split()) >= 2
            ]

            merged_single_ids: Set[str] = set()
            for s_c in single_token_persons:
                s_m = extract_name_suffix(s_c.name) or extract_name_suffix(s_c.normalized_name)
                s_surname = normalize_entity_name(strip_honorifics(s_c.name)).split()[0]
                matches = [
                    m_c for m_c in multi_token_persons
                    if (extract_name_suffix(m_c.name) or extract_name_suffix(m_c.normalized_name)) == s_m
                    and normalize_entity_name(strip_honorifics(m_c.name)).split()[-1] == s_surname
                ]
                if len(matches) == 1:
                    target_c = matches[0]
                    self._merge_canonical_into_canonical(target_c, s_c)
                    for m_id in s_c.source_mention_ids:
                        mention_to_entity[m_id] = target_c.entity_id
                    merged_single_ids.add(s_c.entity_id)

            if merged_single_ids:
                canonical_entities = [c for c in canonical_entities if c.entity_id not in merged_single_ids]

        # Initial + surname post-pass (optional): 'G. Nebikvane' canonicals created before the full name.
        if self._flag("enable_initial_surname_person_merge"):
            initial_forms = [
                c for c in canonical_entities
                if self._is_person(c.entity_type) and is_initial_surname_form(c.name)
            ]
            merged_initial_ids: Set[str] = set()
            for i_c in initial_forms:
                target_c = self._unique_initial_surname_target(i_c.name, canonical_entities, exclude_id=i_c.entity_id)
                if target_c is not None and target_c.entity_id not in merged_initial_ids:
                    self._merge_canonical_into_canonical(target_c, i_c)
                    for m_id in i_c.source_mention_ids:
                        mention_to_entity[m_id] = target_c.entity_id
                    merged_initial_ids.add(i_c.entity_id)
                    self.last_resolution_report.setdefault("initial_surname_merges", []).append(
                        {"merged": i_c.name, "into": target_c.name, "entity_id": target_c.entity_id}
                    )
            if merged_initial_ids:
                canonical_entities = [c for c in canonical_entities if c.entity_id not in merged_initial_ids]

        return canonical_entities, mention_to_entity

    def _unique_initial_surname_target(
        self,
        name: str,
        canonical_entities: List[CanonicalEntity],
        exclude_id: Optional[str] = None,
    ) -> Optional[CanonicalEntity]:
        """Return the single person-like full-name canonical compatible with an initial + surname form.

        Returns None when zero or more than one candidate matches (conservative: never guess).
        """
        if not is_initial_surname_form(name):
            return None
        suffix = extract_name_suffix(name)
        matches = [
            c for c in canonical_entities
            if c.entity_id != exclude_id
            and self._is_person(c.entity_type)
            and not is_initial_surname_form(c.name)
            and len(_person_name_tokens(c.name)) >= 2
            and extract_name_suffix(c.name) == suffix
            and person_names_compatible(name, c.name)
        ]
        return matches[0] if len(matches) == 1 else None

    def _reconcile_entity_types(
        self,
        canonical_entities: List[CanonicalEntity],
        mentions: List[EntityMention],
        mention_to_entity: Dict[str, str],
        relationships: Optional[List[Relationship]],
    ) -> None:
        """Set each canonical entity's type by plurality vote over its mention types.

        Ties are broken by (1) the tied type that keeps the most of the entity's evidence-checked
        relationships (status VALIDATED or REJECTED_ONTOLOGY) ontology-valid, then (2) ontology
        declaration order. To avoid circular dependence, plurality types are computed for all entities
        first; tied entities are then resolved in sorted normalized-name order, and a tied neighbour is
        treated as its first tied type in ontology declaration order. The outcome depends only on the
        multiset of mentions and relationships, not on their order.
        """
        mention_type = {m.mention_id: m.entity_type for m in mentions}
        entity_by_id = {c.entity_id: c for c in canonical_entities}

        votes: Dict[str, Counter] = {}
        for c in canonical_entities:
            votes[c.entity_id] = Counter(
                mention_type[m_id] for m_id in c.source_mention_ids if m_id in mention_type
            )

        def tied_types(eid: str) -> List[str]:
            counts = votes[eid]
            if not counts:
                return [entity_by_id[eid].entity_type]
            top = max(counts.values())
            return sorted(
                (t for t, n in counts.items() if n == top),
                key=lambda t: (self.ontology.entity_type_rank(t), t),
            )

        provisional = {eid: tied_types(eid)[0] for eid in votes}

        eligible = (ValidationStatus.VALIDATED, ValidationStatus.REJECTED_ONTOLOGY)
        rels_by_entity: Dict[str, List[Tuple[str, str, str]]] = {}
        for r in relationships or []:
            if r.validation_status not in eligible:
                continue
            src = mention_to_entity.get(r.source_entity_id)
            tgt = mention_to_entity.get(r.target_entity_id)
            if src is None or tgt is None or src == tgt:
                continue
            rels_by_entity.setdefault(src, []).append(("S", r.relation_type, tgt))
            rels_by_entity.setdefault(tgt, []).append(("T", r.relation_type, src))

        report: List[Dict[str, Any]] = []
        final_type: Dict[str, str] = {}
        ordered = sorted(canonical_entities, key=lambda c: (c.normalized_name, c.entity_id))
        for c in ordered:
            eid = c.entity_id
            candidates = tied_types(eid)
            tie_break = "plurality"
            chosen = candidates[0]
            if len(candidates) > 1:
                valid_counts: Dict[str, int] = {}
                for cand in candidates:
                    n_valid = 0
                    for role, rel_type, other in rels_by_entity.get(eid, []):
                        other_type = provisional[other]
                        if role == "S":
                            ok, _ = self.ontology.validate_triple_types(cand, rel_type, other_type)
                        else:
                            ok, _ = self.ontology.validate_triple_types(other_type, rel_type, cand)
                        n_valid += 1 if ok else 0
                    valid_counts[cand] = n_valid
                best = max(valid_counts.values())
                best_types = [t for t in candidates if valid_counts[t] == best]
                chosen = best_types[0]  # candidates are already in ontology declaration order
                tie_break = "relationship_validity" if len(best_types) == 1 else "ontology_declaration_order"
                report.append(
                    {
                        "entity_id": eid,
                        "name": c.name,
                        "votes": dict(votes[eid]),
                        "tied_types": candidates,
                        "valid_relationship_counts": valid_counts,
                        "chosen_type": chosen,
                        "tie_break": tie_break,
                    }
                )
            elif len(votes[eid]) > 1:
                report.append(
                    {
                        "entity_id": eid,
                        "name": c.name,
                        "votes": dict(votes[eid]),
                        "chosen_type": chosen,
                        "tie_break": tie_break,
                    }
                )
            final_type[eid] = chosen

        for c in canonical_entities:
            c.type_vote_counts = dict(sorted(votes[c.entity_id].items()))
            c.entity_type = final_type[c.entity_id]
        self.last_resolution_report["type_reconciliation"] = report

    def resolve_and_remap(
        self, mentions: List[EntityMention], relationships: List[Relationship]
    ) -> Tuple[List[CanonicalEntity], Dict[str, str], List[Relationship]]:
        """Resolve mentions and remap validated relationship endpoints with deduplication."""
        canonical_entities, mention_to_entity = self.resolve_mentions(mentions, relationships)

        # Optional: re-validate ontology constraints with canonical entity types (after resolution,
        # before the VALIDATED filter). Only triples that passed every evidence check are touched.
        self.last_revalidated_triples = None
        original_status: Dict[str, ValidationStatus] = {}
        if self._flag("revalidate_relationships_after_resolution"):
            relationships = self._revalidate_with_canonical_types(
                relationships, canonical_entities, mention_to_entity
            )
            self.last_revalidated_triples = relationships
            original_status = {
                r.relationship_id: r.mention_level_status
                for r in relationships
                if r.mention_level_status is not None
            }

        # Aggregate validated relationships
        aggregated: Dict[Tuple[str, str, str, Optional[str]], Relationship] = {}

        for r in relationships:
            if r.validation_status != ValidationStatus.VALIDATED:
                continue

            new_src = mention_to_entity.get(r.source_entity_id, r.source_entity_id)
            new_tgt = mention_to_entity.get(r.target_entity_id, r.target_entity_id)

            # Drop relationships that resolve to self-loops
            if new_src == new_tgt:
                logger.info(
                    "Dropping self-loop relationship %s -> %s after resolution",
                    new_src,
                    new_tgt,
                )
                continue

            key = (new_src, r.relation_type, new_tgt, r.scope_id)
            if key in aggregated:
                existing = aggregated[key]
                merged_chunks = list(
                    dict.fromkeys(existing.source_chunk_ids + r.source_chunk_ids)
                )
                merged_docs = list(
                    dict.fromkeys(existing.source_doc_ids + r.source_doc_ids)
                )
                merged_quotes = list(
                    dict.fromkeys(existing.evidence_quotes + r.evidence_quotes)
                )

                conf = None
                if existing.uncalibrated_confidence is not None and r.uncalibrated_confidence is not None:
                    conf = max(existing.uncalibrated_confidence, r.uncalibrated_confidence)
                elif existing.uncalibrated_confidence is not None:
                    conf = existing.uncalibrated_confidence
                elif r.uncalibrated_confidence is not None:
                    conf = r.uncalibrated_confidence

                aggregated[key] = Relationship(
                    kb_id=r.kb_id,
                    relationship_id=existing.relationship_id,
                    source_entity_id=new_src,
                    target_entity_id=new_tgt,
                    relation_type=r.relation_type,
                    source_chunk_ids=merged_chunks,
                    source_doc_ids=merged_docs,
                    evidence_quotes=merged_quotes,
                    support_count=len(merged_chunks),
                    polarity=Polarity.POSITIVE,
                    validation_status=ValidationStatus.VALIDATED,
                    uncalibrated_confidence=conf,
                    scope_id=r.scope_id,
                    mention_level_status=existing.mention_level_status,
                )
            else:
                rel_id = f"REL_{len(aggregated) + 1:04d}"
                aggregated[key] = Relationship(
                    kb_id=r.kb_id,
                    relationship_id=rel_id,
                    source_entity_id=new_src,
                    target_entity_id=new_tgt,
                    relation_type=r.relation_type,
                    source_chunk_ids=list(dict.fromkeys(r.source_chunk_ids)),
                    source_doc_ids=list(dict.fromkeys(r.source_doc_ids)),
                    evidence_quotes=list(dict.fromkeys(r.evidence_quotes)),
                    support_count=len(set(r.source_chunk_ids)),
                    polarity=Polarity.POSITIVE,
                    validation_status=ValidationStatus.VALIDATED,
                    uncalibrated_confidence=r.uncalibrated_confidence,
                    scope_id=r.scope_id,
                    mention_level_status=original_status.get(r.relationship_id),
                )

        return canonical_entities, mention_to_entity, list(aggregated.values())

    def _revalidate_with_canonical_types(
        self,
        relationships: List[Relationship],
        canonical_entities: List[CanonicalEntity],
        mention_to_entity: Dict[str, str],
    ) -> List[Relationship]:
        """Recompute ontology validity of mention-level triples using canonical endpoint types.

        Only triples whose status is VALIDATED or REJECTED_ONTOLOGY are recomputed; triples rejected for
        quote, endpoint, negation or uncertainty reasons are returned unchanged. The original status is
        kept in `mention_level_status`. Returns new Relationship objects (inputs are not mutated).
        """
        type_by_entity = {c.entity_id: c.entity_type for c in canonical_entities}
        eligible = (ValidationStatus.VALIDATED, ValidationStatus.REJECTED_ONTOLOGY)
        out: List[Relationship] = []
        changes: List[Dict[str, str]] = []
        for r in relationships:
            if r.validation_status not in eligible:
                out.append(r)
                continue
            src_type = type_by_entity.get(mention_to_entity.get(r.source_entity_id, ""))
            tgt_type = type_by_entity.get(mention_to_entity.get(r.target_entity_id, ""))
            if src_type is None or tgt_type is None:
                out.append(r)
                continue
            ok, _ = self.ontology.validate_triple_types(src_type, r.relation_type, tgt_type)
            new_status = ValidationStatus.VALIDATED if ok else ValidationStatus.REJECTED_ONTOLOGY
            if new_status != r.validation_status:
                changes.append(
                    {
                        "relationship_id": r.relationship_id,
                        "relation_type": r.relation_type,
                        "from": r.validation_status.value,
                        "to": new_status.value,
                        "canonical_source_type": src_type,
                        "canonical_target_type": tgt_type,
                    }
                )
            out.append(
                r.model_copy(
                    update={"validation_status": new_status, "mention_level_status": r.validation_status}
                )
            )
        self.last_resolution_report["revalidation"] = {
            "recomputed": sum(1 for r in out if r.mention_level_status is not None),
            "validated_to_rejected_ontology": sum(1 for c in changes if c["to"] == "REJECTED_ONTOLOGY"),
            "rejected_ontology_to_validated": sum(1 for c in changes if c["to"] == "VALIDATED"),
            "changes": changes,
        }
        return out

    def _find_match(
        self, mention: EntityMention, canonical_entities: List[CanonicalEntity]
    ) -> Optional[CanonicalEntity]:
        """Find matching canonical entity using 17B (exact/alias) and 17C/D (embedding/LLM)."""
        # Optional: identical normalized names merge regardless of entity type (type chosen later by vote).
        if self._flag("merge_exact_name_across_types"):
            for canonical in canonical_entities:
                if mention.normalized_name == canonical.normalized_name:
                    return canonical

        # Step 17B: Exact and Alias Matching
        for canonical in canonical_entities:
            if self.resolution_config.require_type_match and mention.entity_type != canonical.entity_type:
                continue

            if self._is_exact_or_alias_match(mention, canonical):
                return canonical

        # Rule C2 (person-like single-token surname merge during sequential pass)
        if (
            getattr(self.resolution_config, "enable_single_token_person_merge", True)
            and self._is_person(mention.entity_type)
        ):
            s_m = extract_name_suffix(mention.surface_name) or extract_name_suffix(mention.normalized_name)
            m_toks = normalize_entity_name(strip_honorifics(mention.surface_name)).split()
            if len(m_toks) == 1:
                surname = m_toks[0]
                matching_canonicals = []
                for c in canonical_entities:
                    if not self._is_person(c.entity_type):
                        continue
                    s_c = extract_name_suffix(c.name) or extract_name_suffix(c.normalized_name)
                    if s_m != s_c:
                        continue
                    c_toks = normalize_entity_name(strip_honorifics(c.name)).split()
                    if len(c_toks) >= 2 and c_toks[-1] == surname:
                        matching_canonicals.append(c)
                if len(matching_canonicals) == 1:
                    return matching_canonicals[0]

        # Optional: initial + surname mention merges into the single compatible full-name canonical.
        if self._flag("enable_initial_surname_person_merge") and self._is_person(mention.entity_type):
            initial_target = self._unique_initial_surname_target(mention.surface_name, canonical_entities)
            if initial_target is not None:
                self.last_resolution_report.setdefault("initial_surname_merges", []).append(
                    {"merged": mention.surface_name, "into": initial_target.name, "entity_id": initial_target.entity_id}
                )
                return initial_target

        # Step 17C: Embedding Candidate Discovery
        if not canonical_entities or self.embedding_provider is None:
            return None

        candidates = [
            c
            for c in canonical_entities
            if not self.resolution_config.require_type_match or c.entity_type == mention.entity_type
        ]
        if self._is_person(mention.entity_type):
            s_m = extract_name_suffix(mention.surface_name) or extract_name_suffix(mention.normalized_name)
            filtered_candidates = []
            for c in candidates:
                if self._is_person(c.entity_type):
                    s_c = extract_name_suffix(c.name) or extract_name_suffix(c.normalized_name)
                    if s_m != s_c:
                        continue
                filtered_candidates.append(c)
            candidates = filtered_candidates
        if not candidates:
            return None

        mention_text = f"{mention.surface_name} ({mention.entity_type}): {mention.description}"
        try:
            mention_vec = np.asarray(self.embedding_provider.embed_text(mention_text))
            candidate_texts = [f"{c.name} ({c.entity_type}): {c.description}" for c in candidates]
            candidate_vecs = np.asarray(self.embedding_provider.embed_batch(candidate_texts))
            sims = cosine_similarity_matrix(mention_vec, candidate_vecs)[0]
        except Exception as e:
            logger.warning("Error generating embeddings during resolution: %s", e)
            return None

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_candidate = candidates[best_idx]

        # Step 17D: Verification Thresholds
        if self._is_person(mention.entity_type) and self._is_person(best_candidate.entity_type):
            s_m = extract_name_suffix(mention.surface_name) or extract_name_suffix(mention.normalized_name)
            s_c = extract_name_suffix(best_candidate.name) or extract_name_suffix(best_candidate.normalized_name)
            if s_m != s_c:
                logger.info(
                    "Generational suffix mismatch between '%s' and candidate '%s'; skipping merge.",
                    mention.surface_name,
                    best_candidate.name,
                )
                return None

        if best_sim >= self.resolution_config.auto_merge_threshold:
            logger.debug(
                "Auto-merging '%s' into '%s' (sim=%.4f >= %.2f)",
                mention.surface_name,
                best_candidate.name,
                best_sim,
                self.resolution_config.auto_merge_threshold,
            )
            return best_candidate

        if (
            self.resolution_config.similarity_candidate_threshold <= best_sim < self.resolution_config.auto_merge_threshold
            and self.resolution_config.use_llm_verification_for_ambiguous
            and self.llm_client is not None
        ):
            if self._verify_ambiguous_with_llm(mention, best_candidate):
                logger.info(
                    "LLM confirmed merge of '%s' into '%s' (sim=%.4f)",
                    mention.surface_name,
                    best_candidate.name,
                    best_sim,
                )
                return best_candidate
            else:
                logger.info(
                    "LLM conservatively rejected merge of '%s' into '%s' (sim=%.4f)",
                    mention.surface_name,
                    best_candidate.name,
                    best_sim,
                )

        return None

    def _is_exact_or_alias_match(
        self, mention: EntityMention, canonical: CanonicalEntity
    ) -> bool:
        """Check if mention matches canonical via exact name, static aliases, or honorific-stripped core tokens."""
        # 0. Generational suffix conflict check for person-like entities:
        # Generational distinction must NEVER be bypassed by any matching step.
        if self._is_person(mention.entity_type) and self._is_person(canonical.entity_type):
            s_m = extract_name_suffix(mention.surface_name) or extract_name_suffix(mention.normalized_name)
            s_c = extract_name_suffix(canonical.name) or extract_name_suffix(canonical.normalized_name)

            # Generational distinction: suffixes must match exactly (both None or both the same suffix)
            if s_m != s_c:
                return False

        # 1. Exact normalized name
        if mention.normalized_name == canonical.normalized_name:
            return True

        # 2. Match in canonical's existing aliases
        canonical_norm_aliases = {normalize_entity_name(a) for a in canonical.aliases}
        if mention.normalized_name in canonical_norm_aliases:
            return True

        # 3. Static aliases from domain ontology
        for canon_name, aliases in self.ontology.static_aliases.items():
            cluster_norms = {normalize_entity_name(canon_name)} | {
                normalize_entity_name(a) for a in aliases
            }
            if mention.normalized_name in cluster_norms and canonical.normalized_name in cluster_norms:
                return True

        # 4. Academic / professional honorific prefix stripping for PERSON
        # (e.g., 'Prof. Sharma' == 'Dr. Sharma' == 'Professor Sharma' -> 'sharma')
        if self._is_person(mention.entity_type) and self._is_person(canonical.entity_type):
            s_m = extract_name_suffix(mention.surface_name) or extract_name_suffix(mention.normalized_name)
            s_c = extract_name_suffix(canonical.name) or extract_name_suffix(canonical.normalized_name)

            # Suffixes must match identically (either both None, or both the same suffix like 'jr')
            # Suffix stripping alone must NEVER equate a suffixed name with an unsuffixed name!
            if s_m == s_c:
                # Compare honorific-stripped normalized forms (handles Prof./Dr./Mr./Ms.)
                m_no_honorific = normalize_entity_name(strip_honorifics(mention.surface_name))
                c_no_honorific = normalize_entity_name(strip_honorifics(canonical.name))
                # Surnames alone (e.g. 'Sharma' from 'Prof. Sharma' vs 'Dr. Sharma')
                # are NOT sufficient proof of identity without given names or static aliases.
                # Only multi-token names containing given name + surname (e.g. 'Ananya Sharma')
                # are considered sufficiently informative to auto-merge across title variants.
                if (
                    m_no_honorific
                    and m_no_honorific == c_no_honorific
                    and len(m_no_honorific.split()) >= 2
                ):
                    return True

                # If both have the identical generational suffix (e.g. both 'jr'),
                # allow punctuation variation in suffix formatting (e.g. 'Wood, Jr.' vs 'Wood Jr.')
                if s_m is not None:
                    m_core = normalize_entity_name(strip_name_affixes(mention.surface_name))
                    c_core = normalize_entity_name(strip_name_affixes(canonical.name))
                    if m_core and m_core == c_core and len(m_core.split()) >= 2:
                        return True

        # 5. Type-specific generic-descriptor normalization for non-PERSON (Rule C1)
        if (
            getattr(self.resolution_config, "enable_descriptor_normalization", True)
            and not self._is_person(mention.entity_type)
            and not self._is_person(canonical.entity_type)
            and mention.entity_type == canonical.entity_type
        ):
            descriptors = []
            if hasattr(self.ontology, "generic_descriptors") and self.ontology.generic_descriptors:
                descriptors = self.ontology.generic_descriptors.get(mention.entity_type, [])
            if not descriptors and mention.entity_type == "PROJECT":
                descriptors = ["project", "program", "programme", "initiative"]

            if descriptors:
                def strip_generic_descriptor(norm_name: str) -> str:
                    toks = norm_name.split()
                    if len(toks) > 1 and toks[-1] in descriptors:
                        return " ".join(toks[:-1])
                    return norm_name

                m_stripped = strip_generic_descriptor(mention.normalized_name)
                c_stripped = strip_generic_descriptor(canonical.normalized_name)
                if m_stripped != mention.normalized_name or c_stripped != canonical.normalized_name:
                    if (
                        m_stripped == canonical.normalized_name
                        or c_stripped == mention.normalized_name
                        or (m_stripped == c_stripped and m_stripped)
                    ):
                        return True

        return False

    def _verify_ambiguous_with_llm(
        self, mention: EntityMention, candidate: CanonicalEntity
    ) -> bool:
        """Query LLM to decide ambiguous resolution in the [candidate_threshold, auto_merge_threshold) band."""
        entity_a_json = json.dumps(
            {
                "name": candidate.name,
                "entity_type": candidate.entity_type,
                "description": candidate.description,
                "aliases": candidate.aliases,
            },
            indent=2,
        )
        entity_b_json = json.dumps(
            {
                "name": mention.surface_name,
                "entity_type": mention.entity_type,
                "description": mention.description,
                "source_chunk_id": mention.source_chunk_id,
            },
            indent=2,
        )

        prompt = (
            self.prompt_template.replace("{entity_a_json}", entity_a_json)
            .replace("{entity_b_json}", entity_b_json)
        )

        try:
            verdict: EntityResolutionVerdict = self.llm_client.generate_json(
                prompt, EntityResolutionVerdict
            )
            return verdict.is_same_entity
        except Exception as e:
            logger.warning("LLM verification failed during resolution, defaulting to conservative separate: %s", e)
            return False

    def _merge_into_canonical(
        self, canonical: CanonicalEntity, mention: EntityMention
    ) -> None:
        """Merge mention into canonical entity, preserving unique aliases and provenance IDs."""
        if mention.surface_name != canonical.name and mention.surface_name not in canonical.aliases:
            canonical.aliases.append(mention.surface_name)

        if mention.mention_id not in canonical.source_mention_ids:
            canonical.source_mention_ids.append(mention.mention_id)

        if mention.source_chunk_id not in canonical.source_chunk_ids:
            canonical.source_chunk_ids.append(mention.source_chunk_id)

        if mention.source_doc_id not in canonical.source_doc_ids:
            canonical.source_doc_ids.append(mention.source_doc_id)

        if mention.description and mention.description not in canonical.description:
            if canonical.description:
                canonical.description = f"{canonical.description}; {mention.description}"
            else:
                canonical.description = mention.description

    def _merge_canonical_into_canonical(
        self, target: CanonicalEntity, source: CanonicalEntity
    ) -> None:
        """Merge a single-token canonical entity into a target multi-token canonical entity."""
        if source.name != target.name and source.name not in target.aliases:
            target.aliases.append(source.name)
        for a in source.aliases:
            if a != target.name and a not in target.aliases:
                target.aliases.append(a)
        for m in source.source_mention_ids:
            if m not in target.source_mention_ids:
                target.source_mention_ids.append(m)
        for c in source.source_chunk_ids:
            if c not in target.source_chunk_ids:
                target.source_chunk_ids.append(c)
        for d in source.source_doc_ids:
            if d not in target.source_doc_ids:
                target.source_doc_ids.append(d)
        if source.type_vote_counts or target.type_vote_counts:
            merged_votes = Counter(target.type_vote_counts)
            merged_votes.update(source.type_vote_counts)
            target.type_vote_counts = dict(sorted(merged_votes.items()))
        if source.description and source.description not in target.description:
            if target.description:
                target.description = f"{target.description}; {source.description}"
            else:
                target.description = source.description
