"""Seed entity retriever resolving natural language entity specifications to canonical entities."""

from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple
from src.core.interfaces import EmbeddingProvider
from src.core.models import CanonicalEntity, ResolutionDecision, SeedEntitySpec
from src.core.ontology import DEFAULT_PERSON_LIKE_TYPES
from src.extraction.entity_extractor import normalize_entity_name
from src.resolution.entity_resolver import (
    person_names_compatible,
    strip_honorifics,
    strip_name_affixes,
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore
from src.retrieval.entity_description_discoverer import (
    EntityDescriptionCandidate,
    EntityDescriptionDiscoverer,
)

logger = logging.getLogger(__name__)


class SeedResolutionResult(NamedTuple):
    resolved_entity_ids: List[str]
    ambiguous_seeds: Dict[str, List[str]]
    unresolved_seeds: List[str]
    resolution_diagnostics: Tuple[ResolutionDecision, ...] = ()
    description_candidates: Tuple[EntityDescriptionCandidate, ...] = ()


class SeedEntityRetriever:
    """Resolves SeedEntitySpec objects to canonical entity IDs using confidence-aware candidate ranking."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.82,
        max_candidates_per_seed: int = 5,
        entity_description_discovery_enabled: bool = False,
        entity_description_similarity_threshold: float = 0.65,
        max_description_candidates_per_seed: int = 3,
        max_description_candidates_per_query: int = 6,
        description_discoverer: Optional[EntityDescriptionDiscoverer] = None,
        person_like_types: Optional[Iterable[str]] = None,
    ) -> None:
        if max_candidates_per_seed < 1:
            raise ValueError(
                f"max_candidates_per_seed must be >= 1, got {max_candidates_per_seed}"
            )
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_candidates_per_seed = max_candidates_per_seed
        # Entity types that get person-name matching (name variations, initial + surname). Supplied by the
        # ontology (DomainOntology.person_like_types) through the engine; never a literal type name here.
        self.person_like_types: Set[str] = {
            t.strip().upper()
            for t in (person_like_types if person_like_types is not None else DEFAULT_PERSON_LIKE_TYPES)
        }
        self.entity_description_discovery_enabled = entity_description_discovery_enabled
        self.entity_description_similarity_threshold = entity_description_similarity_threshold
        self.max_description_candidates_per_seed = max_description_candidates_per_seed
        self.max_description_candidates_per_query = max_description_candidates_per_query

        self.description_discoverer = description_discoverer or EntityDescriptionDiscoverer(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            similarity_threshold=self.entity_description_similarity_threshold,
            max_candidates_per_seed=self.max_description_candidates_per_seed,
            max_candidates_per_query=self.max_description_candidates_per_query,
        )

    def _is_person(self, entity_type: Optional[str]) -> bool:
        return bool(entity_type) and entity_type.strip().upper() in self.person_like_types

    def resolve_seeds(self, seed_specs: List[SeedEntitySpec]) -> SeedResolutionResult:
        """Resolve a list of seed entity specifications to canonical entity IDs with confidence-aware ranking."""
        resolved_ids: List[str] = []
        ambiguous: Dict[str, List[str]] = {}
        unresolved: List[str] = []
        all_decisions: List[ResolutionDecision] = []
        all_desc_candidates: List[EntityDescriptionCandidate] = []

        # Preload canonical entities from store
        entities = self.doc_store.load_canonical_entities()

        for spec in seed_specs:
            entity_ids, is_ambiguous, decisions, desc_candidates = self._resolve_single_spec_detailed(spec, entities)
            all_decisions.extend(decisions)
            all_desc_candidates.extend(desc_candidates)
            if is_ambiguous:
                ambiguous[spec.name] = entity_ids
                # Include candidate IDs so downstream traversal can explore all candidates
                for eid in entity_ids:
                    if eid not in resolved_ids:
                        resolved_ids.append(eid)
            elif entity_ids:
                for eid in entity_ids:
                    if eid not in resolved_ids:
                        resolved_ids.append(eid)
            else:
                unresolved.append(spec.name)

        return SeedResolutionResult(
            resolved_entity_ids=resolved_ids,
            ambiguous_seeds=ambiguous,
            unresolved_seeds=unresolved,
            resolution_diagnostics=tuple(all_decisions),
            description_candidates=tuple(all_desc_candidates),
        )

    def _resolve_single_spec(
        self,
        spec: SeedEntitySpec,
        entities: List[CanonicalEntity],
    ) -> Tuple[List[str], bool]:
        """Resolve a single SeedEntitySpec through the confidence-aware ranking hierarchy.

        Preserves existing 2-tuple return signature (entity_ids, is_ambiguous) for backward compatibility.
        """
        eids, is_ambiguous, _, _ = self._resolve_single_spec_detailed(spec, entities)
        return eids, is_ambiguous

    def _resolve_single_spec_detailed(
        self,
        spec: SeedEntitySpec,
        entities: List[CanonicalEntity],
    ) -> Tuple[List[str], bool, List[ResolutionDecision], List[EntityDescriptionCandidate]]:
        """Resolve a single SeedEntitySpec through confidence-aware candidate ranking."""
        norm_name = normalize_entity_name(spec.name)
        expected_type = spec.expected_type.strip().upper() if spec.expected_type else None
        decisions: List[ResolutionDecision] = []
        entities_by_id: Dict[str, CanonicalEntity] = {e.entity_id: e for e in entities}

        # -------------------------------------------------------------
        # Step 1: Exact canonical name matches & Person name variations
        # -------------------------------------------------------------
        exact_matches = [e for e in entities if e.normalized_name == norm_name]

        person_matches: List[CanonicalEntity] = []
        if self._is_person(expected_type) or any(self._is_person(e.entity_type) for e in exact_matches):
            for e in entities:
                if self._is_person(e.entity_type) and e not in exact_matches:
                    if person_names_compatible(spec.name, e.name):
                        person_matches.append(e)

        phase1_candidates: List[Tuple[CanonicalEntity, str, bool]] = []
        for e in exact_matches:
            is_comp = (expected_type is None or e.entity_type.upper() == expected_type)
            phase1_candidates.append((e, "EXACT_CANONICAL", is_comp))
        for e in person_matches:
            is_comp = (expected_type is None or e.entity_type.upper() == expected_type)
            phase1_candidates.append((e, "PERSON_NAME_VARIATION", is_comp))

        if phase1_candidates:
            compatible = [(e, m) for e, m, comp in phase1_candidates if comp]
            conflicting = [(e, m) for e, m, comp in phase1_candidates if not comp]

            if compatible:
                # Tier 1 (Score 1.00): Type-compatible exact canonical matches
                compatible.sort(key=lambda x: x[0].entity_id)
                selected = compatible[: self.max_candidates_per_seed]
                is_ambiguous = len(compatible) > 1

                for rank_idx, (cand, method) in enumerate(selected, start=1):
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=True,
                            confidence_score=1.00,
                            similarity_score=None,
                            ranking=rank_idx,
                            status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                            reason=(
                                f"Ambiguous exact canonical match with compatible type {cand.entity_type.upper()}"
                                if is_ambiguous
                                else f"Exact canonical match with compatible type {cand.entity_type.upper()}"
                            ),
                        )
                    )

                for cand, method in compatible[self.max_candidates_per_seed :]:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=True,
                            confidence_score=1.00,
                            similarity_score=None,
                            ranking=self.max_candidates_per_seed + 1,
                            status="REJECTED",
                            reason=f"Exceeded max candidates limit ({self.max_candidates_per_seed})",
                        )
                    )

                for cand, method in conflicting:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.80,
                            similarity_score=None,
                            ranking=len(selected) + 1,
                            status="REJECTED",
                            reason=f"Superseded by type-compatible exact canonical match (expected: {expected_type}, candidate: {cand.entity_type.upper()})",
                        )
                    )

                return [c[0].entity_id for c in selected], is_ambiguous, decisions, []

            else:
                # Tier 2 (Score 0.80): Exact canonical match retained despite type conflict
                conflicting.sort(key=lambda x: x[0].entity_id)
                selected = conflicting[: self.max_candidates_per_seed]
                is_ambiguous = len(conflicting) > 1

                for rank_idx, (cand, method) in enumerate(selected, start=1):
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.80,
                            similarity_score=None,
                            ranking=rank_idx,
                            status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                            reason=(
                                f"Ambiguous exact canonical match retained despite type conflict (expected: {expected_type}, actual: {cand.entity_type.upper()})"
                                if is_ambiguous
                                else f"Exact canonical name match retained despite type conflict (expected: {expected_type}, actual: {cand.entity_type.upper()})"
                            ),
                        )
                    )

                for cand, method in conflicting[self.max_candidates_per_seed :]:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.80,
                            similarity_score=None,
                            ranking=self.max_candidates_per_seed + 1,
                            status="REJECTED",
                            reason=f"Exceeded max candidates limit ({self.max_candidates_per_seed})",
                        )
                    )

                return [c[0].entity_id for c in selected], is_ambiguous, decisions, []

        # -------------------------------------------------------------
        # Step 2: Known alias match & honorific/affix-stripped match
        # -------------------------------------------------------------
        norm_stripped = normalize_entity_name(strip_name_affixes(spec.name))
        alias_candidates: List[Tuple[CanonicalEntity, str, bool]] = []

        for e in entities:
            norm_aliases = [normalize_entity_name(a) for a in e.aliases]
            norm_aliases_stripped = [
                normalize_entity_name(strip_name_affixes(a)) for a in e.aliases
            ]
            e_name_stripped = normalize_entity_name(strip_name_affixes(e.name))

            matched = False
            method = ""
            if norm_name in norm_aliases:
                matched = True
                method = "EXACT_ALIAS"
            elif norm_stripped and (
                norm_stripped == e_name_stripped or norm_stripped in norm_aliases_stripped
            ):
                matched = True
                method = "AFFIX_STRIPPED"
            elif (
                # Runs for every person-like entity whatever the planner's expected_type (including None),
                # so alias seeds such as initial + surname resolve without a planner type.
                (self._is_person(expected_type) or self._is_person(e.entity_type))
                and person_names_compatible(spec.name, e.name)
            ):
                matched = True
                method = "PERSON_NAME_COMPATIBLE"

            if matched:
                is_comp = (expected_type is None or e.entity_type.upper() == expected_type)
                alias_candidates.append((e, method, is_comp))

        if alias_candidates:
            compatible = [(e, m) for e, m, comp in alias_candidates if comp]
            conflicting = [(e, m) for e, m, comp in alias_candidates if not comp]

            if compatible:
                # Tier 3 (Score 0.75): Type-compatible alias match
                compatible.sort(key=lambda x: x[0].entity_id)
                selected = compatible[: self.max_candidates_per_seed]
                is_ambiguous = len(compatible) > 1

                for rank_idx, (cand, method) in enumerate(selected, start=1):
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=True,
                            confidence_score=0.75,
                            similarity_score=None,
                            ranking=rank_idx,
                            status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                            reason=(
                                f"Ambiguous alias match with compatible type {cand.entity_type.upper()}"
                                if is_ambiguous
                                else f"Exact alias match with compatible type {cand.entity_type.upper()}"
                            ),
                        )
                    )

                for cand, method in compatible[self.max_candidates_per_seed :]:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=True,
                            confidence_score=0.75,
                            similarity_score=None,
                            ranking=self.max_candidates_per_seed + 1,
                            status="REJECTED",
                            reason=f"Exceeded max candidates limit ({self.max_candidates_per_seed})",
                        )
                    )

                for cand, method in conflicting:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.60,
                            similarity_score=None,
                            ranking=len(selected) + 1,
                            status="REJECTED",
                            reason=f"Superseded by type-compatible alias match (expected: {expected_type}, candidate: {cand.entity_type.upper()})",
                        )
                    )

                return [c[0].entity_id for c in selected], is_ambiguous, decisions, []

            else:
                # Tier 4 (Score 0.60): Alias match retained despite type conflict
                conflicting.sort(key=lambda x: x[0].entity_id)
                selected = conflicting[: self.max_candidates_per_seed]
                is_ambiguous = len(conflicting) > 1

                for rank_idx, (cand, method) in enumerate(selected, start=1):
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.60,
                            similarity_score=None,
                            ranking=rank_idx,
                            status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                            reason=(
                                f"Ambiguous alias match retained despite type conflict (expected: {expected_type}, actual: {cand.entity_type.upper()})"
                                if is_ambiguous
                                else f"Exact alias match retained despite type conflict (expected: {expected_type}, actual: {cand.entity_type.upper()})"
                            ),
                        )
                    )

                for cand, method in conflicting[self.max_candidates_per_seed :]:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=cand.entity_id,
                            candidate_name=cand.name,
                            candidate_type=cand.entity_type.upper(),
                            matching_method=method,
                            type_compatible=False,
                            confidence_score=0.60,
                            similarity_score=None,
                            ranking=self.max_candidates_per_seed + 1,
                            status="REJECTED",
                            reason=f"Exceeded max candidates limit ({self.max_candidates_per_seed})",
                        )
                    )

                return [c[0].entity_id for c in selected], is_ambiguous, decisions, []

        # -------------------------------------------------------------
        # Step 3: Embedding-based candidate discovery (Task 4)
        # -------------------------------------------------------------
        query_emb = self.embedding_provider.embed_text(spec.name)

        # 1. First probe for type-compatible candidates if expected_type is specified
        raw_candidates: List[Dict[str, Any]] = []
        if expected_type:
            raw_candidates = self.vector_store.query_entities(
                query_embedding=query_emb,
                top_k=5,
                entity_type=expected_type,
            )

        # 2. Check if any compatible candidate meets the similarity threshold
        compatible_found = False
        for cand in raw_candidates:
            dist = cand.get("distance")
            sim = 1.0 - dist if dist is not None else 0.0
            if sim >= self.similarity_threshold:
                compatible_found = True
                break

        # 3. If no compatible candidate met threshold, query without entity_type filter
        if not compatible_found:
            raw_candidates = self.vector_store.query_entities(
                query_embedding=query_emb,
                top_k=max(10, self.max_candidates_per_seed * 2),
                entity_type=None,
            )

        valid_candidates: List[Tuple[str, str, str, float, bool, float]] = []
        # format: (entity_id, name, type, similarity, is_compatible, confidence_score)

        for cand in raw_candidates:
            meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
            eid = cand.get("id") or meta.get("entity_id")
            dist = cand.get("distance")
            sim = 1.0 - dist if dist is not None else 0.0

            cand_meta_name = str(meta.get("name", "Unknown")) if meta.get("name") is not None else "Unknown"
            cand_meta_type = str(meta.get("entity_type", "UNKNOWN")) if meta.get("entity_type") is not None else "UNKNOWN"

            # Validate that canonical entity exists in the knowledge base
            ent_obj = entities_by_id.get(eid) if eid else None
            if not ent_obj:
                decisions.append(
                    ResolutionDecision(
                        seed_name=spec.name,
                        expected_type=expected_type,
                        candidate_id=eid or "UNKNOWN",
                        candidate_name=cand_meta_name,
                        candidate_type=cand_meta_type,
                        matching_method="EMBEDDING",
                        type_compatible=False,
                        confidence_score=0.0,
                        similarity_score=round(sim, 4),
                        ranking=len(raw_candidates),
                        status="REJECTED",
                        reason="Canonical entity ID missing or absent from knowledge base document store",
                    )
                )
                continue

            cand_name = cand_meta_name if cand_meta_name != "Unknown" else ent_obj.name
            cand_type = cand_meta_type if cand_meta_type != "UNKNOWN" else ent_obj.entity_type.upper()

            is_comp = (expected_type is None or cand_type.upper() == expected_type)
            score = sim if is_comp else (sim * 0.70)

            if sim >= self.similarity_threshold:
                valid_candidates.append((eid, cand_name, cand_type.upper(), sim, is_comp, score))
            else:
                decisions.append(
                    ResolutionDecision(
                        seed_name=spec.name,
                        expected_type=expected_type,
                        candidate_id=eid,
                        candidate_name=cand_name,
                        candidate_type=cand_type.upper(),
                        matching_method="EMBEDDING",
                        type_compatible=is_comp,
                        confidence_score=0.0,
                        similarity_score=round(sim, 4),
                        ranking=len(raw_candidates),
                        status="REJECTED",
                        reason=f"Embedding similarity {sim:.3f} below threshold {self.similarity_threshold}",
                    )
                )

        if not valid_candidates:
            # -------------------------------------------------------------
            # Step 4: Entity-description-based fallback discovery (Task 6)
            # Only fires after Steps 1-3 (exact, alias, embedding) all fail.
            # -------------------------------------------------------------
            if self.entity_description_discovery_enabled:
                candidates, is_ambiguous, desc_decisions = self.description_discoverer.discover_candidates(
                    seed_spec=spec,
                    entities_by_id=entities_by_id,
                    query_embedding=query_emb,
                )
                decisions.extend(desc_decisions)
                if candidates:
                    return [c.candidate_id for c in candidates], is_ambiguous, decisions, candidates
                return [], False, decisions, []

            decisions.append(
                ResolutionDecision(
                    seed_name=spec.name,
                    expected_type=expected_type,
                    candidate_id=None,
                    candidate_name=None,
                    candidate_type=None,
                    matching_method=None,
                    type_compatible=False,
                    confidence_score=0.0,
                    similarity_score=None,
                    ranking=0,
                    status="UNRESOLVED",
                    reason=f"No candidate met exact canonical, alias, or embedding similarity threshold ({self.similarity_threshold})",
                )
            )
            return [], False, decisions, []

        # Split into compatible (Tier 5) and conflicting (Tier 6)
        comp_v = [c for c in valid_candidates if c[4]]
        conf_v = [c for c in valid_candidates if not c[4]]

        if comp_v:
            # Sort by similarity descending
            comp_v.sort(key=lambda x: x[3], reverse=True)
            top_cand = comp_v[0]
            top_sim = top_cand[3]

            # Ambiguity check: virtually tied candidates within 0.02
            tied = [c for c in comp_v if abs(top_sim - c[3]) < 0.02]
            is_ambiguous = len(tied) > 1
            selected = tied[: self.max_candidates_per_seed] if is_ambiguous else [top_cand]

            for rank_idx, (eid, cname, ctype, sim, is_comp, score) in enumerate(selected, start=1):
                decisions.append(
                    ResolutionDecision(
                        seed_name=spec.name,
                        expected_type=expected_type,
                        candidate_id=eid,
                        candidate_name=cname,
                        candidate_type=ctype,
                        matching_method="EMBEDDING",
                        type_compatible=True,
                        confidence_score=round(score, 4),
                        similarity_score=round(sim, 4),
                        ranking=rank_idx,
                        status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                        reason=(
                            f"Embedding match virtually tied with top candidate (similarity: {sim:.3f}, compatible type {ctype})"
                            if is_ambiguous
                            else f"Embedding similarity {sim:.3f} meets threshold with compatible type {ctype}"
                        ),
                    )
                )

            selected_ids = {c[0] for c in selected}
            for eid, cname, ctype, sim, is_comp, score in comp_v:
                if eid not in selected_ids:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=eid,
                            candidate_name=cname,
                            candidate_type=ctype,
                            matching_method="EMBEDDING",
                            type_compatible=True,
                            confidence_score=round(score, 4),
                            similarity_score=round(sim, 4),
                            ranking=len(selected) + 1,
                            status="REJECTED",
                            reason=f"Lower similarity than top candidate ({sim:.3f} vs {top_sim:.3f})",
                        )
                    )
            for eid, cname, ctype, sim, is_comp, score in conf_v:
                decisions.append(
                    ResolutionDecision(
                        seed_name=spec.name,
                        expected_type=expected_type,
                        candidate_id=eid,
                        candidate_name=cname,
                        candidate_type=ctype,
                        matching_method="EMBEDDING",
                        type_compatible=False,
                        confidence_score=round(score, 4),
                        similarity_score=round(sim, 4),
                        ranking=len(selected) + 1,
                        status="REJECTED",
                        reason=f"Superseded by type-compatible embedding match (expected: {expected_type}, candidate: {ctype})",
                    )
                )

            return [c[0] for c in selected], is_ambiguous, decisions, []

        else:
            # Tier 6: Embedding candidate retained despite type conflict
            conf_v.sort(key=lambda x: x[3], reverse=True)
            top_cand = conf_v[0]
            top_sim = top_cand[3]

            tied = [c for c in conf_v if abs(top_sim - c[3]) < 0.02]
            is_ambiguous = len(tied) > 1
            selected = tied[: self.max_candidates_per_seed] if is_ambiguous else [top_cand]

            for rank_idx, (eid, cname, ctype, sim, is_comp, score) in enumerate(selected, start=1):
                decisions.append(
                    ResolutionDecision(
                        seed_name=spec.name,
                        expected_type=expected_type,
                        candidate_id=eid,
                        candidate_name=cname,
                        candidate_type=ctype,
                        matching_method="EMBEDDING",
                        type_compatible=False,
                        confidence_score=round(score, 4),
                        similarity_score=round(sim, 4),
                        ranking=rank_idx,
                        status="AMBIGUOUS" if is_ambiguous else "ACCEPTED",
                        reason=(
                            f"Embedding match virtually tied, retained despite type conflict (expected: {expected_type}, actual: {ctype}, similarity: {sim:.3f})"
                            if is_ambiguous
                            else f"Embedding match retained despite type conflict (expected: {expected_type}, actual: {ctype}, similarity: {sim:.3f})"
                        ),
                    )
                )

            selected_ids = {c[0] for c in selected}
            for eid, cname, ctype, sim, is_comp, score in conf_v:
                if eid not in selected_ids:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=spec.name,
                            expected_type=expected_type,
                            candidate_id=eid,
                            candidate_name=cname,
                            candidate_type=ctype,
                            matching_method="EMBEDDING",
                            type_compatible=False,
                            confidence_score=round(score, 4),
                            similarity_score=round(sim, 4),
                            ranking=len(selected) + 1,
                            status="REJECTED",
                            reason=f"Lower similarity than top candidate ({sim:.3f} vs {top_sim:.3f})",
                        )
                    )

            return [c[0] for c in selected], is_ambiguous, decisions, []
