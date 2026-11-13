"""Seed entity retriever resolving natural language entity specifications to canonical entities."""

from __future__ import annotations
import logging
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
from src.core.interfaces import EmbeddingProvider
from src.core.models import CanonicalEntity, SeedEntitySpec
from src.extraction.entity_extractor import normalize_entity_name
from src.resolution.entity_resolver import (
    person_names_compatible,
    strip_honorifics,
    strip_name_affixes,
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class SeedResolutionResult(NamedTuple):
    resolved_entity_ids: List[str]
    ambiguous_seeds: Dict[str, List[str]]
    unresolved_seeds: List[str]


class SeedEntityRetriever:
    """Resolves SeedEntitySpec objects to canonical entity IDs using 4-step conservative resolution."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.82,
    ) -> None:
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def resolve_seeds(self, seed_specs: List[SeedEntitySpec]) -> SeedResolutionResult:
        """Resolve a list of seed entity specifications to canonical entity IDs."""
        resolved_ids: List[str] = []
        ambiguous: Dict[str, List[str]] = {}
        unresolved: List[str] = []

        # Preload canonical entities from store
        entities = self.doc_store.load_canonical_entities()

        for spec in seed_specs:
            entity_ids, is_ambiguous = self._resolve_single_spec(spec, entities)
            if is_ambiguous:
                ambiguous[spec.name] = entity_ids
                # Still include candidate IDs so downstream traversal can explore if desired
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
        )

    def _resolve_single_spec(
        self,
        spec: SeedEntitySpec,
        entities: List[CanonicalEntity],
    ) -> Tuple[List[str], bool]:
        """Resolve a single SeedEntitySpec through the 4-step hierarchy."""
        norm_name = normalize_entity_name(spec.name)
        expected_type = spec.expected_type.upper() if spec.expected_type else None

        # --- Step 1: Exact normalized canonical name match & compatible name variations ---
        exact_matches = [
            e for e in entities if e.normalized_name == norm_name
        ]
        candidate_entities: List[CanonicalEntity] = []
        if exact_matches:
            if expected_type:
                type_filtered = [e for e in exact_matches if e.entity_type.upper() == expected_type]
            else:
                type_filtered = exact_matches
            candidate_entities.extend(type_filtered)

        # Check for compatible person name variations (e.g. diminutives, suffixes, middle names)
        if expected_type == "PERSON" or any(e.entity_type.upper() == "PERSON" for e in candidate_entities):
            for e in entities:
                if e.entity_type.upper() == "PERSON" and e not in candidate_entities:
                    if person_names_compatible(spec.name, e.name):
                        candidate_entities.append(e)

        if candidate_entities:
            if len(candidate_entities) == 1:
                return [candidate_entities[0].entity_id], False
            return [e.entity_id for e in candidate_entities], True

        # --- Step 2: Known alias match & honorific/affix-stripped match ---
        norm_stripped = normalize_entity_name(strip_name_affixes(spec.name))
        alias_matches: List[CanonicalEntity] = []

        for e in entities:
            # Check aliases
            norm_aliases = [normalize_entity_name(a) for a in e.aliases]
            norm_aliases_stripped = [
                normalize_entity_name(strip_name_affixes(a)) for a in e.aliases
            ]
            e_name_stripped = normalize_entity_name(strip_name_affixes(e.name))

            matched = False
            if norm_name in norm_aliases:
                matched = True
            elif norm_stripped and (
                norm_stripped == e_name_stripped or norm_stripped in norm_aliases_stripped
            ):
                matched = True
            elif (
                (expected_type == "PERSON" or e.entity_type.upper() == "PERSON")
                and person_names_compatible(spec.name, e.name)
            ):
                matched = True

            if matched:
                if expected_type is None or e.entity_type.upper() == expected_type:
                    alias_matches.append(e)

        if alias_matches:
            if len(alias_matches) == 1:
                return [alias_matches[0].entity_id], False
            return [e.entity_id for e in alias_matches], True

        # --- Step 3: Embedding-based candidate discovery ---
        query_emb = self.embedding_provider.embed_text(spec.name)
        candidates = self.vector_store.query_entities(
            query_embedding=query_emb,
            top_k=5,
            entity_type=expected_type,
        )

        valid_candidates = []
        for cand in candidates:
            dist = cand.get("distance")
            # In Chroma cosine distance: similarity = 1.0 - dist
            sim = 1.0 - dist if dist is not None else 0.0
            if sim >= self.similarity_threshold:
                valid_candidates.append((cand["id"], sim))

        if valid_candidates:
            # Sort by similarity descending
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            top_id, top_sim = valid_candidates[0]

            # --- Step 4: Verification & Ambiguity check ---
            if len(valid_candidates) > 1:
                second_id, second_sim = valid_candidates[1]
                # If top candidate is virtually tied with second candidate (within 0.02)
                if abs(top_sim - second_sim) < 0.02:
                    return [c[0] for c in valid_candidates[:3]], True

            return [top_id], False

        return [], False
