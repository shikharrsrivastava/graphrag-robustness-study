"""Entity-description-based seed discovery mechanism.

Discovers canonical graph entities matching natural language descriptive seed
specifications via dense embedding search over persisted entity descriptions in ChromaDB.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field
from src.core.configuration import RetrievalConfig
from src.core.interfaces import EmbeddingProvider
from src.core.models import (
    CanonicalEntity,
    EntityDescriptionProvenance,
    ResolutionDecision,
    SeedEntitySpec,
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class EntityDescriptionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_name: str = Field(..., min_length=1, description="Original seed entity query string")
    expected_type: Optional[str] = Field(default=None, description="Expected entity type from SeedEntitySpec")
    candidate_id: str = Field(..., min_length=1, description="Canonical entity ID in knowledge base")
    candidate_name: str = Field(..., min_length=1, description="Canonical entity name")
    candidate_type: str = Field(..., min_length=1, description="Canonical entity type from KB")
    matched_description: str = Field(default="", description="Persisted entity description matched")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Raw cosine similarity score")
    distance_score: Optional[float] = Field(default=None, description="Raw ChromaDB cosine distance")
    matching_method: str = Field(
        default="ENTITY_DESCRIPTION_SEARCH",
        description="Matching method identifier",
    )
    type_compatible: bool = Field(default=False, description="Whether candidate_type matches expected_type")
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic ordinal ranking heuristic score",
    )
    ranking: int = Field(default=1, ge=1, description="Rank within candidate pool")
    is_ambiguous: bool = Field(default=False, description="True if multiple candidates were virtually tied")
    candidate_ids: List[str] = Field(
        default_factory=list,
        description="All matching candidate IDs if ambiguous",
    )
    status: str = Field(..., description="ACCEPTED, AMBIGUOUS, REJECTED, or UNRESOLVED")
    reason: str = Field(..., description="Human-readable decision explanation")

    def to_provenance(self) -> EntityDescriptionProvenance:
        return EntityDescriptionProvenance(
            seed_name=self.seed_name,
            candidate_id=self.candidate_id,
            candidate_name=self.candidate_name,
            candidate_type=self.candidate_type,
            matched_description=self.matched_description,
            similarity_score=self.similarity_score,
            distance_score=self.distance_score,
            matching_method=self.matching_method,
            type_compatible=self.type_compatible,
            confidence_score=self.confidence_score,
            ranking=self.ranking,
            is_ambiguous=self.is_ambiguous,
            candidate_ids=list(self.candidate_ids),
        )


class EntityDescriptionDiscoverer:
    """Discovers canonical graph entities using dense vector search over entity_descriptions."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        config: Optional[RetrievalConfig] = None,
        similarity_threshold: float = 0.65,
        max_candidates_per_seed: int = 3,
        max_candidates_per_query: int = 6,
    ) -> None:
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

        if config is not None:
            self.similarity_threshold = config.entity_description_similarity_threshold
            self.max_candidates_per_seed = config.max_description_candidates_per_seed
            self.max_candidates_per_query = config.max_description_candidates_per_query
        else:
            self.similarity_threshold = similarity_threshold
            self.max_candidates_per_seed = max_candidates_per_seed
            self.max_candidates_per_query = max_candidates_per_query

    def discover_candidates(
        self,
        seed_spec: SeedEntitySpec,
        entities_by_id: Optional[Dict[str, CanonicalEntity]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[EntityDescriptionCandidate], bool, List[ResolutionDecision]]:
        """Discover candidate canonical entities matching a descriptive seed specification.
        
        Args:
            seed_spec: SeedEntitySpec from QueryPlan.
            entities_by_id: Optional preloaded mapping of canonical entities by ID.
            query_embedding: Optional precomputed query embedding to avoid re-embedding.
            
        Returns:
            Tuple of:
            - List of accepted/ambiguous EntityDescriptionCandidate objects (up to limit).
            - Boolean flag indicating ambiguity.
            - Complete list of ResolutionDecision diagnostic records (accepted and rejected).
        """
        seed_name = seed_spec.name.strip()
        expected_type = seed_spec.expected_type.strip().upper() if seed_spec.expected_type else None

        if not seed_name:
            unres_decision = ResolutionDecision(
                seed_name=seed_spec.name,
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
                reason="Empty seed name",
            )
            return [], False, [unres_decision]

        # Ensure canonical entity lookup is available for validation
        if entities_by_id is None:
            loaded_entities = self.doc_store.load_canonical_entities()
            entities_by_id = {e.entity_id: e for e in loaded_entities}

        decisions: List[ResolutionDecision] = []

        # 1. Embed query seed text (or reuse precomputed embedding)
        if query_embedding is not None:
            query_emb = query_embedding
        else:
            try:
                query_emb = self.embedding_provider.embed_text(seed_name)
            except Exception as exc:
                logger.warning("Embedding provider failed on seed '%s': %s", seed_name, exc)
                unres_decision = ResolutionDecision(
                    seed_name=seed_spec.name,
                    expected_type=expected_type,
                    candidate_id=None,
                    candidate_name=None,
                    candidate_type=None,
                    matching_method="ENTITY_DESCRIPTION_SEARCH",
                    type_compatible=False,
                    confidence_score=0.0,
                    similarity_score=None,
                    ranking=0,
                    status="UNRESOLVED",
                    reason=f"Embedding provider error: {exc}",
                )
                return [], False, [unres_decision]

        # 2. Query ONLY the entity_descriptions collection (never text_chunks)
        raw_candidates: List[Dict[str, Any]] = []

        # Probe with entity_type filter first if expected_type is provided
        if expected_type:
            try:
                raw_candidates = self.vector_store.query_entities(
                    query_embedding=query_emb,
                    top_k=max(5, self.max_candidates_per_seed),
                    entity_type=expected_type,
                )
            except Exception as exc:
                logger.warning("ChromaDB query_entities with type filter failed: %s", exc)
                raw_candidates = []

        # Check if any type-filtered candidate met the threshold
        has_compatible_match = False
        for c in raw_candidates:
            dist = c.get("distance")
            sim = 1.0 - dist if dist is not None else 0.0
            if sim >= self.similarity_threshold:
                has_compatible_match = True
                break

        # If no type-filtered match met threshold, query without entity_type filter
        if not has_compatible_match:
            try:
                unfiltered = self.vector_store.query_entities(
                    query_embedding=query_emb,
                    top_k=max(10, self.max_candidates_per_seed * 2),
                    entity_type=None,
                )
                raw_candidates = unfiltered
            except Exception as exc:
                logger.warning("ChromaDB query_entities unfiltered failed: %s", exc)
                unres_decision = ResolutionDecision(
                    seed_name=seed_spec.name,
                    expected_type=expected_type,
                    candidate_id=None,
                    candidate_name=None,
                    candidate_type=None,
                    matching_method="ENTITY_DESCRIPTION_SEARCH",
                    type_compatible=False,
                    confidence_score=0.0,
                    similarity_score=None,
                    ranking=0,
                    status="UNRESOLVED",
                    reason=f"Vector store error: {exc}",
                )
                return [], False, [unres_decision]

        if not raw_candidates:
            unres_decision = ResolutionDecision(
                seed_name=seed_spec.name,
                expected_type=expected_type,
                candidate_id=None,
                candidate_name=None,
                candidate_type=None,
                matching_method="ENTITY_DESCRIPTION_SEARCH",
                type_compatible=False,
                confidence_score=0.0,
                similarity_score=None,
                ranking=0,
                status="UNRESOLVED",
                reason=f"No candidates found in entity_descriptions collection",
            )
            return [], False, [unres_decision]

        # 3. Process, validate canonical ID existence, and score candidates
        scored_candidates: List[Dict[str, Any]] = []

        for cand in raw_candidates:
            meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
            eid = cand.get("id") or meta.get("entity_id")
            dist = cand.get("distance")
            sim = max(0.0, min(1.0, (1.0 - dist) if dist is not None else 0.0))

            cand_meta_name = str(meta.get("name", "Unknown")) if meta.get("name") is not None else "Unknown"
            cand_meta_type = str(meta.get("entity_type", "UNKNOWN")) if meta.get("entity_type") is not None else "UNKNOWN"

            # Validate that canonical entity exists in the knowledge base
            ent_obj = entities_by_id.get(eid) if eid else None
            if not ent_obj:
                decisions.append(
                    ResolutionDecision(
                        seed_name=seed_spec.name,
                        expected_type=expected_type,
                        candidate_id=eid or "UNKNOWN",
                        candidate_name=cand_meta_name,
                        candidate_type=cand_meta_type,
                        matching_method="ENTITY_DESCRIPTION_SEARCH",
                        type_compatible=False,
                        confidence_score=0.0,
                        similarity_score=round(sim, 4),
                        ranking=len(decisions) + 1,
                        status="REJECTED",
                        reason="Canonical entity ID missing or absent from knowledge base document store",
                    )
                )
                continue

            cname = ent_obj.name
            ctype = ent_obj.entity_type.upper()
            cdesc = ent_obj.description

            is_comp = (expected_type is None or ctype == expected_type)
            # Tier 5 (compatible): score = sim; Tier 6 (conflicting): score = sim * 0.70
            conf_score = sim if is_comp else (sim * 0.70)

            if sim >= self.similarity_threshold:
                scored_candidates.append(
                    {
                        "candidate_id": eid,
                        "candidate_name": cname,
                        "candidate_type": ctype,
                        "matched_description": cdesc,
                        "similarity_score": round(sim, 4),
                        "distance_score": round(dist, 4) if dist is not None else None,
                        "type_compatible": is_comp,
                        "confidence_score": round(conf_score, 4),
                    }
                )
            else:
                decisions.append(
                    ResolutionDecision(
                        seed_name=seed_spec.name,
                        expected_type=expected_type,
                        candidate_id=eid,
                        candidate_name=cname,
                        candidate_type=ctype,
                        matching_method="ENTITY_DESCRIPTION_SEARCH",
                        type_compatible=is_comp,
                        confidence_score=0.0,
                        similarity_score=round(sim, 4),
                        ranking=len(decisions) + 1,
                        status="REJECTED",
                        reason=f"Entity description similarity {sim:.3f} below threshold {self.similarity_threshold}",
                    )
                )

        if not scored_candidates:
            unres_decision = ResolutionDecision(
                seed_name=seed_spec.name,
                expected_type=expected_type,
                candidate_id=None,
                candidate_name=None,
                candidate_type=None,
                matching_method="ENTITY_DESCRIPTION_SEARCH",
                type_compatible=False,
                confidence_score=0.0,
                similarity_score=None,
                ranking=0,
                status="UNRESOLVED",
                reason=f"No candidate met entity description similarity threshold ({self.similarity_threshold})",
            )
            decisions.append(unres_decision)
            return [], False, decisions

        # 4. Partition by type compatibility: Tier 5 (compatible) takes precedence over Tier 6
        compatible_group = [c for c in scored_candidates if c["type_compatible"]]

        # F2 type safety: when expected_type is specified and no compatible candidates exist,
        # return UNRESOLVED instead of falling back to conflicting-type candidates.
        if expected_type and not compatible_group:
            for cand in scored_candidates:
                decisions.append(
                    ResolutionDecision(
                        seed_name=seed_spec.name,
                        expected_type=expected_type,
                        candidate_id=cand["candidate_id"],
                        candidate_name=cand["candidate_name"],
                        candidate_type=cand["candidate_type"],
                        matching_method="ENTITY_DESCRIPTION_SEARCH",
                        type_compatible=False,
                        confidence_score=cand["confidence_score"],
                        similarity_score=cand["similarity_score"],
                        ranking=len(decisions) + 1,
                        status="REJECTED",
                        reason=f"Entity description candidate type '{cand['candidate_type']}' conflicts with expected type '{expected_type}'; no compatible candidate found",
                    )
                )
            unres_decision = ResolutionDecision(
                seed_name=seed_spec.name,
                expected_type=expected_type,
                candidate_id=None,
                candidate_name=None,
                candidate_type=None,
                matching_method="ENTITY_DESCRIPTION_SEARCH",
                type_compatible=False,
                confidence_score=0.0,
                similarity_score=None,
                ranking=0,
                status="UNRESOLVED",
                reason=f"All entity description candidates conflict with expected type '{expected_type}'",
            )
            decisions.append(unres_decision)
            return [], False, decisions

        active_group = compatible_group if compatible_group else scored_candidates

        # Sort by similarity descending, then candidate_id ascending for deterministic stability
        active_group.sort(key=lambda x: (-x["similarity_score"], x["candidate_id"]))

        top_sim = active_group[0]["similarity_score"]
        # Virtually tied candidates within 0.02 of top score are considered ambiguous
        tied_candidates = [c for c in active_group if abs(top_sim - c["similarity_score"]) < 0.02]
        is_ambiguous = len(tied_candidates) > 1

        selected_candidates = tied_candidates[: self.max_candidates_per_seed]
        selected_ids = [c["candidate_id"] for c in selected_candidates]
        all_tied_ids = [c["candidate_id"] for c in tied_candidates]

        accepted_objects: List[EntityDescriptionCandidate] = []

        for rank_idx, cand in enumerate(selected_candidates, start=1):
            status = "AMBIGUOUS" if is_ambiguous else "ACCEPTED"
            reason = (
                f"Ambiguous entity description match virtually tied with top candidate (similarity: {cand['similarity_score']:.3f})"
                if is_ambiguous
                else f"Entity description similarity {cand['similarity_score']:.3f} meets threshold {self.similarity_threshold}"
            )
            cand_obj = EntityDescriptionCandidate(
                seed_name=seed_spec.name,
                expected_type=expected_type,
                candidate_id=cand["candidate_id"],
                candidate_name=cand["candidate_name"],
                candidate_type=cand["candidate_type"],
                matched_description=cand["matched_description"],
                similarity_score=cand["similarity_score"],
                distance_score=cand["distance_score"],
                matching_method="ENTITY_DESCRIPTION_SEARCH",
                type_compatible=cand["type_compatible"],
                confidence_score=cand["confidence_score"],
                ranking=rank_idx,
                is_ambiguous=is_ambiguous,
                candidate_ids=all_tied_ids,
                status=status,
                reason=reason,
            )
            accepted_objects.append(cand_obj)

            decisions.append(
                ResolutionDecision(
                    seed_name=seed_spec.name,
                    expected_type=expected_type,
                    candidate_id=cand["candidate_id"],
                    candidate_name=cand["candidate_name"],
                    candidate_type=cand["candidate_type"],
                    matching_method="ENTITY_DESCRIPTION_SEARCH",
                    type_compatible=cand["type_compatible"],
                    confidence_score=cand["confidence_score"],
                    similarity_score=cand["similarity_score"],
                    ranking=rank_idx,
                    status=status,
                    reason=reason,
                )
            )

        # Record rejected candidates
        for cand in active_group:
            if cand["candidate_id"] not in selected_ids:
                decisions.append(
                    ResolutionDecision(
                        seed_name=seed_spec.name,
                        expected_type=expected_type,
                        candidate_id=cand["candidate_id"],
                        candidate_name=cand["candidate_name"],
                        candidate_type=cand["candidate_type"],
                        matching_method="ENTITY_DESCRIPTION_SEARCH",
                        type_compatible=cand["type_compatible"],
                        confidence_score=cand["confidence_score"],
                        similarity_score=cand["similarity_score"],
                        ranking=len(selected_ids) + 1,
                        status="REJECTED",
                        reason=f"Lower similarity than top candidate ({cand['similarity_score']:.3f} vs {top_sim:.3f})",
                    )
                )

        # If compatible group existed, mark conflicting candidates as rejected
        if compatible_group:
            for cand in scored_candidates:
                if not cand["type_compatible"]:
                    decisions.append(
                        ResolutionDecision(
                            seed_name=seed_spec.name,
                            expected_type=expected_type,
                            candidate_id=cand["candidate_id"],
                            candidate_name=cand["candidate_name"],
                            candidate_type=cand["candidate_type"],
                            matching_method="ENTITY_DESCRIPTION_SEARCH",
                            type_compatible=False,
                            confidence_score=cand["confidence_score"],
                            similarity_score=cand["similarity_score"],
                            ranking=len(selected_ids) + 1,
                            status="REJECTED",
                            reason=f"Superseded by type-compatible entity description candidate (expected: {expected_type}, candidate: {cand['candidate_type']})",
                        )
                    )

        return accepted_objects, is_ambiguous, decisions
