"""Relationship extraction implementation with quote verification, polarity filtering, and ontology validation."""

from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
import re
from typing import List, Optional, Union
from src.core.interfaces import RelationshipExtractor
from src.core.models import Chunk, EntityMention, Polarity, Relationship, ValidationStatus
from src.core.ontology import DomainOntology
from src.extraction.schemas import OllamaLLMClient, RawRelationshipExtractionOutput

logger = logging.getLogger(__name__)

DEFAULT_RELATION_EXTRACTION_PROMPT = """You are an expert knowledge graph extraction system. Your task is to extract factual relationships between the provided entity mentions from the text below, returning valid JSON.

ALLOWED RELATIONSHIP TYPES AND CONSTRAINTS:
{relationship_types}

AVAILABLE ENTITY MENTIONS:
{mentions_json}

TEXT:
\"\"\"
{chunk_text}
\"\"\"

INSTRUCTIONS:
1. Identify all direct relationships asserted in the text between the available entity mentions.
2. The source and target entities must be distinct (no self-loops).
3. Use the exact mention ID or surface name for source_mention_id and target_mention_id.
4. relation_type must strictly be one of the allowed relationship types.
5. evidence_quote MUST be an exact verbatim substring copied directly from the text supporting this relationship.
6. polarity must be "POSITIVE", "NEGATIVE", or "UNCERTAIN".
7. CRITICAL DIRECTIONALITY: Each relationship MUST strictly follow the direction defined in the constraints: source_mention_id MUST have an allowed source type, and target_mention_id MUST have an allowed target type. Never invert the direction (e.g. for DIRECTED, the person director is source and the film is target; for ACTED_IN, the person actor is source and the film is target; for AUTHORED, the person author is source and the work/film/book is target).
8. Return ONLY a valid JSON object matching this schema:

{
  "relationships": [
    {
      "source_mention_id": "mention_id",
      "target_mention_id": "mention_id",
      "relation_type": "ALLOWED_RELATION_TYPE",
      "evidence_quote": "exact verbatim text from the chunk",
      "polarity": "POSITIVE",
      "confidence": 1.0
    }
  ]
}
"""

NEGATION_REGEX = re.compile(
    r"\b(not|n't|never|no longer|neither|nor|cannot|denies|denied|rejects|rejected|fails to|failed to)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


class OllamaRelationshipExtractor(RelationshipExtractor):
    """Extracts candidate relationships between entities and applies strict three-stage validation."""

    def __init__(
        self,
        ontology: DomainOntology,
        llm_client: OllamaLLMClient,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/relationship_extraction.txt",
        prompt_template: Optional[str] = None,
    ) -> None:
        self.ontology = ontology
        self.llm_client = llm_client
        self._raw_cache = {}

        if prompt_template is not None:
            self.prompt_template = prompt_template
        elif prompt_template_path is not None and Path(prompt_template_path).exists():
            self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        else:
            self.prompt_template = DEFAULT_RELATION_EXTRACTION_PROMPT

    def extract_relationships(
        self, chunk: Chunk, mentions: List[EntityMention]
    ) -> List[Relationship]:
        """Extract relationships between mentions in a Chunk, validating quote, polarity, and ontology."""
        if len(mentions) < 2 or not chunk.text.strip():
            return []

        formatted_relations = "\n".join(
            f"- {name}: {c.description} (Allowed: {c.allowed_source_types or 'ANY'} -> {c.allowed_target_types or 'ANY'})"
            for name, c in self.ontology.relationship_types.items()
        )

        mentions_data = [
            {
                "mention_id": m.mention_id,
                "surface_name": m.surface_name,
                "entity_type": m.entity_type,
                "description": m.description,
            }
            for m in mentions
        ]
        mentions_json_str = json.dumps(mentions_data, indent=2)

        prompt = (
            self.prompt_template.replace("{relationship_types}", formatted_relations)
            .replace("{mentions_json}", mentions_json_str)
            .replace("{chunk_text}", chunk.text)
        )

        cache_basis = self.ontology.model_dump_json() + chunk.text + "|" + "|".join(
            f"{m.normalized_name}:{m.entity_type}" for m in mentions
        )
        cache_key = hashlib.sha256(cache_basis.encode("utf-8")).hexdigest()
        raw_output = self._raw_cache.get(cache_key)
        if raw_output is None:
            generated = self.llm_client.generate_json(prompt, RawRelationshipExtractionOutput)
            id_to_surface = {m.mention_id: m.surface_name for m in mentions}
            cache_data = generated.model_dump()
            for item in cache_data["relationships"]:
                item["source_mention_id"] = id_to_surface.get(item["source_mention_id"], item["source_mention_id"])
                item["target_mention_id"] = id_to_surface.get(item["target_mention_id"], item["target_mention_id"])
            raw_output = RawRelationshipExtractionOutput.model_validate(cache_data)
            self._raw_cache[cache_key] = raw_output

        # Lookup tables for resolving source and target
        mentions_by_id = {m.mention_id: m for m in mentions}
        mentions_by_surface = {m.surface_name.lower(): m for m in mentions}
        mentions_by_norm = {m.normalized_name: m for m in mentions}

        def resolve_mention(key: str) -> Optional[EntityMention]:
            k = key.strip()
            if k in mentions_by_id:
                return mentions_by_id[k]
            k_lower = k.lower()
            if k_lower in mentions_by_surface:
                return mentions_by_surface[k_lower]
            if k_lower in mentions_by_norm:
                return mentions_by_norm[k_lower]
            return None

        normalized_chunk_text = normalize_whitespace(chunk.text).lower()
        relationships: List[Relationship] = []
        rel_idx = 1

        for raw_item in raw_output.relationships:
            src = resolve_mention(raw_item.source_mention_id)
            tgt = resolve_mention(raw_item.target_mention_id)

            if src is None or tgt is None:
                logger.warning(
                    "Skipping relationship with unresolvable entities: src='%s', tgt='%s'",
                    raw_item.source_mention_id,
                    raw_item.target_mention_id,
                )
                continue

            # Skip self-loops
            if src.mention_id == tgt.mention_id:
                logger.warning("Skipping self-loop relationship on mention %s", src.mention_id)
                continue

            quote = raw_item.evidence_quote.strip()
            quote_norm = normalize_whitespace(quote).lower()
            polarity = raw_item.polarity

            # --- Validation Stage 1: Quote Verification ---
            if not quote_norm or quote_norm not in normalized_chunk_text:
                validation_status = ValidationStatus.REJECTED_QUOTE_MISMATCH
            # --- Validation Stage 2: Negation & Uncertainty Filtering ---
            elif polarity != Polarity.POSITIVE or NEGATION_REGEX.search(quote_norm):
                validation_status = ValidationStatus.REJECTED_NEGATION
            # --- Validation Stage 3: Ontology Validation ---
            else:
                ok, _ = self.ontology.validate_triple_types(
                    src.entity_type, raw_item.relation_type, tgt.entity_type
                )
                if not ok:
                    validation_status = ValidationStatus.REJECTED_ONTOLOGY
                else:
                    validation_status = ValidationStatus.VALIDATED

            rel_id = f"{chunk.chunk_id}_R{rel_idx:03d}"
            relationship = Relationship(
                kb_id=chunk.kb_id,
                relationship_id=rel_id,
                source_entity_id=src.mention_id,
                target_entity_id=tgt.mention_id,
                relation_type=raw_item.relation_type,
                source_chunk_ids=[chunk.chunk_id],
                source_doc_ids=[chunk.doc_id],
                evidence_quotes=[quote if quote else chunk.text],
                support_count=1,
                polarity=polarity,
                validation_status=validation_status,
                uncalibrated_confidence=raw_item.confidence,
            )
            relationships.append(relationship)
            rel_idx += 1

        return relationships
