"""Relationship extraction implementation with quote verification, polarity filtering, and ontology validation."""

from __future__ import annotations
from collections import Counter
import json
import logging
import hashlib
from pathlib import Path
import re
from typing import List, Optional, Set, Union
from src.core.configuration import ValidationConfig
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
5. evidence_quote MUST be a single contiguous verbatim substring copied directly from the text that explicitly mentions or directly relates both entities.
6. polarity must be "POSITIVE", "NEGATIVE", or "UNCERTAIN". Do not extract relationships for negative, hypothetical, or speculative statements as positive facts.
7. CRITICAL DIRECTIONALITY: Each relationship MUST strictly follow the direction defined in the constraints: source_mention_id MUST have an allowed source type, and target_mention_id MUST have an allowed target type. Never invert the direction (e.g. for DIRECTED, the person director is source and the film is target; for ACTED_IN, the person actor is source and the film is target; for AUTHORED, the person author is source and the work/film/book is target; for SUPERVISES, the person leader is source and the person or project is target).
8. Specificity: Choose the most specific allowed relationship type for the facts described (e.g., if one person leads, directs, or supervises another person or project, choose SUPERVISES rather than general WORKS_ON or WORKS_AT).
9. Return ONLY a valid JSON object matching this schema:

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

PRONOUN_START_REGEX = re.compile(
    r"^\s*(he|she|it|they|his|her|its|their|him|them)\b", re.IGNORECASE
)


def build_cues_regex(cues: List[str]) -> re.Pattern:
    patterns = []
    for cue in cues:
        c = cue.strip()
        if not c:
            continue
        if c.startswith("n't"):
            patterns.append(r"n't\b")
        else:
            patterns.append(r"\b" + re.escape(c) + r"\b")
    if not patterns:
        return re.compile(r"(?!)")
    return re.compile("|".join(patterns), re.IGNORECASE)


NEGATION_REGEX = re.compile(
    r"\b(not|n't|never|no longer|neither|nor|without|cannot|denies|denied|rejects|rejected|fails to|failed to)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def strip_honorific(name: str, honorifics: List[str]) -> str:
    """Strip leading honorific tokens from a name string."""
    cleaned = normalize_whitespace(name)
    tokens = cleaned.split()
    h_set = {h.lower().rstrip(".") for h in honorifics}
    while tokens and tokens[0].lower().rstrip(".") in h_set:
        tokens.pop(0)
    return " ".join(tokens)


def get_person_surname(name: str, honorifics: List[str]) -> Optional[str]:
    """Extract person surname (final meaningful token) after stripping honorifics and suffixes."""
    stripped = strip_honorific(name, honorifics)
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "ph.d.", "phd"}
    while tokens and tokens[-1].lower() in suffixes:
        tokens.pop()
    if tokens:
        last = re.sub(r"^[^\w]+|[^\w]+$", "", tokens[-1])
        if len(last) >= 2:
            return last.lower()
    return None


class OllamaRelationshipExtractor(RelationshipExtractor):
    """Extracts candidate relationships between entities and applies strict three-stage validation."""

    def __init__(
        self,
        ontology: DomainOntology,
        llm_client: OllamaLLMClient,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/relationship_extraction.txt",
        prompt_template: Optional[str] = None,
        validation_config: Optional[ValidationConfig] = None,
    ) -> None:
        self.ontology = ontology
        self.llm_client = llm_client
        self.validation_config = validation_config or ValidationConfig()
        self.negation_regex = build_cues_regex(self.validation_config.negation_cues)
        self.uncertainty_regex = build_cues_regex(self.validation_config.uncertainty_cues)
        self.honorifics = [
            h.lower().strip()
            for h in getattr(
                self.validation_config,
                "honorifics",
                ["prof", "prof.", "professor", "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms."],
            )
            if h.strip()
        ]
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

        # Collect surnames for person-like mentions in this chunk to determine uniqueness
        person_surnames: List[str] = []
        for m in mentions:
            if self.ontology.is_person_like_type(m.entity_type):
                sname = get_person_surname(m.normalized_name, self.honorifics) or get_person_surname(m.surface_name, self.honorifics)
                if sname:
                    person_surnames.append(sname)
        surname_counts = Counter(person_surnames)

        def resolve_mention(key: str) -> Optional[EntityMention]:
            k = key.strip()
            if k in mentions_by_id:
                return mentions_by_id[k]
            k_lower = k.lower()
            if k_lower in mentions_by_surface:
                return mentions_by_surface[k_lower]
            if k_lower in mentions_by_norm:
                return mentions_by_norm[k_lower]
            stripped = strip_honorific(k_lower, self.honorifics).lower()
            if stripped in mentions_by_surface:
                return mentions_by_surface[stripped]
            if stripped in mentions_by_norm:
                return mentions_by_norm[stripped]
            sname = get_person_surname(k_lower, self.honorifics)
            if sname and surname_counts.get(sname, 0) == 1:
                for m in mentions:
                    if self.ontology.is_person_like_type(m.entity_type):
                        m_sname = get_person_surname(m.normalized_name, self.honorifics) or get_person_surname(m.surface_name, self.honorifics)
                        if m_sname == sname:
                            return m
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
            repaired_quote_flag = False
            quote_starts_with_pronoun_flag = False

            # Helper to get all surface forms for an entity mention
            def get_mention_surface_forms(m: EntityMention) -> Set[str]:
                forms = {
                    normalize_whitespace(m.surface_name).lower(),
                    normalize_whitespace(m.normalized_name).lower(),
                }
                # Also include static aliases from ontology if available
                aliases = (
                    self.ontology.static_aliases.get(m.surface_name, [])
                    + self.ontology.static_aliases.get(m.normalized_name, [])
                )
                for a in aliases:
                    forms.add(normalize_whitespace(a).lower())

                # Honorific-stripped forms
                stripped_surface = strip_honorific(m.surface_name, self.honorifics).lower()
                if stripped_surface:
                    forms.add(stripped_surface)
                stripped_norm = strip_honorific(m.normalized_name, self.honorifics).lower()
                if stripped_norm:
                    forms.add(stripped_norm)

                # For person-like mentions: accept unique chunk surname
                if self.ontology.is_person_like_type(m.entity_type):
                    sname = get_person_surname(m.normalized_name, self.honorifics) or get_person_surname(m.surface_name, self.honorifics)
                    if sname and surname_counts.get(sname, 0) == 1:
                        forms.add(sname)

                return {f for f in forms if f}

            src_forms = get_mention_surface_forms(src)
            tgt_forms = get_mention_surface_forms(tgt)

            def contains_form(text_norm: str, forms: Set[str]) -> bool:
                for f in forms:
                    pattern = r"(?<!\w)" + re.escape(f.strip()) + r"(?!\w)"
                    if re.search(pattern, text_norm):
                        return True
                return False

            # --- Validation Stage 1: Quote Verification & Repair ---
            quote_verified = bool(quote_norm and quote_norm in normalized_chunk_text)
            if not quote_verified and self.validation_config.quote_repair_enabled:
                # A(iii) Quote repair:
                # search sentences of source chunk containing both endpoints' surfaces.
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if s.strip()]
                matching_sentences = [
                    s for s in sentences
                    if contains_form(normalize_whitespace(s).lower(), src_forms)
                    and contains_form(normalize_whitespace(s).lower(), tgt_forms)
                ]
                if len(matching_sentences) == 1:
                    quote = matching_sentences[0]
                    quote_norm = normalize_whitespace(quote).lower()
                    quote_verified = True
                    repaired_quote_flag = True

            if not quote_verified:
                validation_status = ValidationStatus.REJECTED_QUOTE_MISMATCH
            else:
                # --- Validation Stage 1b: Endpoint-in-evidence Check (A(i)) ---
                endpoints_in_evidence = True
                if self.validation_config.endpoint_in_evidence_enabled:
                    has_src = contains_form(quote_norm, src_forms)
                    has_tgt = contains_form(quote_norm, tgt_forms)
                    if not (has_src and has_tgt):
                        endpoints_in_evidence = False

                if not endpoints_in_evidence:
                    validation_status = ValidationStatus.REJECTED_ENDPOINT_NOT_IN_EVIDENCE
                    if PRONOUN_START_REGEX.search(quote_norm):
                        quote_starts_with_pronoun_flag = True
                # --- Validation Stage 2: Negation & Uncertainty Filtering (A(ii)) ---
                elif polarity == Polarity.UNCERTAIN or self.uncertainty_regex.search(quote_norm):
                    validation_status = ValidationStatus.REJECTED_UNCERTAIN
                elif polarity == Polarity.NEGATIVE or self.negation_regex.search(quote_norm):
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
                repaired_quote=repaired_quote_flag,
                quote_starts_with_pronoun=quote_starts_with_pronoun_flag,
            )
            relationships.append(relationship)
            rel_idx += 1

        return relationships
