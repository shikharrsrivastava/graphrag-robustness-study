"""Entity extraction implementation using Ollama and domain ontology validation."""

from __future__ import annotations
import logging
import hashlib
from pathlib import Path
import re
import string
from typing import List, Optional, Union
from src.core.interfaces import EntityExtractor
from src.core.models import Chunk, EntityMention
from src.core.ontology import DomainOntology
from src.extraction.schemas import OllamaLLMClient, RawEntityExtractionOutput

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_EXTRACTION_PROMPT = """You are an expert information extraction system. Your task is to extract entity mentions from the text below strictly adhering to the allowed entity types and returning valid JSON.

ALLOWED ENTITY TYPES AND DESCRIPTIONS:
{entity_types}

TEXT:
\"\"\"
{chunk_text}
\"\"\"

INSTRUCTIONS:
1. Identify all salient entity mentions in the text that strictly match the allowed entity types.
2. Extract every named entity, including named labs, groups, institutes, and projects, as its own mention even when it appears only inside a descriptive phrase about another entity.
3. Do not extract entity types not listed above.
4. surface_name must be the exact text span from the text.
5. description should be a concise summary of the entity's role or context in this text.
6. Return ONLY a valid JSON object matching this schema:

{
  "entities": [
    {
      "surface_name": "Entity Name",
      "entity_type": "ALLOWED_TYPE",
      "description": "Brief description"
    }
  ]
}
"""


def normalize_entity_name(name: str) -> str:
    """Normalize surface name: collapse internal whitespace, strip leading/trailing punctuation, lowercase."""
    collapsed = re.sub(r"\s+", " ", name).strip()
    stripped = collapsed.strip(string.punctuation + " \t\r\n")
    return stripped.lower()


class OllamaEntityExtractor(EntityExtractor):
    """Extracts entity mentions from chunks using an LLM and validates against DomainOntology."""

    def __init__(
        self,
        ontology: DomainOntology,
        llm_client: OllamaLLMClient,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/entity_extraction.txt",
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
            self.prompt_template = DEFAULT_ENTITY_EXTRACTION_PROMPT

    def extract_entities(self, chunk: Chunk) -> List[EntityMention]:
        """Extract valid entity mentions from a Chunk matching the active ontology."""
        if not chunk.text.strip():
            return []

        formatted_types = "\n".join(
            f"- {etype}: {desc}" for etype, desc in self.ontology.entity_types.items()
        )
        prompt = (
            self.prompt_template.replace("{entity_types}", formatted_types)
            .replace("{chunk_text}", chunk.text)
        )

        cache_key = hashlib.sha256((self.ontology.model_dump_json() + chunk.text).encode("utf-8")).hexdigest()
        raw_output = self._raw_cache.get(cache_key)
        if raw_output is None:
            raw_output = self.llm_client.generate_json(prompt, RawEntityExtractionOutput)
            self._raw_cache[cache_key] = raw_output

        mentions: List[EntityMention] = []
        mention_idx = 1

        for raw_item in raw_output.entities:
            surface = raw_item.surface_name.strip()
            raw_type = raw_item.entity_type.strip().upper()

            if not surface:
                continue

            # Validate against domain ontology
            if not self.ontology.is_valid_entity_type(raw_type):
                logger.warning(
                    "Skipping extracted entity '%s' with unrecognized ontology type '%s'.",
                    surface,
                    raw_type,
                )
                continue

            normalized = normalize_entity_name(surface)
            if not normalized:
                normalized = surface.lower()

            # Find character offsets in chunk.text
            char_start, char_end = self._find_offsets(chunk.text, surface, normalized)

            mention_id = f"{chunk.chunk_id}_M{mention_idx:03d}"
            mention = EntityMention(
                kb_id=chunk.kb_id,
                mention_id=mention_id,
                surface_name=surface,
                normalized_name=normalized,
                entity_type=raw_type,
                description=raw_item.description.strip(),
                source_chunk_id=chunk.chunk_id,
                source_doc_id=chunk.doc_id,
                page_numbers=list(chunk.page_numbers),
                char_start=char_start,
                char_end=char_end,
            )
            mentions.append(mention)
            mention_idx += 1

        return mentions

    def _find_offsets(
        self, text: str, surface: str, normalized: str
    ) -> tuple[Optional[int], Optional[int]]:
        """Locate start and end character offsets of entity mention within chunk text."""
        # 1. Exact match
        pos = text.find(surface)
        if pos != -1:
            return pos, pos + len(surface)

        # 2. Case-insensitive exact match
        text_lower = text.lower()
        pos = text_lower.find(surface.lower())
        if pos != -1:
            return pos, pos + len(surface)

        # 3. Case-insensitive normalized match
        pos = text_lower.find(normalized)
        if pos != -1:
            return pos, pos + len(normalized)

        return None, None
