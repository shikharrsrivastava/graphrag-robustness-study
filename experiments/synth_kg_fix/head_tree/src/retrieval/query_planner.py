"""Ollama-backed Query Planner parsing natural-language questions into validated QueryPlans."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Union
from src.core.models import QueryIntent, QueryPlan
from src.core.ontology import DomainOntology
from src.extraction.schemas import OllamaLLMClient

logger = logging.getLogger(__name__)

DEFAULT_QUERY_PLANNING_PROMPT = """You are an expert query planning assistant for a Hybrid GraphRAG knowledge base system.
Your job is to analyze the user's natural language question and construct a structured retrieval plan.

DOMAIN ONTOLOGY CONSTRAINTS:
Allowed Entity Types:
{entity_types}

Allowed Relationship Types:
{relationship_types}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Determine the query intent:
   - "LOCAL": Questions focusing on specific named entities, their direct properties, or multi-hop relationship paths connecting specific entities.
   - "GLOBAL": Broad summary, thematic questions, or high-level overviews across the entire knowledge base without specific seed entities.
   - "MIXED": Questions combining high-level domain themes with specific entities or comparing specific entities within broad topics.

2. Extract seed entities (for LOCAL and MIXED queries):
   - Extract candidate seed entities mentioned or implied in the question.
   - For each seed entity, extract its name and assign an expected_type from the allowed entity types if applicable.

3. Identify target entity type and relationship path:
   - target_type: The expected entity type of the answer/target entity if known, or null.
   - relationship_path: An ordered list of relationship types from the allowed relationship types that need to be traversed from the seed entities to find the answer. If no specific relationship path is required, provide an empty list []. Only include relationship types present in the allowed list! Do not confuse nationality or citizenship (NATIONALITY) with birthplace (BORN_IN).

4. Clarification check:
   - Set requires_clarification to true ONLY if the question is genuinely ambiguous, refers to an unclear entity where disambiguation is impossible, or lacks vital information needed to initiate retrieval.
   - Set clarification_reason to explain why clarification is required, or null if requires_clarification is false.

Return ONLY a valid JSON object matching this schema:
{
  "question": "{question}",
  "intent": "LOCAL",
  "seed_entities": [
    {
      "name": "Entity Name",
      "expected_type": "TYPE"
    }
  ],
  "target_type": "TARGET_TYPE",
  "relationship_path": ["RELATION_TYPE"],
  "requires_clarification": false,
  "clarification_reason": null
}
"""


class OllamaQueryPlanner:
    """Plans queries into validated QueryPlans with ontology constraint validation."""

    def __init__(
        self,
        ontology: DomainOntology,
        llm_client: OllamaLLMClient,
        prompt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.ontology = ontology
        self.llm_client = llm_client

        if prompt_path is not None and Path(prompt_path).exists():
            self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        else:
            default_path = Path("prompts/query_planning.txt")
            if default_path.exists():
                self.prompt_template = default_path.read_text(encoding="utf-8")
            else:
                self.prompt_template = DEFAULT_QUERY_PLANNING_PROMPT

    def plan(self, question: str) -> QueryPlan:
        """Convert a user's question into a validated QueryPlan."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        clean_question = question.strip()
        entity_types_str = ", ".join(sorted(self.ontology.entity_types.keys()))
        rel_types_str = ", ".join(sorted(self.ontology.relationship_types.keys()))

        prompt = (
            self.prompt_template.replace("{entity_types}", entity_types_str)
            .replace("{relationship_types}", rel_types_str)
            .replace("{question}", clean_question)
        )

        plan = self.llm_client.generate_json(prompt, QueryPlan)

        # Validate question preservation
        plan.question = clean_question

        # Validate & filter relationship_path against domain ontology
        valid_path = []
        invalid_types = []
        for rel in plan.relationship_path:
            clean_rel = rel.strip().upper()
            if self.ontology.is_valid_relation_type(clean_rel):
                valid_path.append(clean_rel)
            else:
                invalid_types.append(rel)

        if invalid_types:
            logger.warning(
                "QueryPlan contained invalid relation types %s not permitted by ontology; filtered out.",
                invalid_types,
            )
            # Invalid graph hints are diagnostic only. Retrieval can still use
            # resolved seeds, vector evidence, and communities.
            plan.invalid_relationship_types = invalid_types

        plan.relationship_path = valid_path

        # Validate seed entity types if provided
        for seed in plan.seed_entities:
            if seed.expected_type:
                clean_type = seed.expected_type.strip().upper()
                if not self.ontology.is_valid_entity_type(clean_type):
                    logger.warning(
                        "Seed entity '%s' had invalid expected_type '%s'; resetting to None",
                        seed.name,
                        seed.expected_type,
                    )
                    seed.expected_type = None
                else:
                    seed.expected_type = clean_type

        # Validate target_type if provided
        if plan.target_type:
            clean_target = plan.target_type.strip().upper()
            if not self.ontology.is_valid_entity_type(clean_target):
                logger.warning(
                    "QueryPlan had invalid target_type '%s'; resetting to None",
                    plan.target_type,
                )
                plan.target_type = None
            else:
                plan.target_type = clean_target

        return plan
