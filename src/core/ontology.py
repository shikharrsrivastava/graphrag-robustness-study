"""Domain ontology definitions and validation rules."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RelationConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    allowed_source_types: List[str] = Field(default_factory=list)
    allowed_target_types: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")

    @field_validator("allowed_source_types", "allowed_target_types")
    @classmethod
    def normalize_type_list(cls, v: List[str]) -> List[str]:
        return [item.strip().upper() for item in v]


class DomainOntology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_name: str = Field(..., min_length=1)
    version: str = Field(default="1.0.0")
    entity_types: Dict[str, str] = Field(
        ..., min_length=1, description="Mapping of UPPERCASE entity type to description"
    )
    relationship_types: Dict[str, RelationConstraint] = Field(
        ..., min_length=1, description="Mapping of UPPERCASE relation type to constraint"
    )
    static_aliases: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Optional domain-specific canonical name -> list of known aliases",
    )

    @field_validator("entity_types")
    @classmethod
    def normalize_entity_keys(cls, v: Dict[str, str]) -> Dict[str, str]:
        return {k.strip().upper(): desc for k, desc in v.items()}

    def is_valid_entity_type(self, entity_type: str) -> bool:
        return entity_type.strip().upper() in self.entity_types

    def is_valid_relation_type(self, relation_type: str) -> bool:
        norm = relation_type.strip().upper().replace(" ", "_")
        return norm in self.relationship_types

    def validate_triple_types(
        self, source_type: str, relation_type: str, target_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate entity types and directional endpoint constraints for a relationship."""
        s_type = source_type.strip().upper()
        r_type = relation_type.strip().upper().replace(" ", "_")
        t_type = target_type.strip().upper()

        if not self.is_valid_entity_type(s_type):
            return False, f"Unknown source entity type: {s_type}"
        if not self.is_valid_entity_type(t_type):
            return False, f"Unknown target entity type: {t_type}"
        if not self.is_valid_relation_type(r_type):
            return False, f"Unknown relationship type: {r_type}"

        constraint = self.relationship_types[r_type]
        if constraint.allowed_source_types and s_type not in constraint.allowed_source_types:
            return (
                False,
                f"Source type '{s_type}' not permitted for '{r_type}' (allowed: {constraint.allowed_source_types})",
            )
        if constraint.allowed_target_types and t_type not in constraint.allowed_target_types:
            return (
                False,
                f"Target type '{t_type}' not permitted for '{r_type}' (allowed: {constraint.allowed_target_types})",
            )
        return True, None
