"""Domain ontology definitions and validation rules."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# Default entity types treated as person-like when an ontology does not declare its own list.
DEFAULT_PERSON_LIKE_TYPES: Tuple[str, ...] = ("PERSON",)


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
    generic_descriptors: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Optional entity type -> list of generic descriptors (e.g. PROJECT: ['project', 'program'])",
    )
    person_like_types: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PERSON_LIKE_TYPES),
        description=(
            "Entity type names treated as person-like for name resolution (surname, initial, honorific and "
            "generational-suffix rules). Defaults to DEFAULT_PERSON_LIKE_TYPES."
        ),
    )

    @field_validator("entity_types")
    @classmethod
    def normalize_entity_keys(cls, v: Dict[str, str]) -> Dict[str, str]:
        return {k.strip().upper(): desc for k, desc in v.items()}

    @field_validator("person_like_types")
    @classmethod
    def normalize_person_like_types(cls, v: List[str]) -> List[str]:
        return [item.strip().upper() for item in v]

    @model_validator(mode="after")
    def validate_person_like_types_exist(self) -> "DomainOntology":
        """A custom person_like_types list must only name declared entity types.

        The built-in default is exempt so ontologies without a person type keep loading unchanged.
        """
        if list(self.person_like_types) != list(DEFAULT_PERSON_LIKE_TYPES):
            unknown = [t for t in self.person_like_types if t not in self.entity_types]
            if unknown:
                raise ValueError(
                    f"person_like_types contains types not declared in entity_types: {unknown}"
                )
        return self

    def is_person_like_type(self, entity_type: Optional[str]) -> bool:
        if not entity_type:
            return False
        return entity_type.strip().upper() in set(self.person_like_types)

    def entity_type_rank(self, entity_type: str) -> int:
        """Position of an entity type in ontology declaration order (unknown types sort last)."""
        keys = list(self.entity_types.keys())
        t = entity_type.strip().upper()
        return keys.index(t) if t in keys else len(keys)

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
