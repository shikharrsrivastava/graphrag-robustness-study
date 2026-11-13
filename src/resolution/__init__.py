"""Resolution package providing entity resolution and graph remapping."""

from src.resolution.entity_resolver import (
    ConservativeEntityResolver,
    EntityResolutionVerdict,
    strip_honorifics,
)

__all__ = [
    "ConservativeEntityResolver",
    "EntityResolutionVerdict",
    "strip_honorifics",
]
