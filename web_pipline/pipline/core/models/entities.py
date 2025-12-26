"""
Entity Models for Query Processing
===================================

Data structures for query intent classification and entity extraction.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class QueryIntent(str, Enum):
    """Classification of query intent types."""

    FACTUAL = "factual"           # Direct fact-based questions
    PROCEDURAL = "procedural"     # How-to, process questions
    COMPARISON = "comparison"     # Compare X vs Y questions
    EXPLORATORY = "exploratory"   # Open-ended exploration

    @classmethod
    def from_string(cls, value: str) -> "QueryIntent":
        """Create intent from string, defaulting to FACTUAL."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.FACTUAL


@dataclass
class AcademicEntity:
    """
    Represents an extracted academic entity from a query.

    Attributes:
        text: Original text from the query
        label: Entity type (PROGRAM, COURSE, LECTURER, etc.)
        normalized: Normalized/canonical form of the entity
    """
    text: str
    label: str
    normalized: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "normalized": self.normalized,
        }


@dataclass
class QueryAnalysis:
    """
    Complete analysis of a user query.

    Contains cleaned query, intent classification, extracted entities,
    and metadata filters for retrieval.
    """
    cleaned_query: str
    intent: QueryIntent
    entities: List[AcademicEntity] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    sub_queries: List[str] = field(default_factory=list)
    programs: List[str] = field(default_factory=list)
    page_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cleaned_query": self.cleaned_query,
            "intent": self.intent.value if isinstance(self.intent, QueryIntent) else str(self.intent),
            "entities": [e.to_dict() for e in self.entities],
            "metadata_filters": self.metadata_filters,
            "sub_queries": self.sub_queries,
            "programs": self.programs,
            "page_types": self.page_types,
        }

    def get_track(self) -> str:
        """Get the primary track/program or 'GENERAL'."""
        if self.programs:
            return self.programs[0]
        return "GENERAL"

    def get_entity_dict(self) -> Dict[str, List[str]]:
        """Get entities grouped by label."""
        result: Dict[str, List[str]] = {
            "course_codes": [],
            "lecturers": [],
            "course_titles": [],
        }
        for entity in self.entities:
            label_key = entity.label.lower() + "s"
            if label_key in result:
                result[label_key].append(entity.normalized)
        return result
