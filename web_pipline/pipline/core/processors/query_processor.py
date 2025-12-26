"""
Query Processor Module
======================

Query analysis, intent classification, and entity extraction.
Combines enhanced LLM-based processing with rule-based fallbacks.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.models.entities import QueryIntent, AcademicEntity, QueryAnalysis

logger = logging.getLogger(__name__)


class QueryProcessorBase(ABC):
    """Abstract base class for query processors."""

    @abstractmethod
    def process_query(self, query: str) -> QueryAnalysis:
        """Process a query and return analysis results."""
        pass


class BaselineQueryProcessor(QueryProcessorBase):
    """
    Minimal fallback query processor using rule-based analysis.

    Used when the enhanced LLM processor is unavailable.
    Extracts naive signals: track, course code, entities.
    """

    # Track detection patterns
    TRACK_PATTERNS = {
        "DS": ["data science", "ds&ai", "dsai", "data science & ai", "ds "],
        "SE": ["software engineering", "se "],
        "CN": ["computer networks", "cn "],
    }

    # Course code pattern
    COURSE_CODE_PATTERN = r"\b[0-9]{4}[A-Z]{3,}\b"

    # Intent keywords
    PROCEDURAL_KEYWORDS = ["how to", "apply", "deadline", "admission", "requirements"]
    COMPARISON_KEYWORDS = ["compare", "difference", "vs", "versus", "or"]

    def process_query(self, query: str) -> QueryAnalysis:
        """
        Process query using rule-based analysis.

        Args:
            query: User's input question

        Returns:
            QueryAnalysis with extracted information
        """
        query_lower = query.lower()

        # Detect track/program
        programs = self._detect_programs(query_lower)

        # Extract course codes
        course_codes = re.findall(self.COURSE_CODE_PATTERN, query)

        # Classify intent
        intent = self._classify_intent(query_lower)

        # Build entities
        entities = []
        for code in course_codes:
            entities.append(AcademicEntity(
                text=code,
                label="COURSE_CODE",
                normalized=code.upper()
            ))

        for program in programs:
            entities.append(AcademicEntity(
                text=program,
                label="PROGRAM",
                normalized=program
            ))

        return QueryAnalysis(
            cleaned_query=query.strip(),
            intent=intent,
            entities=entities,
            programs=programs,
            metadata_filters=self._build_filters(programs, course_codes),
            page_types=self._infer_page_types(query_lower, intent),
        )

    def _detect_programs(self, query_lower: str) -> List[str]:
        """Detect program mentions in query."""
        programs = []
        for track, patterns in self.TRACK_PATTERNS.items():
            if any(p in query_lower for p in patterns):
                programs.append(track)
        return programs

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify query intent using keywords."""
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON
        if any(kw in query_lower for kw in self.PROCEDURAL_KEYWORDS):
            return QueryIntent.PROCEDURAL
        return QueryIntent.FACTUAL

    def _build_filters(self, programs: List[str], course_codes: List[str]) -> Dict[str, Any]:
        """Build metadata filters for retrieval."""
        conditions = []

        if programs:
            if len(programs) == 1:
                conditions.append({"program": {"$eq": programs[0]}})
            else:
                conditions.append({"program": {"$in": programs}})

        if course_codes:
            conditions.append({"course_code": {"$in": course_codes}})

        if not conditions:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _infer_page_types(self, query_lower: str, intent: QueryIntent) -> List[str]:
        """Infer relevant page types from query."""
        page_types = []

        if any(kw in query_lower for kw in ["admission", "requirements", "apply"]):
            page_types.append("admission-and-enrolment")
        if any(kw in query_lower for kw in ["course", "curriculum", "module"]):
            page_types.append("programme")
        if any(kw in query_lower for kw in ["contact", "email", "phone"]):
            page_types.append("contact")
        if any(kw in query_lower for kw in ["fee", "tuition", "cost"]):
            page_types.append("admission-and-enrolment")

        return page_types if page_types else ["programme", "admission-and-enrolment"]


class QueryProcessor(QueryProcessorBase):
    """
    Main query processor that wraps enhanced or baseline processing.

    Attempts to use the enhanced LLM-based processor if available,
    falling back to baseline rule-based processing.
    """

    def __init__(self, ollama_client: Optional[Any] = None):
        """
        Initialize query processor.

        Args:
            ollama_client: Optional Ollama client for LLM-based processing
        """
        self.ollama_client = ollama_client
        self.baseline = BaselineQueryProcessor()
        self.enhanced_processor = None

        # Try to import the enhanced processor
        try:
            from ...enhanced_query_processor_ollama_fixed import EnhancedQueryProcessorOllamaFixed
            self.enhanced_processor = EnhancedQueryProcessorOllamaFixed(ollama_client)
            logger.info("Enhanced Query Processor initialized successfully")
        except ImportError as e:
            logger.warning(f"Enhanced Query Processor not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Enhanced Query Processor: {e}")

    def process_query(self, query: str) -> QueryAnalysis:
        """
        Process query using best available processor.

        Args:
            query: User's input question

        Returns:
            QueryAnalysis with extracted information
        """
        if self.enhanced_processor is not None:
            try:
                result = self.enhanced_processor.process_query(query)
                return self._convert_enhanced_result(result)
            except Exception as e:
                logger.warning(f"Enhanced processing failed: {e}. Using baseline.")

        return self.baseline.process_query(query)

    def _convert_enhanced_result(self, result: Any) -> QueryAnalysis:
        """Convert enhanced processor result to QueryAnalysis."""
        # Handle different result formats
        if isinstance(result, QueryAnalysis):
            return result

        # If result is a different format, convert it
        if hasattr(result, 'programs'):
            intent = result.intent if hasattr(result, 'intent') else QueryIntent.FACTUAL
            if hasattr(intent, 'value'):
                intent = QueryIntent.from_string(intent.value)
            elif isinstance(intent, str):
                intent = QueryIntent.from_string(intent)

            entities = []
            if hasattr(result, 'entities'):
                for e in result.entities:
                    if isinstance(e, AcademicEntity):
                        entities.append(e)
                    elif hasattr(e, 'text'):
                        entities.append(AcademicEntity(
                            text=e.text,
                            label=getattr(e, 'label', 'UNKNOWN'),
                            normalized=getattr(e, 'normalized', e.text)
                        ))

            return QueryAnalysis(
                cleaned_query=getattr(result, 'cleaned_query', ''),
                intent=intent,
                entities=entities,
                metadata_filters=getattr(result, 'metadata_filters', {}),
                sub_queries=getattr(result, 'sub_queries', []),
                programs=getattr(result, 'programs', []),
                page_types=getattr(result, 'page_types', []),
            )

        # Fallback to baseline
        return self.baseline.process_query(str(result))
