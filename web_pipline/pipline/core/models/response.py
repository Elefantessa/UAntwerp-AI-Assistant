"""
Response Data Structures
========================

Data classes for RAG system responses and confidence metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfidenceMetrics:
    """
    Container for confidence calculation metrics.

    Stores individual confidence scores from different evaluation methods
    along with the final weighted confidence score.
    """
    rerank_confidence: float
    entity_match_confidence: float
    source_diversity_confidence: float
    context_completeness_confidence: float
    semantic_coherence_confidence: float
    final_confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rerank_confidence": self.rerank_confidence,
            "entity_match_confidence": self.entity_match_confidence,
            "source_diversity_confidence": self.source_diversity_confidence,
            "context_completeness_confidence": self.context_completeness_confidence,
            "semantic_coherence_confidence": self.semantic_coherence_confidence,
            "final_confidence": self.final_confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class RAGResponse:
    """
    Complete response from the RAG system.

    Contains the answer, confidence metrics, sources, and processing metadata.
    """
    query: str
    answer: str
    confidence: float
    sources: List[str]
    generation_mode: str
    processing_time: float
    reasoning_steps: List[str]
    conflicts_detected: List[str]
    metadata: Dict[str, Any]
    contexts: Optional[List[str]] = None
    retrieved_contexts: Optional[List[str]] = None
    used_urls: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "generation_mode": self.generation_mode,
            "processing_time": self.processing_time,
            "reasoning_steps": self.reasoning_steps,
            "conflicts_detected": self.conflicts_detected,
            "metadata": self.metadata,
            "contexts": self.contexts,
            "retrieved_contexts": self.retrieved_contexts,
            "used_urls": self.used_urls,
        }

    @classmethod
    def error_response(
        cls,
        query: str,
        error: str,
        processing_time: float = 0.0,
        model_name: str = "unknown"
    ) -> "RAGResponse":
        """Create an error response."""
        return cls(
            query=query,
            answer="I encountered an error while processing your question. Please try again or rephrase your query.",
            confidence=0.0,
            sources=[],
            generation_mode="error",
            processing_time=processing_time,
            reasoning_steps=[f"Error: {error}"],
            conflicts_detected=[],
            contexts=[],
            metadata={"error": error, "model_name": model_name},
        )
