"""
Graph State Definition
======================

TypedDict definition for LangGraph state management.
"""

from typing import Any, Dict, List, TypedDict

# Import Document type - handle optional dependency
try:
    from langchain.schema import Document
except ImportError:
    # Fallback for when langchain is not installed
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class GraphState(TypedDict, total=False):
    """
    State object passed through the LangGraph pipeline.

    Attributes:
        question: The user's input question
        analysis: Query analysis results (intent, entities, etc.)
        plan: Retrieval plan (filters, k values, etc.)
        retrieved_docs: Documents from initial retrieval
        reranked_docs: Documents after cross-encoder reranking
        contexts: Extracted context texts for generation
        used_urls: Source URLs for citations
        answer: Generated answer text
        strict_ok: Whether strict generation succeeded
        telemetry: Processing telemetry data
        cross_encoder_scores: Reranking scores for confidence calculation
    """
    question: str
    analysis: Dict[str, Any]
    plan: Dict[str, Any]
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    contexts: List[str]
    used_urls: List[str]
    answer: str
    strict_ok: bool
    telemetry: Dict[str, Any]
    cross_encoder_scores: List[float]
