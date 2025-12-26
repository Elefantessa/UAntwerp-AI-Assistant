"""
Advanced Confidence Calculator
==============================

Multi-factor confidence calculation for RAG responses using:
- Cross-encoder reranking scores
- Entity matching
- Source diversity
- Context completeness
- Semantic coherence (LLM-based)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.models.response import ConfidenceMetrics

# Handle optional LangChain dependency
try:
    from langchain.schema import Document
    from langchain_ollama import ChatOllama
except ImportError:
    Document = None  # type: ignore
    ChatOllama = None  # type: ignore

logger = logging.getLogger(__name__)


class AdvancedConfidenceCalculator:
    """
    Advanced confidence calculator for RAG systems.

    Calculates comprehensive confidence scores using multiple metrics
    and LLM-based semantic evaluation.
    """

    # Default weights for confidence factors
    DEFAULT_WEIGHTS = {
        'rerank': 0.30,
        'entity': 0.20,
        'source': 0.15,
        'completeness': 0.15,
        'semantic': 0.20,
    }

    def __init__(
        self,
        ollama_model: str = "llama3.1:latest",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize confidence calculator.

        Args:
            ollama_model: Ollama model for semantic evaluation
            weights: Custom weights for confidence factors
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Initialize LLM for semantic coherence if available
        self.confidence_llm = None
        if ChatOllama is not None:
            try:
                self.confidence_llm = ChatOllama(
                    model=ollama_model,
                    temperature=0,
                    format="json"
                )
            except Exception as e:
                logger.warning(f"Could not initialize confidence LLM: {e}")

    def calculate_confidence(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Any],
        reranked_docs: List[Any],
        cross_encoder_scores: Optional[List[float]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        generation_mode: str = "standard"
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score using multiple metrics.

        Args:
            query: Original user query
            answer: Generated answer
            retrieved_docs: Initially retrieved documents
            reranked_docs: Documents after reranking
            cross_encoder_scores: Scores from cross-encoder
            analysis: Query analysis results
            generation_mode: Generation mode used (strict/flexible)

        Returns:
            ConfidenceMetrics with individual and final scores
        """
        analysis = analysis or {}

        # 1. Cross-encoder reranking confidence
        rerank_confidence = self._calculate_rerank_confidence(cross_encoder_scores or [])

        # 2. Entity matching confidence
        entity_confidence = self._calculate_entity_match_confidence(
            query, reranked_docs, analysis
        )

        # 3. Source diversity confidence
        source_confidence = self._calculate_source_diversity(reranked_docs, generation_mode)

        # 4. Context completeness confidence
        completeness_confidence = self._calculate_context_completeness(query, reranked_docs)

        # 5. Semantic coherence using LLM
        semantic_confidence, reasoning = self._calculate_semantic_coherence(
            query, answer, reranked_docs
        )

        # Calculate weighted confidence
        final_confidence = (
            self.weights["rerank"] * rerank_confidence +
            self.weights["entity"] * entity_confidence +
            self.weights["source"] * source_confidence +
            self.weights["completeness"] * completeness_confidence +
            self.weights["semantic"] * semantic_confidence
        )

        # Ensure confidence is between 0 and 1
        final_confidence = max(0.0, min(1.0, final_confidence))

        return ConfidenceMetrics(
            rerank_confidence=rerank_confidence,
            entity_match_confidence=entity_confidence,
            source_diversity_confidence=source_confidence,
            context_completeness_confidence=completeness_confidence,
            semantic_coherence_confidence=semantic_confidence,
            final_confidence=final_confidence,
            reasoning=reasoning
        )

    def _calculate_rerank_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on cross-encoder reranking scores."""
        if not scores:
            return 0.5  # neutral confidence

        try:
            # Convert to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

            # Filter valid scores
            valid_scores = [float(s) for s in scores if s is not None]
            if not valid_scores:
                return 0.5

            # Normalize scores (cross-encoder scores are usually between -10 and 10)
            normalized_scores = [(s + 10) / 20 for s in valid_scores]

            # Use top scores with position weighting
            top_scores = sorted(normalized_scores, reverse=True)[:3]
            if not top_scores:
                return 0.5

            # Weighted average favoring top results
            weights = [0.5, 0.3, 0.2][:len(top_scores)]
            weighted_score = sum(s * w for s, w in zip(top_scores, weights))

            return max(0.0, min(1.0, weighted_score))

        except Exception:
            return 0.5

    def _calculate_entity_match_confidence(
        self,
        query: str,
        docs: List[Any],
        analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence based on entity matching."""
        if not docs:
            return 0.0

        # Extract expected entities from query and analysis
        expected_entities = set()

        # From analysis
        track = analysis.get("track")
        if track and track != "GENERAL":
            expected_entities.add(track.lower())

        entities = analysis.get("entities", {})
        if isinstance(entities, dict) and entities.get("course_codes"):
            expected_entities.update(code.lower() for code in entities["course_codes"])

        # Simple keyword extraction from query
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        expected_entities.update(keywords[:5])

        if not expected_entities:
            return 0.7  # neutral when no specific entities expected

        # Check matches in documents
        matched_entities = set()
        for doc in docs:
            content = ""
            metadata = {}

            if hasattr(doc, 'page_content'):
                content = (doc.page_content or "").lower()
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata or {}

            # Check in content and metadata
            for entity in expected_entities:
                if entity in content or entity in str(metadata).lower():
                    matched_entities.add(entity)

        match_ratio = len(matched_entities) / len(expected_entities)
        return match_ratio

    def _calculate_source_diversity(self, docs: List[Any], generation_mode: str) -> float:
        """Calculate confidence based on source diversity."""
        if not docs:
            return 0.0

        unique_sources = set()
        unique_metadata_keys = set()

        for doc in docs:
            metadata = getattr(doc, 'metadata', {}) or {}

            # Count unique sources
            source = (
                metadata.get("source") or
                metadata.get("url") or
                metadata.get("document_url") or
                ""
            )
            if source:
                unique_sources.add(source)

            # Count unique metadata patterns
            if metadata:
                unique_metadata_keys.update(metadata.keys())

        source_diversity = min(1.0, len(unique_sources) / max(1, len(docs)))
        metadata_diversity = min(1.0, len(unique_metadata_keys) / 10)

        return (source_diversity * 0.7) + (metadata_diversity * 0.3)

    def _calculate_context_completeness(self, query: str, docs: List[Any]) -> float:
        """Calculate confidence based on context completeness."""
        if not docs:
            return 0.0

        query_terms = set(query.split())

        # Check how many query terms appear in the context
        covered_terms = set()
        total_content_length = 0

        for doc in docs:
            content = ""
            if hasattr(doc, 'page_content'):
                content = (doc.page_content or "").lower()

            total_content_length += len(content)

            for term in query_terms:
                if term.lower() in content:
                    covered_terms.add(term.lower())

        # Calculate coverage
        term_coverage = len(covered_terms) / max(1, len(query_terms))

        # Bonus for sufficient content length
        length_bonus = min(1.0, total_content_length / 1000)

        return (term_coverage * 0.8) + (length_bonus * 0.2)

    def _calculate_semantic_coherence(
        self,
        query: str,
        answer: str,
        docs: List[Any]
    ) -> Tuple[float, str]:
        """Use LLM to evaluate semantic coherence."""
        if not answer.strip():
            return 0.0, "Empty answer"

        if self.confidence_llm is None:
            return self._fallback_semantic_evaluation(query, answer, docs), "Fallback evaluation used"

        evaluation_prompt = f"""
        Evaluate this RAG system response and return ONLY valid JSON.

        Query: {query[:300]}
        Answer: {answer[:500]}
        Context Available: {len(docs)} documents

        Evaluate the answer quality on a scale 0.0 to 1.0 based on:
        - Relevance to query
        - Use of provided context
        - Completeness and accuracy
        - Clarity and coherence

        Return JSON in this EXACT format:
        {{
            "confidence_score": 0.75,
            "reasoning": "Brief evaluation explanation",
            "relevance": "high/medium/low",
            "completeness": "complete/partial/incomplete"
        }}

        Return ONLY the JSON object.
        """

        try:
            response = self.confidence_llm.invoke(evaluation_prompt)
            response_content = response.content.strip()

            # Extract JSON from response
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                evaluation_data = json.loads(json_str)
            else:
                evaluation_data = json.loads(response_content)

            confidence_score = float(evaluation_data.get("confidence_score", 0.5))
            relevance = evaluation_data.get("relevance", "medium")
            completeness = evaluation_data.get("completeness", "partial")
            reasoning = evaluation_data.get("reasoning", "LLM evaluation completed")

            detailed_reasoning = f"{reasoning} (Relevance: {relevance}, Completeness: {completeness})"

            return max(0.0, min(1.0, confidence_score)), detailed_reasoning

        except Exception as e:
            logger.error(f"LLM confidence evaluation failed: {e}")
            return self._fallback_semantic_evaluation(query, answer, docs), "Fallback evaluation used"

    def _fallback_semantic_evaluation(
        self,
        query: str,
        answer: str,
        docs: List[Any]
    ) -> float:
        """Fallback heuristic evaluation when LLM fails."""
        score = 0.5

        # Answer length heuristic
        answer_length = len(answer.split())
        if 10 <= answer_length <= 200:
            score += 0.15
        elif answer_length < 5:
            score -= 0.2

        # Query-answer term overlap
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        overlap = len(query_terms.intersection(answer_terms))
        if overlap > 0:
            score += min(0.2, overlap * 0.03)

        # Context utilization check
        if docs:
            context_text = ""
            for doc in docs[:3]:
                if hasattr(doc, 'page_content'):
                    context_text += " " + (doc.page_content or "").lower()

            answer_lower = answer.lower()
            context_terms = set(context_text.split())
            answer_context_overlap = len(set(answer_lower.split()).intersection(context_terms))
            if answer_context_overlap > 5:
                score += 0.15

        return max(0.0, min(1.0, score))
