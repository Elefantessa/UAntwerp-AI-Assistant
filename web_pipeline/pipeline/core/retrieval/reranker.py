"""
Cross-Encoder Reranker
======================

Document reranking using cross-encoder models for improved relevance.
"""

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Handle optional dependency
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None  # type: ignore


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.

    Cross-encoders provide more accurate relevance scoring than
    bi-encoders but are slower since they process query-document
    pairs together.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        enabled: bool = True
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Cross-encoder model name
            device: Device to run on ("cpu" or "cuda")
            enabled: Whether reranking is enabled
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.enabled = enabled
        self.cross_encoder = None

        if enabled and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(self.model_name, device=device)
                logger.info(f"CrossEncoder initialized: {self.model_name}")
            except Exception as e:
                logger.warning(f"CrossEncoder init failed: {e}")
                self.enabled = False
        elif enabled and not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers not available. Reranking disabled.")
            self.enabled = False

    @property
    def is_available(self) -> bool:
        """Check if cross-encoder is available and enabled."""
        return self.enabled and self.cross_encoder is not None

    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_n: int = 12
    ) -> Tuple[List[Any], List[float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: User query
            documents: List of documents to rerank
            top_n: Number of top documents to return

        Returns:
            Tuple of (reranked documents, scores)
        """
        if not documents:
            return [], []

        if not self.is_available:
            # Return original documents without scores
            return documents[:top_n], []

        # Create query-document pairs
        pairs = []
        for doc in documents:
            content = ""
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, str):
                content = doc
            elif isinstance(doc, dict):
                content = doc.get('page_content', doc.get('content', ''))
            pairs.append((query, content))

        try:
            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Convert to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

            # Sort by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

            # Extract top documents and scores
            reranked = [doc for doc, _ in scored_docs[:top_n]]
            rerank_scores = [float(score) for _, score in scored_docs[:top_n]]

            logger.info(
                f"[rerank] Processed {len(documents)} docs, kept top {len(reranked)}"
            )

            return reranked, rerank_scores

        except Exception as e:
            logger.warning(f"[rerank] CrossEncoder failed ({e}); returning original docs.")
            return documents[:top_n], []
