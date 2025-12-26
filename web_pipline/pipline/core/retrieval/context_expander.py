"""
Context Expander
================

Expands and manages context within token budget for generation.
"""

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class ContextExpander:
    """
    Manages context expansion within a token budget.

    Extracts relevant context from documents while avoiding
    duplicates and staying within token limits.
    """

    # Approximate characters per token
    CHARS_PER_TOKEN = 4

    def __init__(self, token_budget: int = 2000):
        """
        Initialize context expander.

        Args:
            token_budget: Maximum tokens for context
        """
        self.token_budget = token_budget
        self.char_budget = token_budget * self.CHARS_PER_TOKEN

    def expand_context(
        self,
        documents: List[Any],
        max_contexts: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Extract context from documents within budget.

        Args:
            documents: Documents to extract context from
            max_contexts: Maximum number of context chunks

        Returns:
            Tuple of (context texts, source URLs)
        """
        if not documents:
            return [], []

        seen_urls = set()
        contexts: List[str] = []
        used_urls: List[str] = []
        current_length = 0

        for doc in documents:
            if len(contexts) >= max_contexts:
                break

            # Extract URL
            url = self._get_url(doc)

            # Skip duplicate sources
            if url and url in seen_urls:
                continue

            # Extract content
            chunk = self._get_content(doc)
            if not chunk:
                continue

            # Check budget
            if current_length + len(chunk) > self.char_budget:
                # Try to fit partial content
                remaining = self.char_budget - current_length
                if remaining > 100:  # Only add if meaningful
                    chunk = chunk[:remaining] + "..."
                else:
                    break

            # Add context
            contexts.append(chunk)
            current_length += len(chunk)

            if url:
                used_urls.append(url)
                seen_urls.add(url)

        logger.info(
            f"[expand_context] contexts={len(contexts)} urls={len(used_urls)} "
            f"chars={current_length}/{self.char_budget}"
        )

        return contexts, used_urls

    def _get_url(self, doc: Any) -> str:
        """Extract URL from document metadata."""
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata or {}
            return (
                metadata.get("source") or
                metadata.get("url") or
                metadata.get("document_url") or
                ""
            )
        elif isinstance(doc, dict):
            return (
                doc.get("source") or
                doc.get("url") or
                doc.get("document_url") or
                ""
            )
        return ""

    def _get_content(self, doc: Any) -> str:
        """Extract content from document."""
        if hasattr(doc, 'page_content'):
            return (doc.page_content or "").strip()
        elif isinstance(doc, dict):
            return (doc.get('page_content') or doc.get('content') or "").strip()
        elif isinstance(doc, str):
            return doc.strip()
        return ""

    def format_context_blob(
        self,
        contexts: List[str],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Format contexts into a single text blob.

        Args:
            contexts: List of context strings
            separator: Separator between contexts

        Returns:
            Formatted context blob
        """
        return separator.join(contexts)
