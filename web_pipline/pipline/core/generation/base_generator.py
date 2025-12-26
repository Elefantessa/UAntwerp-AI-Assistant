"""
Base Generator
===============

Abstract base class for answer generators.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for answer generators.

    Provides common interface for different generation strategies.
    """

    def __init__(self, llm: Any):
        """
        Initialize generator with LLM.

        Args:
            llm: Language model for generation
        """
        self.llm = llm

    @abstractmethod
    def generate(
        self,
        question: str,
        contexts: List[str],
        urls: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Generate an answer based on question and contexts.

        Args:
            question: User's question
            contexts: List of context texts
            urls: Optional list of source URLs

        Returns:
            Tuple of (answer text, success flag)
        """
        pass

    def _invoke_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Invoke the LLM with messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Generated text content
        """
        try:
            response = self.llm.invoke(messages)

            # Handle different response formats
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return ""

    def _format_context_blob(
        self,
        contexts: List[str],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """Format contexts into a single blob."""
        return separator.join(contexts) if contexts else "(no context available)"
