"""
Flexible Generator
==================

Answer generation in flexible mode.
Uses context when available but can supplement with general knowledge.
"""

import json
import logging
from typing import Any, List, Optional, Tuple

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class FlexibleGenerator(BaseGenerator):
    """
    Flexible mode generator that can use general knowledge.

    Uses provided context when available but can supplement
    with general knowledge for more complete answers.
    """

    SYSTEM_PROMPT = (
        "You are a helpful academic assistant. If context is insufficient, "
        "you may use general knowledge, but keep answers concise and cite "
        "provided URLs when relevant."
    )

    USER_PROMPT_TEMPLATE = (
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "{url_hint}\n"
        "Write a concise answer (4-6 sentences). "
        "If you use any facts from the context, cite relevant URLs inline in brackets."
    )

    def generate(
        self,
        question: str,
        contexts: List[str],
        urls: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Generate answer in flexible mode.

        Args:
            question: User's question
            contexts: List of context texts
            urls: Optional list of source URLs

        Returns:
            Tuple of (answer text, success flag)
        """
        # Format context
        context_blob = self._format_context_blob(contexts)

        # Build URL hint
        url_hint = ""
        if urls:
            url_hint = f"Available URLs: {json.dumps(urls)}"

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_blob,
            question=question,
            url_hint=url_hint
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response
        content = self._invoke_llm(messages)

        success = bool(content.strip())
        logger.info(f"[flexible_generator] success={success}, answer_len={len(content)}")

        return content, success
