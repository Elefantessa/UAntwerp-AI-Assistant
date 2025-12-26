"""
Strict Generator
================

Answer generation in strict mode using JSON output.
Only answers from provided context, returns empty if unsure.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class StrictGenerator(BaseGenerator):
    """
    Strict mode generator that only answers from provided context.

    Returns structured JSON output with answer and citations.
    Returns empty answer if information is not in context.
    """

    SYSTEM_PROMPT = (
        "You are a precise academic assistant. Answer ONLY using the provided CONTEXT. "
        "If the answer is not present, return an empty string. "
        'Output a compact JSON object: {"answer": str, "citations": [str]}.'
    )

    USER_PROMPT_TEMPLATE = (
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        'Respond in JSON as {{"answer": ..., "citations": [...]}}. '
        "Do NOT add any text outside JSON."
    )

    def generate(
        self,
        question: str,
        contexts: List[str],
        urls: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Generate answer in strict mode.

        Args:
            question: User's question
            contexts: List of context texts
            urls: Optional list of source URLs

        Returns:
            Tuple of (answer text, success flag)
        """
        if not contexts:
            logger.info("[strict_generator] No contexts provided")
            return "", False

        # Format context
        context_blob = self._format_context_blob(contexts)

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_blob,
            question=question
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response
        content = self._invoke_llm(messages)

        # Parse JSON response
        data = self._parse_json_response(content)
        answer = (data.get("answer") or "").strip()

        success = bool(answer)
        logger.info(f"[strict_generator] success={success}, answer_len={len(answer)}")

        return answer, success

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Args:
            content: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        content = content.strip()

        # Try to find JSON object in response
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"[strict_generator] JSON parse failed: {e}")
            return {}
