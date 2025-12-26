"""
Token Estimator
===============

Token counting using HuggingFace tokenizers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore


class TokenEstimator:
    """
    Counts tokens using an HuggingFace tokenizer.

    Provides accurate token counting for embedding model alignment.
    """

    DEFAULT_MODEL = "Salesforce/SFR-Embedding-Mistral"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token estimator.

        Args:
            model_name: HuggingFace model name for tokenizer
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Prevent max_length warnings
                self.tokenizer.model_max_length = int(1e9)
                logger.info(f"TokenEstimator initialized: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Using word-based estimation.")

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if not text:
            return 0

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback to word-based estimation (approx 1.3 tokens per word)
            return int(len(text.split()) * 1.3)

    def estimate_chars_per_token(self, sample_text: str = None) -> float:
        """
        Estimate average characters per token.

        Args:
            sample_text: Optional sample text for estimation

        Returns:
            Average characters per token
        """
        if sample_text is None:
            return 4.0  # reasonable default for English

        tokens = self.count(sample_text)
        if tokens == 0:
            return 4.0
        return len(sample_text) / tokens
