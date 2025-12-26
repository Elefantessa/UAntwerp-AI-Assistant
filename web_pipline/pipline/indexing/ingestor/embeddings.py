"""
Embeddings
==========

Embedding model loading and prompt detection.
"""

import logging
from typing import List, Optional

from .config import DEFAULT_EMBED_MODEL, DEFAULT_MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


def detect_prompt_keys(model: "SentenceTransformer") -> List[str]:
    """Detect available prompt keys from model."""
    if not hasattr(model, "prompts") or not model.prompts:
        return []
    return list(model.prompts.keys())


def pick_doc_prompt_name(
    model: "SentenceTransformer",
    prefer: Optional[str] = None
) -> Optional[str]:
    """
    Pick document prompt name from model.

    Args:
        model: SentenceTransformer model
        prefer: Preferred prompt name

    Returns:
        Selected prompt name or None
    """
    keys = detect_prompt_keys(model)

    if not keys:
        return None

    if prefer and prefer != "auto" and prefer in keys:
        return prefer

    # Preference order for document prompts
    doc_preferences = ["document", "passage", "text", "doc"]

    for pref in doc_preferences:
        for key in keys:
            if pref in key.lower():
                return key

    # Fall back to first available
    return keys[0] if keys else None


def make_model(
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    resolved_device: str = "cpu",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    trust_remote_code: bool = True,
) -> "SentenceTransformer":
    """
    Create SentenceTransformer model.

    Args:
        embed_model_name: Model name or path
        resolved_device: Device to load on
        dtype: Data type
        load_in_8bit: Whether to use 8-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device map for distributed loading
        max_seq_length: Maximum sequence length
        trust_remote_code: Whether to trust remote code

    Returns:
        SentenceTransformer model
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers not available. "
            "Install with: pip install -U sentence-transformers"
        )

    logger.info(f"Loading embedding model: {embed_model_name}")
    logger.info(f"Device: {resolved_device}")

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }

    # Handle quantization
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                )
            elif load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        except ImportError:
            logger.warning("bitsandbytes not available for quantization")

    model = SentenceTransformer(
        embed_model_name,
        device=resolved_device,
        trust_remote_code=trust_remote_code,
    )

    # Set max sequence length
    if max_seq_length:
        model.max_seq_length = max_seq_length

    logger.info(f"Model loaded. Max seq length: {model.max_seq_length}")

    return model
