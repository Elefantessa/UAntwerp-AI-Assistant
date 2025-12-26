# Chunker Module
"""Text chunking components with token-aware splitting."""

from .config import DEFAULT_TARGET, DEFAULT_OVERLAP, PAGE_TYPE_PARAMS
from .token_estimator import TokenEstimator
from .text_utils import (
    parse_md_sections,
    normalize_headings,
    consolidate_sections,
)
from .chunker import HybridChunker, postprocess_chunks

__all__ = [
    "DEFAULT_TARGET",
    "DEFAULT_OVERLAP",
    "PAGE_TYPE_PARAMS",
    "TokenEstimator",
    "HybridChunker",
    "parse_md_sections",
    "normalize_headings",
    "consolidate_sections",
    "postprocess_chunks",
]
