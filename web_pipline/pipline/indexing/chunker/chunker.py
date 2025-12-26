"""
Hybrid Chunker
==============

Token-aware text chunking with section-based splitting.
"""

import logging
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_TARGET,
    DEFAULT_OVERLAP,
    DEFAULT_MIN_WORDS,
    DEFAULT_DROP_SHORT,
    BUDGET_TOKENS,
    PAGE_TYPE_PARAMS,
)
from .token_estimator import TokenEstimator
from .text_utils import (
    parse_md_sections,
    normalize_headings,
    consolidate_with_escalation,
    words_count,
)

logger = logging.getLogger(__name__)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore


def make_splitter(target_tokens: int, overlap_pct: int, estimator: TokenEstimator):
    """Create text splitter with token-based chunk size."""
    if not LANGCHAIN_AVAILABLE:
        return None

    # Estimate chars per token
    chars_per_token = estimator.estimate_chars_per_token()
    chunk_chars = int(target_tokens * chars_per_token)
    overlap_chars = int(chunk_chars * overlap_pct / 100)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def hybrid_chunk_text(
    text: str,
    target_tokens: int,
    overlap_pct: int,
    estimator: TokenEstimator,
    min_words: int,
    budget_tokens: int,
) -> List[str]:
    """
    Chunk text using hybrid approach: section-based + character splitting.

    Args:
        text: Input text
        target_tokens: Target tokens per chunk
        overlap_pct: Overlap percentage
        estimator: Token estimator
        min_words: Minimum words per section
        budget_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Normalize headings
    text = normalize_headings(text)

    # Parse into sections
    sections = parse_md_sections(text)

    # Consolidate small sections
    sections = consolidate_with_escalation(
        sections,
        base_min_words=min_words,
        parent_level=2,
        target_for_page=target_tokens,
    )

    # Create splitter for large sections
    splitter = make_splitter(target_tokens, overlap_pct, estimator)

    chunks = []
    for sec in sections:
        heading = sec.get("heading", "")
        body = sec.get("body", "")

        # Compose section text
        if heading:
            level = sec.get("level", 1)
            section_text = f"{'#' * level} {heading}\n\n{body}".strip()
        else:
            section_text = body.strip()

        if not section_text:
            continue

        # Check if section fits in budget
        section_tokens = estimator.count(section_text)

        if section_tokens <= budget_tokens:
            chunks.append(section_text)
        elif splitter is not None:
            # Split large sections
            sub_chunks = splitter.split_text(section_text)
            chunks.extend(sub_chunks)
        else:
            # Fallback: simple split by paragraphs
            paragraphs = section_text.split("\n\n")
            current = ""
            for para in paragraphs:
                test = (current + "\n\n" + para).strip() if current else para
                if estimator.count(test) <= budget_tokens:
                    current = test
                else:
                    if current:
                        chunks.append(current)
                    current = para
            if current:
                chunks.append(current)

    return chunks


def postprocess_chunks(
    chunks: List[str],
    estimator: TokenEstimator,
    min_tokens: int = 0,
    drop_short_tokens: int = 0,
    budget_tokens: int = 999999,
    overlap_pct: int = 0,
) -> List[str]:
    """
    Post-process chunks: merge small, drop tiny, split large.

    Args:
        chunks: Input chunks
        estimator: Token estimator
        min_tokens: Minimum tokens per chunk (merge if below)
        drop_short_tokens: Drop chunks below this token count
        budget_tokens: Maximum tokens per chunk
        overlap_pct: Overlap percentage for splits

    Returns:
        Processed chunks
    """
    if not chunks:
        return []

    result = []

    # Step 1: Merge small chunks
    buffer = ""
    for chunk in chunks:
        if not chunk.strip():
            continue

        test = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        test_tokens = estimator.count(test)

        if test_tokens <= budget_tokens:
            buffer = test
        else:
            if buffer:
                result.append(buffer)
            buffer = chunk

    if buffer:
        result.append(buffer)

    # Step 2: Drop short chunks
    if drop_short_tokens > 0:
        result = [c for c in result if estimator.count(c) >= drop_short_tokens]

    # Step 3: Split chunks that exceed budget
    final = []
    splitter = make_splitter(budget_tokens // 2, overlap_pct, estimator)

    for chunk in result:
        if estimator.count(chunk) > budget_tokens:
            if splitter is not None:
                final.extend(splitter.split_text(chunk))
            else:
                final.append(chunk)
        else:
            final.append(chunk)

    return final


class HybridChunker:
    """
    Hybrid text chunker with token-aware splitting.

    Combines section-based parsing with character-based splitting
    to produce semantically coherent chunks.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_target: int = DEFAULT_TARGET,
        default_overlap: int = DEFAULT_OVERLAP,
        min_words: int = DEFAULT_MIN_WORDS,
        drop_short: int = DEFAULT_DROP_SHORT,
        budget_tokens: int = BUDGET_TOKENS,
    ):
        """
        Initialize chunker.

        Args:
            model_name: Tokenizer model name
            default_target: Default target tokens per chunk
            default_overlap: Default overlap percentage
            min_words: Minimum words per section
            drop_short: Drop chunks below this token count
            budget_tokens: Maximum tokens per chunk
        """
        self.estimator = TokenEstimator(model_name)
        self.default_target = default_target
        self.default_overlap = default_overlap
        self.min_words = min_words
        self.drop_short = drop_short
        self.budget_tokens = budget_tokens

    def get_params_for_page_type(self, page_type: Optional[str]) -> Dict[str, int]:
        """Get chunking parameters for page type."""
        params = PAGE_TYPE_PARAMS.get(
            page_type or "other",
            {"target": self.default_target, "overlap": self.default_overlap}
        )
        return params

    def chunk(
        self,
        text: str,
        page_type: Optional[str] = None,
        target: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Chunk text into semantically coherent pieces.

        Args:
            text: Input text
            page_type: Optional page type for custom parameters
            target: Override target tokens
            overlap: Override overlap percentage

        Returns:
            List of text chunks
        """
        # Get parameters
        params = self.get_params_for_page_type(page_type)
        target_tokens = target or params["target"]
        overlap_pct = overlap or params["overlap"]

        # Perform chunking
        chunks = hybrid_chunk_text(
            text=text,
            target_tokens=target_tokens,
            overlap_pct=overlap_pct,
            estimator=self.estimator,
            min_words=self.min_words,
            budget_tokens=self.budget_tokens,
        )

        # Post-process
        chunks = postprocess_chunks(
            chunks=chunks,
            estimator=self.estimator,
            min_tokens=self.min_words,
            drop_short_tokens=self.drop_short,
            budget_tokens=self.budget_tokens,
            overlap_pct=overlap_pct,
        )

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.estimator.count(text)
