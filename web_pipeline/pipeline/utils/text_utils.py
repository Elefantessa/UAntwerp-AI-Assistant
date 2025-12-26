"""
Text Utilities
==============

Text processing and manipulation utilities.
"""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean text by normalizing whitespace and removing control characters.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize line endings
    text = re.sub(r'\r\n|\r', '\n', text)

    # Remove multiple consecutive newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True
) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncated
        word_boundary: Whether to truncate at word boundaries

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]

    truncated = text[:truncate_at]

    if word_boundary:
        # Find last space before truncation point
        last_space = truncated.rfind(' ')
        if last_space > truncate_at // 2:  # Only if reasonable
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def extract_keywords(
    text: str,
    min_length: int = 3,
    max_keywords: int = 10
) -> List[str]:
    """
    Extract keywords from text (simple word extraction).

    Args:
        text: Input text
        min_length: Minimum word length
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    if not text:
        return []

    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Filter by length and uniqueness
    seen = set()
    keywords = []
    for word in words:
        if len(word) >= min_length and word not in seen:
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= max_keywords:
                break

    return keywords


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    return ' '.join(text.split())
