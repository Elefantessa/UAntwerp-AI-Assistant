"""
Metadata Utilities
==================

Metadata processing for ChromaDB ingestion.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def is_primitive(x: Any) -> bool:
    """Check if value is a primitive type."""
    return isinstance(x, PRIMITIVE_TYPES)


def to_primitive(value: Any, key: str = "") -> Any:
    """Convert value to primitive type."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def ensure_meta_primitives(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all metadata values are primitives."""
    return {k: to_primitive(v, k) for k, v in meta.items()}


def validate_meta(meta: Dict[str, Any]) -> bool:
    """Validate metadata contains only primitives."""
    return all(is_primitive(v) for v in meta.values())


def stable_id(row: Dict[str, Any]) -> str:
    """Generate stable ID from row data."""
    content = row.get("text") or row.get("content") or ""
    url = row.get("url") or row.get("source_url") or ""
    idx = row.get("chunk_index", 0)

    composite = f"{url}::{idx}::{content[:200]}"
    return hashlib.sha256(composite.encode("utf-8", errors="ignore")).hexdigest()[:32]


def _alias(value: Optional[str]) -> str:
    """Normalize alias values."""
    if not value:
        return ""
    return value


def load_keywords_map(path: str) -> Dict[str, List[str]]:
    """Load keywords map from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_meta_prefix(meta: Dict[str, Any], keywords_map: Dict[str, List[str]]) -> str:
    """Build metadata prefix for index text."""
    parts = []

    if meta.get("title"):
        parts.append(f"Title: {meta['title']}")
    if meta.get("program"):
        parts.append(f"Program: {meta['program']}")
    if meta.get("page_type"):
        parts.append(f"Page Type: {meta['page_type']}")

    return " | ".join(parts)


def choose_index_text(
    row: Dict[str, Any],
    use_meta_prefix: bool,
    keywords_map: Dict[str, List[str]]
) -> str:
    """Choose index text for embedding."""
    text = row.get("text") or row.get("content") or ""

    if use_meta_prefix:
        meta = row.get("metadata", {})
        prefix = build_meta_prefix(meta, keywords_map)
        if prefix:
            text = f"{prefix}\n\n{text}"

    return text


def to_chroma_metadata(
    row: Dict[str, Any],
    index_text_used: str,
    embedding_model_name: str
) -> Dict[str, Any]:
    """Convert row to ChromaDB metadata format."""
    meta = row.get("metadata", {})

    def _toi(v, default=0):
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    result = {
        # Core fields
        "source": meta.get("url") or meta.get("source") or row.get("url", ""),
        "title": meta.get("title", ""),
        "program": meta.get("program", "UNKNOWN"),
        "page_type": meta.get("page_type", "other"),

        # Chunk info
        "chunk_index": _toi(row.get("chunk_index", 0)),
        "chunk_count": _toi(row.get("total_chunks", 1)),
        "token_count": _toi(row.get("token_count", 0)),
        "word_count": _toi(row.get("word_count", 0)),

        # Processing info
        "embedding_model": embedding_model_name,
        "index_text_length": len(index_text_used),

        # Optional fields
        "breadcrumbs": meta.get("breadcrumbs", ""),
        "lang": meta.get("lang", "en"),
        "seed_root": meta.get("seed_root", ""),
    }

    return ensure_meta_primitives(result)
