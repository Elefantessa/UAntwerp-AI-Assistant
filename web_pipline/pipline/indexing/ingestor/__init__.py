# Ingestor Module
"""ChromaDB ingestion components with embedding support."""

from .config import DEFAULT_EMBED_MODEL, DEFAULT_BATCH_SIZE
from .metadata_utils import ensure_meta_primitives, validate_meta, stable_id, to_chroma_metadata
from .device_planner import DevicePlanner, resolve_device
from .embeddings import make_model, detect_prompt_keys, pick_doc_prompt_name
from .ingestor import ChromaIngestor, ingest

__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_BATCH_SIZE",
    "ensure_meta_primitives",
    "validate_meta",
    "stable_id",
    "to_chroma_metadata",
    "DevicePlanner",
    "resolve_device",
    "make_model",
    "detect_prompt_keys",
    "pick_doc_prompt_name",
    "ChromaIngestor",
    "ingest",
]
