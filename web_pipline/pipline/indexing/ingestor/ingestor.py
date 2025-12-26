"""
ChromaDB Ingestor
=================

Ingest chunks into ChromaDB with embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENCODE_BATCH,
    DEFAULT_MAX_SEQ_LENGTH,
)
from .metadata_utils import (
    stable_id,
    to_chroma_metadata,
    choose_index_text,
    load_keywords_map,
)
from .device_planner import DevicePlanner, resolve_device
from .embeddings import make_model, pick_doc_prompt_name

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def batches(seq: List, n: int):
    """Yield batches of size n."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class ChromaIngestor:
    """
    ChromaDB ingestion with embeddings.

    Handles batched ingestion with OOM recovery and device hopping.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        encode_batch: int = DEFAULT_ENCODE_BATCH,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        use_meta_prefix: bool = True,
        recreate: bool = False,
    ):
        """
        Initialize ingestor.

        Args:
            persist_dir: ChromaDB persistence directory
            collection_name: Collection name
            embed_model_name: Embedding model name
            device: Device for embeddings
            batch_size: Outer batch size
            encode_batch: Encoding batch size
            max_seq_length: Maximum sequence length
            use_meta_prefix: Whether to use metadata prefix
            recreate: Whether to recreate collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not available. Install with: pip install chromadb")

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.encode_batch = encode_batch
        self.max_seq_length = max_seq_length
        self.use_meta_prefix = use_meta_prefix
        self.recreate = recreate

        # Initialize device planner
        self.planner = DevicePlanner()
        self.resolved_device = resolve_device(device, self.planner)

        # Load model
        self.model = make_model(
            embed_model_name=embed_model_name,
            resolved_device=self.resolved_device,
            max_seq_length=max_seq_length,
        )

        # Pick prompt name
        self.prompt_name = pick_doc_prompt_name(self.model)
        if self.prompt_name:
            logger.info(f"Using prompt: {self.prompt_name}")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embed_model_name},
        )

        logger.info(f"Collection ready: {collection_name} ({self.collection.count()} existing docs)")

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts to embeddings."""
        kwargs = {}
        if self.prompt_name:
            kwargs["prompt_name"] = self.prompt_name

        embeddings = self.model.encode(
            texts,
            batch_size=self.encode_batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            **kwargs,
        )

        if NUMPY_AVAILABLE:
            return embeddings.tolist()
        return [list(e) for e in embeddings]

    def ingest(
        self,
        chunks: List[Dict[str, Any]],
        keywords_map: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """
        Ingest chunks into ChromaDB.

        Args:
            chunks: List of chunk dictionaries
            keywords_map: Optional keywords map for prefix

        Returns:
            Number of ingested documents
        """
        keywords_map = keywords_map or {}
        total_ingested = 0

        for batch in batches(chunks, self.batch_size):
            ids = []
            texts = []
            metadatas = []

            for row in batch:
                # Generate stable ID
                doc_id = stable_id(row)

                # Choose index text
                index_text = choose_index_text(row, self.use_meta_prefix, keywords_map)
                if not index_text.strip():
                    continue

                # Build metadata
                metadata = to_chroma_metadata(row, index_text, self.embed_model_name)

                ids.append(doc_id)
                texts.append(index_text)
                metadatas.append(metadata)

            if not texts:
                continue

            try:
                # Encode batch
                embeddings = self._encode_batch(texts)

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

                total_ingested += len(ids)
                logger.info(f"Ingested batch of {len(ids)} documents")

            except Exception as e:
                logger.error(f"Failed to ingest batch: {e}")
                # Try with smaller batches
                for sub_batch in batches(list(zip(ids, texts, metadatas)), max(1, len(ids) // 4)):
                    try:
                        sub_ids, sub_texts, sub_metas = zip(*sub_batch)
                        sub_embeddings = self._encode_batch(list(sub_texts))
                        self.collection.add(
                            ids=list(sub_ids),
                            embeddings=sub_embeddings,
                            documents=list(sub_texts),
                            metadatas=list(sub_metas),
                        )
                        total_ingested += len(sub_ids)
                    except Exception as sub_e:
                        logger.error(f"Failed sub-batch: {sub_e}")

        logger.info(f"Total ingested: {total_ingested} documents")
        return total_ingested


def ingest(
    chunks_path: str,
    persist_dir: str,
    collection_name: str,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    outer_batch_size: int = DEFAULT_BATCH_SIZE,
    encode_batch: int = DEFAULT_ENCODE_BATCH,
    recreate: bool = False,
    device: str = "auto",
    use_meta_prefix: bool = True,
    keywords_json: Optional[str] = None,
) -> int:
    """
    Convenience function to ingest chunks from file.

    Args:
        chunks_path: Path to JSONL chunks file
        persist_dir: ChromaDB persistence directory
        collection_name: Collection name
        embed_model_name: Embedding model name
        outer_batch_size: Outer batch size
        encode_batch: Encoding batch size
        recreate: Whether to recreate collection
        device: Device for embeddings
        use_meta_prefix: Whether to use metadata prefix
        keywords_json: Optional path to keywords JSON

    Returns:
        Number of ingested documents
    """
    # Load chunks
    chunks = load_jsonl(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Load keywords
    keywords_map = {}
    if keywords_json:
        keywords_map = load_keywords_map(keywords_json)

    # Create ingestor
    ingestor = ChromaIngestor(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embed_model_name=embed_model_name,
        device=device,
        batch_size=outer_batch_size,
        encode_batch=encode_batch,
        use_meta_prefix=use_meta_prefix,
        recreate=recreate,
    )

    # Ingest
    return ingestor.ingest(chunks, keywords_map)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python ingestor.py <chunks_path> <persist_dir> <collection_name>")
        sys.exit(1)

    count = ingest(
        chunks_path=sys.argv[1],
        persist_dir=sys.argv[2],
        collection_name=sys.argv[3],
    )
    print(f"Ingested {count} documents")
