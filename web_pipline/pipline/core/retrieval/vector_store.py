"""
Vector Store Manager
====================

Manages vector store initialization and retrieval operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    # Document moved to langchain_core in newer versions
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None  # type: ignore
    Chroma = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore


class VectorStoreManager:
    """
    Manages vector store operations including initialization and retrieval.

    Handles embedding model loading with device fallback and provides
    MMR-based retrieval with metadata filtering.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embed_model: str,
        device: Optional[str] = None
    ):
        """
        Initialize vector store manager.

        Args:
            persist_dir: Path to persisted Chroma DB
            collection_name: Name of the collection
            embed_model: HuggingFace embedding model name
            device: Device for embeddings (None for auto-detect)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Install with: pip install langchain langchain-chroma langchain-huggingface"
            )

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.device = device

        # Build components
        self.vectorstore, self.embeddings = self._build_components()
        logger.info(f"VectorStoreManager initialized: collection={collection_name}")

    def _build_components(self) -> Tuple[Any, Any]:
        """Build vector store and embeddings components."""

        def make_embeddings(dev: str):
            return HuggingFaceEmbeddings(
                model_name=self.embed_model,
                model_kwargs={"device": dev}
            )

        # Try primary device, fallback to CPU
        try:
            primary = self.device or "cuda"
            embeddings = make_embeddings(primary)
            logger.info(f"Embeddings initialized on device: {primary}")
        except Exception as e:
            logger.warning(f"Device '{primary}' failed ({e}); falling back to CPU.")
            embeddings = make_embeddings("cpu")

        # Initialize vector store
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=embeddings,
        )

        return vectorstore, embeddings

    def retrieve_mmr(
        self,
        query: str,
        k: int = 50,
        fetch_k: int = 100,
        mmr_lambda: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        debug_metadata: bool = False
    ) -> List[Any]:
        """
        Retrieve documents using MMR (Maximal Marginal Relevance).

        Args:
            query: Search query
            k: Number of documents to return
            fetch_k: Number of documents to fetch before MMR
            mmr_lambda: Diversity parameter (0=max diversity, 1=max relevance)
            filters: Metadata filters
            debug_metadata: Whether to log metadata debug info

        Returns:
            List of retrieved documents
        """
        # Debug metadata if requested
        if debug_metadata or filters:
            self._debug_metadata_sample(query, mmr_lambda)

        # Create retriever with MMR
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": mmr_lambda,
                "filter": filters,
            },
        )

        try:
            docs = retriever.invoke(query)
            logger.info(
                f"[retrieve_mmr] retrieved={len(docs)} "
                f"(k={k}, fetch_k={fetch_k}, λ={mmr_lambda})"
            )
            return docs

        except Exception as e:
            logger.error(f"[retrieve_mmr] Retrieval failed with filter {filters}: {e}")

            # Retry without filters
            logger.info("[retrieve_mmr] Retrying without filters...")
            fallback_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": mmr_lambda,
                },
            )
            return fallback_retriever.invoke(query)

    def _debug_metadata_sample(self, query: str, mmr_lambda: float) -> None:
        """Log sample metadata for debugging."""
        try:
            debug_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 3,
                    "fetch_k": 10,
                    "lambda_mult": mmr_lambda,
                },
            )
            debug_docs = debug_retriever.invoke(query)
            if debug_docs:
                logger.info(
                    f"[retrieve_mmr] Sample metadata fields: "
                    f"{list(debug_docs[0].metadata.keys())}"
                )
            else:
                logger.warning("[retrieve_mmr] No documents found even without filters!")
        except Exception as e:
            logger.warning(f"[retrieve_mmr] Debug metadata inspection failed: {e}")

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """Simple similarity search for health checks."""
        return self.vectorstore.similarity_search(query, k=k)
