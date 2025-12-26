# Indexing Module - Data Ingestion Pipeline
"""
Data ingestion components for the RAG system.

This module provides three main components:
- scraper: Web scraping with HTML cleaning and Markdown conversion
- chunker: Hybrid text chunking with token-aware splitting
- ingestor: ChromaDB ingestion with embedding support

Note: Each submodule has external dependencies (bs4, transformers, chromadb).
Import specific submodules directly when needed:
    from indexing.scraper import WebScraper
    from indexing.chunker import HybridChunker
    from indexing.ingestor import ChromaIngestor
"""

# Lazy imports to avoid dependency errors
def __getattr__(name):
    if name == "WebScraper":
        from .scraper.scraper import WebScraper
        return WebScraper
    elif name == "HtmlCleaner":
        from .scraper.html_cleaner import HtmlCleaner
        return HtmlCleaner
    elif name == "MarkdownConverter":
        from .scraper.markdown_converter import MarkdownConverter
        return MarkdownConverter
    elif name == "HybridChunker":
        from .chunker.chunker import HybridChunker
        return HybridChunker
    elif name == "TokenEstimator":
        from .chunker.token_estimator import TokenEstimator
        return TokenEstimator
    elif name == "ChromaIngestor":
        from .ingestor.ingestor import ChromaIngestor
        return ChromaIngestor
    elif name == "DevicePlanner":
        from .ingestor.device_planner import DevicePlanner
        return DevicePlanner
    raise AttributeError(f"module 'indexing' has no attribute '{name}'")


__all__ = [
    "WebScraper",
    "HtmlCleaner",
    "MarkdownConverter",
    "HybridChunker",
    "TokenEstimator",
    "ChromaIngestor",
    "DevicePlanner",
]
