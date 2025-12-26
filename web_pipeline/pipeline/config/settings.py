"""
Configuration Settings for RAG LangGraph System
================================================

Centralized configuration using dataclasses for type safety and easy management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for ML models used in the system."""

    # LLM Configuration
    ollama_model: str = "llama3.1:latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120

    # Embedding Model
    embed_model: str = "Salesforce/SFR-Embedding-Mistral"
    embed_device: Optional[str] = None  # None means auto-detect (cuda -> cpu fallback)

    # Cross-Encoder for Reranking
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "cpu"
    use_cross_encoder: bool = True


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline parameters."""

    # Vector Store
    persist_dir: str = ""
    collection_name: str = "uantwerp_cs_web"

    # Retrieval Parameters
    k: int = 50  # Number of documents to retrieve
    fetch_k: int = 100  # Number of documents to fetch before MMR
    mmr_lambda: float = 0.7  # MMR diversity parameter

    # Context Management
    token_budget: int = 2000
    max_contexts: int = 10

    # Reranking
    rerank_top_n: int = 12

    # Confidence Thresholds
    strict_min_length: int = 8  # Minimum answer length for strict mode


@dataclass
class ServerConfig:
    """Configuration for Flask web server."""

    host: str = "127.0.0.1"
    port: int = 5006
    debug: bool = False

    # CORS settings
    cors_enabled: bool = True
    cors_origins: str = "*"


@dataclass
class IndexingConfig:
    """Configuration for indexing pipeline."""

    # Output paths
    output_dir: str = "/project_antwerp/hala_alramli/web_pipline/data/raw"
    db_dir: str = "/project_antwerp/hala_alramli/web_pipline/data/db/unified_chroma_db"
    collection_name: str = "uantwerp_cs_web"

    # Scraping settings
    seeds: tuple = (
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-data-science",
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-software-engineering",
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-computer-networks",
    )
    max_pages_per_seed: int = 100
    max_depth: int = 4
    concurrency: int = 8

    # Chunking settings
    target_tokens: int = 350
    overlap_pct: int = 15
    min_words: int = 30

    # Ingestion settings
    embed_model: str = "Salesforce/SFR-Embedding-Mistral"
    device: str = "auto"
    batch_size: int = 128
    recreate: bool = False


@dataclass
class ConfidenceWeights:
    """Weights for confidence calculation components."""

    rerank: float = 0.30
    entity: float = 0.20
    source: float = 0.15
    completeness: float = 0.15
    semantic: float = 0.20

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.rerank + self.entity + self.source + self.completeness + self.semantic
        return abs(total - 1.0) < 0.001


@dataclass
class AppConfig:
    """Complete application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)

    # Debug options
    debug_metadata: bool = False

    @classmethod
    def from_args(cls, args) -> "AppConfig":
        """Create configuration from command line arguments."""
        return cls(
            model=ModelConfig(
                ollama_model=getattr(args, "ollama_model", "llama3.1:latest"),
                embed_model=getattr(args, "embed_model", "Salesforce/SFR-Embedding-Mistral"),
                embed_device=getattr(args, "device", None),
                use_cross_encoder=getattr(args, "use_cross_encoder", True),
                cross_encoder_model=getattr(args, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            ),
            rag=RAGConfig(
                persist_dir=getattr(args, "persist_dir", ""),
                collection_name=getattr(args, "collection", "uantwerp_cs_web"),
                k=getattr(args, "k", 50),
                fetch_k=getattr(args, "fetch_k", 100),
                mmr_lambda=getattr(args, "mmr_lambda", 0.7),
                token_budget=getattr(args, "token_budget", 2000),
            ),
            server=ServerConfig(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 5006),
                debug=getattr(args, "debug", False),
            ),
            debug_metadata=getattr(args, "debug_metadata", False),
        )


# Default configurations for common use cases
DEFAULT_CONFIG = AppConfig()

DEVELOPMENT_CONFIG = AppConfig(
    server=ServerConfig(debug=True),
    debug_metadata=True,
)

PRODUCTION_CONFIG = AppConfig(
    server=ServerConfig(debug=False, host="0.0.0.0"),
)
