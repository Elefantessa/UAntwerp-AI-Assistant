"""
Evaluation Configuration
========================

Configuration for RAGAS evaluation metrics and settings.
Supports Ollama (free/local) for both LLM and Embeddings - NO OpenAI required.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Available RAGAS metrics
DEFAULT_METRICS = [
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_correctness",
]


@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation."""

    # LLM Provider: "ollama" (free/local) or "openai" (paid)
    llm_provider: str = "ollama"

    # LLM Model
    llm_model: str = "gpt-oss:latest"  # Default Ollama model for evaluation
    llm_temperature: float = 0.0

    # Ollama settings (for local LLM and embeddings)
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI settings (optional, for paid API)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # Embeddings settings (LOCAL by default - no OpenAI needed)
    embed_provider: str = "ollama"  # "ollama" or "huggingface"
    embed_model: str = "nomic-embed-text"  # Ollama embedding model

    # Metrics to evaluate
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS.copy())

    # Output settings
    output_dir: str = "/project_antwerp/hala_alramli/web_pipline/data/evaluation"
    save_detailed_results: bool = True

    # Batch settings
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 2.0

    # Context handling
    max_context_length: int = 4000
    include_sources: bool = True
