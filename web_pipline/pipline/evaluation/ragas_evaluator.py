"""
RAGAS Evaluator
===============

Wrapper for RAGAS evaluation metrics with full local model support.
Uses Ollama for LLM and HuggingFace for embeddings - NO OpenAI required.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .config import EvaluationConfig, DEFAULT_METRICS

logger = logging.getLogger(__name__)

# Try to import RAGAS
try:
    from ragas import evaluate
    # Try new API first
    try:
        from ragas.metrics._simple import (
            AnswerRelevancy,
            Faithfulness,
            ContextPrecision,
            ContextRecall,
        )
        from ragas.metrics._answer_correctness import AnswerCorrectness
        answer_relevancy = AnswerRelevancy()
        faithfulness = Faithfulness()
        context_precision = ContextPrecision()
        context_recall = ContextRecall()
        answer_correctness = AnswerCorrectness()
    except ImportError:
        # Fallback to old API
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
        )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas datasets")

# Try to import LangChain Ollama
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    ChatOllama = None
    OllamaEmbeddings = None

# Try to import LangChain OpenAI (fallback)
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    ChatOpenAI = None

# Try to import HuggingFace embeddings for RAGAS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None


# Metric mapping
METRIC_MAP = {}
if RAGAS_AVAILABLE:
    METRIC_MAP = {
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_correctness": answer_correctness,
    }


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG systems.

    Fully supports local models:
    - LLM: Ollama (gpt-oss:20b, llama3.1, etc.)
    - Embeddings: Ollama or HuggingFace (local, no OpenAI needed)

    Metrics:
    - Answer Relevancy
    - Context Precision
    - Context Recall
    - Faithfulness
    - Answer Correctness
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize RAGAS evaluator.

        Args:
            config: Evaluation configuration
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not available. Install with: pip install ragas datasets"
            )

        self.config = config or EvaluationConfig()
        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
        self.metrics = self._get_metrics()

        logger.info(f"RAGASEvaluator initialized with {len(self.metrics)} metrics")
        logger.info(f"Using LLM: {self.config.llm_model} (provider: {self.config.llm_provider})")
        logger.info(f"Using Embeddings: {self.config.embed_model} (local)")

    def _create_llm(self):
        """Create LLM for RAGAS evaluation."""
        provider = self.config.llm_provider.lower()

        if provider == "ollama":
            if not LANGCHAIN_OLLAMA_AVAILABLE:
                logger.error("langchain-ollama not available")
                raise ImportError("Install langchain-ollama: pip install langchain-ollama")

            logger.info(f"Using Ollama LLM: {self.config.llm_model}")
            return ChatOllama(
                model=self.config.llm_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.llm_temperature,
            )

        elif provider == "openai":
            if not LANGCHAIN_OPENAI_AVAILABLE:
                logger.error("langchain-openai not available")
                raise ImportError("Install langchain-openai: pip install langchain-openai")

            kwargs = {
                "model": self.config.llm_model,
                "temperature": self.config.llm_temperature,
            }
            if self.config.openai_api_key:
                kwargs["api_key"] = self.config.openai_api_key
            if self.config.openai_base_url:
                kwargs["base_url"] = self.config.openai_base_url

            logger.info(f"Using OpenAI LLM: {self.config.llm_model}")
            return ChatOpenAI(**kwargs)

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _create_embeddings(self):
        """Create LOCAL embeddings for RAGAS evaluation (no OpenAI)."""
        embed_provider = getattr(self.config, 'embed_provider', 'ollama')

        # Option 1: Use Ollama embeddings (recommended for full local setup)
        if embed_provider == "ollama" and LANGCHAIN_OLLAMA_AVAILABLE and OllamaEmbeddings:
            try:
                embeddings = OllamaEmbeddings(
                    model=self.config.embed_model,
                    base_url=self.config.ollama_base_url,
                )
                logger.info(f"Using Ollama embeddings: {self.config.embed_model}")
                return embeddings
            except Exception as e:
                logger.warning(f"Could not create Ollama embeddings: {e}")

        # Option 2: Use HuggingFace embeddings (fallback)
        if HUGGINGFACE_AVAILABLE:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embed_model,
                    model_kwargs={"device": "cpu"},
                )
                logger.info(f"Using HuggingFace embeddings: {self.config.embed_model}")
                return embeddings
            except Exception as e:
                logger.warning(f"Could not create HuggingFace embeddings: {e}")

        logger.error("No local embeddings available!")
        return None

    def _get_metrics(self) -> List:
        """Get RAGAS metric instances."""
        metrics = []
        for metric_name in self.config.metrics:
            metric = METRIC_MAP.get(metric_name)
            if metric:
                metrics.append(metric)
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        return metrics

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG responses using RAGAS with LOCAL models only.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (each answer has multiple contexts)
            ground_truths: Optional list of ground truth answers

        Returns:
            Dictionary with evaluation scores
        """
        if not questions or not answers or not contexts:
            return {"error": "Empty input data"}

        if not self.embeddings:
            return {"error": "No local embeddings available. Cannot run RAGAS without embeddings."}

        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths and len(ground_truths) == len(questions):
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        try:
            # CRITICAL: Set dummy OpenAI key to prevent RAGAS from complaining
            # We're using local embeddings, so this won't actually be used
            old_key = os.environ.get("OPENAI_API_KEY", "")
            if not old_key:
                os.environ["OPENAI_API_KEY"] = "sk-local-embeddings-dummy-key"

            # Run evaluation with local models
            eval_kwargs = {
                "dataset": dataset,
                "metrics": self.metrics,
                "llm": self.llm,
                "embeddings": self.embeddings,  # Force local embeddings
            }

            logger.info("Running RAGAS evaluation with LOCAL models (no OpenAI)...")

            try:
                result = evaluate(**eval_kwargs)
            finally:
                # Restore original environment
                if not old_key:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_key

            # Extract scores from RAGAS result
            scores = {}
            try:
                # RAGAS result object has different access patterns
                # Try to get scores from the result
                if hasattr(result, 'scores') and result.scores:
                    # New RAGAS API
                    for score_dict in result.scores:
                        for key, value in score_dict.items():
                            if key in self.config.metrics:
                                if key not in scores:
                                    scores[key] = []
                                scores[key].append(float(value) if value is not None else 0.0)
                    # Average scores across all samples
                    averaged_scores = {}
                    for key, values in scores.items():
                        averaged_scores[key] = sum(values) / len(values) if values else 0.0
                    scores = averaged_scores
                elif hasattr(result, '_scores_dict'):
                    # Direct access to scores dict
                    for metric_name in self.config.metrics:
                        try:
                            val = result._scores_dict.get(metric_name)
                            if val is not None:
                                scores[metric_name] = float(val)
                        except:
                            pass
                else:
                    # Fallback: try dict-like access
                    result_dict = dict(result) if hasattr(result, '__iter__') else {}
                    for metric_name in self.config.metrics:
                        if metric_name in result_dict:
                            scores[metric_name] = float(result_dict[metric_name])

                logger.info(f"Extracted scores: {scores}")
            except Exception as e:
                logger.warning(f"Error extracting scores: {e}")

            # Calculate average (excluding nan values)
            if scores:
                import math
                valid_scores = [v for v in scores.values() if not math.isnan(v)]
                scores["average"] = sum(valid_scores) / len(valid_scores) if valid_scores else float('nan')

            logger.info(f"RAGAS evaluation complete: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single question-answer pair."""
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
