"""
RAG Tester
==========

Complete testing framework for RAG system using RAGAS evaluation.
Integrates with the RAGService for end-to-end testing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvaluationConfig
from .ragas_evaluator import RAGASEvaluator, RAGAS_AVAILABLE

logger = logging.getLogger(__name__)


class RAGTester:
    """
    Complete RAG testing framework.

    Integrates with RAGService to run end-to-end evaluation using RAGAS metrics.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        rag_service: Optional[Any] = None,
    ):
        """
        Initialize RAG tester.

        Args:
            config: Evaluation configuration
            rag_service: Optional RAGService instance (can be set later)
        """
        self.config = config or EvaluationConfig()
        self.rag_service = rag_service
        self.evaluator = None

        if RAGAS_AVAILABLE:
            try:
                self.evaluator = RAGASEvaluator(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize RAGASEvaluator: {e}")

    def set_rag_service(self, rag_service: Any) -> None:
        """Set the RAG service instance."""
        self.rag_service = rag_service

    def load_questions(self, questions_file: str) -> List[Dict[str, str]]:
        """
        Load test questions from JSON file.

        Expected format:
        [
            {"question": "...", "ground_truth": "..."},
            ...
        ]
        """
        path = Path(questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        return questions

    def run_rag_on_questions(
        self,
        questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Run RAG system on all questions.

        Args:
            questions: List of question dictionaries

        Returns:
            List of results with question, answer, contexts, etc.
        """
        if not self.rag_service:
            raise ValueError("RAG service not set. Call set_rag_service() first.")

        results = []

        for i, q in enumerate(questions):
            question = q.get("question", "")
            ground_truth = q.get("ground_truth", "")

            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")

            try:
                # Call RAG service
                response = self.rag_service.process_query(question)

                # Extract components from response
                answer = self._extract_answer(response)
                contexts = self._extract_contexts(response)
                sources = self._extract_sources(response)
                confidence = self._extract_confidence(response)

                result = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "sources": sources,
                    "confidence": confidence,
                    "ground_truth": ground_truth,
                    "success": True,
                }

            except Exception as e:
                logger.error(f"Error processing question: {e}")
                result = {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "sources": [],
                    "confidence": 0.0,
                    "ground_truth": ground_truth,
                    "success": False,
                    "error": str(e),
                }

            results.append(result)

        successful = sum(1 for r in results if r["success"])
        logger.info(f"Completed {successful}/{len(results)} questions successfully")

        return results

    def _extract_answer(self, response: Any) -> str:
        """Extract answer from RAG response."""
        if isinstance(response, dict):
            return response.get("answer", "")
        if hasattr(response, "answer"):
            return response.answer
        return str(response)

    def _extract_contexts(self, response: Any) -> List[str]:
        """Extract contexts from RAG response."""
        if isinstance(response, dict):
            contexts = response.get("contexts", [])
            if contexts:
                return contexts
            # Fallback to documents
            docs = response.get("documents", [])
            return [str(d) for d in docs]
        if hasattr(response, "contexts"):
            return response.contexts
        return []

    def _extract_sources(self, response: Any) -> List[str]:
        """Extract sources from RAG response."""
        if isinstance(response, dict):
            return response.get("sources", [])
        if hasattr(response, "sources"):
            return response.sources
        return []

    def _extract_confidence(self, response: Any) -> float:
        """Extract confidence from RAG response."""
        if isinstance(response, dict):
            return float(response.get("confidence", 0.0))
        if hasattr(response, "confidence"):
            return float(response.confidence)
        return 0.0

    def evaluate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate RAG results using RAGAS.

        Args:
            results: List of RAG results

        Returns:
            Evaluation scores dictionary
        """
        if not self.evaluator:
            logger.warning("RAGAS evaluator not available")
            return {"error": "RAGAS not available"}

        # Filter successful results
        valid_results = [r for r in results if r["success"] and r["contexts"]]

        if not valid_results:
            return {"error": "No valid results to evaluate"}

        # Prepare data
        questions = [r["question"] for r in valid_results]
        answers = [r["answer"] for r in valid_results]
        contexts = [r["contexts"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results if r["ground_truth"]]

        # Run RAGAS evaluation
        scores = self.evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if len(ground_truths) == len(questions) else None,
        )

        return scores

    def save_results(
        self,
        results: List[Dict[str, Any]],
        scores: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Path:
        """
        Save evaluation results to files.

        Args:
            results: RAG results
            scores: RAGAS scores
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_path / f"rag_results_{timestamp}.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Save scores
        scores_file = output_path / f"ragas_scores_{timestamp}.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        # Generate report
        report = self._generate_report(results, scores, timestamp)
        report_file = output_path / f"evaluation_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def _generate_report(
        self,
        results: List[Dict[str, Any]],
        scores: Dict[str, Any],
        timestamp: str,
    ) -> str:
        """Generate markdown evaluation report."""
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        avg_confidence = sum(r["confidence"] for r in results) / total if total else 0

        report = f"""# RAG Evaluation Report

**Generated:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Questions | {total} |
| Successful | {successful} |
| Success Rate | {successful/total*100:.1f}% |
| Avg Confidence | {avg_confidence:.3f} |

## RAGAS Scores

"""
        for metric, score in scores.items():
            if metric != "error":
                report += f"| {metric} | {score:.4f} |\n"

        report += """

## Detailed Results

"""
        for i, result in enumerate(results[:10]):  # Show first 10
            status = "✅" if result["success"] else "❌"
            report += f"""### Question {i+1} {status}

**Q:** {result['question']}

**A:** {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}

**Confidence:** {result['confidence']:.3f}

---

"""

        return report

    def run_complete_evaluation(
        self,
        questions_file: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            questions_file: Path to questions JSON file
            output_dir: Output directory for results

        Returns:
            Dictionary with results and scores
        """
        logger.info("Starting complete RAG evaluation")

        # Load questions
        questions = self.load_questions(questions_file)

        # Run RAG on questions
        results = self.run_rag_on_questions(questions)

        # Evaluate with RAGAS
        scores = self.evaluate_results(results)

        # Save results
        output_path = self.save_results(results, scores, output_dir)

        logger.info(f"Evaluation complete. Results saved to {output_path}")

        return {
            "results": results,
            "scores": scores,
            "output_path": str(output_path),
        }
