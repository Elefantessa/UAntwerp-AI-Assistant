#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RAG Evaluation
==================

CLI script to run RAGAS evaluation on the RAG system.
Uses the API instead of direct database access to avoid ChromaDB lock issues.

Usage:
    # Make sure the server is running first:
    # python main.py --persist-dir ../data/db/unified_chroma_db --port 5007

    # Then run evaluation:
    python run_evaluation.py --questions questions.json --api-url http://localhost:5007
"""

import argparse
import json
import logging
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG system (via API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample questions
  python run_evaluation.py --generate-sample

  # Evaluate using API (default, recommended)
  python run_evaluation.py --questions questions.json

  # Specify API URL
  python run_evaluation.py --questions questions.json --api-url http://localhost:5007

  # Use different Ollama model for RAGAS evaluation
  python run_evaluation.py --questions questions.json --llm-model gpt-oss:20b
        """
    )

    parser.add_argument(
        "--questions", type=str,
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/project_antwerp/hala_alramli/web_pipline/data/evaluation",
        help="Output directory for results"
    )

    # API settings (recommended)
    api_group = parser.add_argument_group("API Settings (Recommended)")
    api_group.add_argument(
        "--api-url", type=str, default="http://localhost:5007",
        help="RAG System API URL (default: http://localhost:5007)"
    )

    # LLM settings for RAGAS evaluation
    llm_group = parser.add_argument_group("RAGAS LLM Settings")
    llm_group.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "openai"],
        help="LLM provider for RAGAS: ollama (free) or openai (paid)"
    )
    llm_group.add_argument(
        "--llm-model", type=str, default="gpt-oss:20b",
        help="LLM model for RAGAS evaluation (default: gpt-oss:20b)"
    )
    llm_group.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL"
    )

    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a sample questions file"
    )

    return parser.parse_args()


def generate_sample_questions(output_path: str) -> None:
    """Generate a sample questions file."""
    sample_questions = [
        {
            "question": "What are the admission requirements for the Master in Computer Science: Data Science and AI?",
            "ground_truth": "Applicants need a bachelor's degree in Computer Science or related field comparable to UAntwerp's bachelor."
        },
        {
            "question": "How many ECTS credits is the Master in Data Science programme?",
            "ground_truth": "The Master in Data Science programme is 120 ECTS credits."
        },
        {
            "question": "What are the tuition fees for international students?",
            "ground_truth": "Tuition fees vary based on nationality and programme."
        },
        {
            "question": "When is the application deadline for the Master programmes?",
            "ground_truth": "Application deadlines vary, typically March 1st for international students."
        },
        {
            "question": "What programming languages are taught in the Software Engineering track?",
            "ground_truth": "The programme covers various programming languages and software development practices."
        },
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, indent=2, ensure_ascii=False)

    logger.info(f"Sample questions saved to {path}")
    print(f"✅ Sample questions file created: {path}")
    print(f"   Contains {len(sample_questions)} questions")


def check_api_health(api_url: str) -> bool:
    """Check if the RAG API is running."""
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def query_api(api_url: str, question: str) -> dict:
    """Query the RAG API."""
    try:
        response = requests.post(
            f"{api_url}/api/query",
            json={"query": question},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API query failed: {e}")
        return {"error": str(e)}


def load_questions(questions_file: str) -> list:
    """Load test questions from JSON file."""
    path = Path(questions_file)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} questions from {questions_file}")
    return questions


def run_rag_via_api(api_url: str, questions: list) -> list:
    """Run RAG queries via API."""
    results = []

    for i, q in enumerate(questions):
        question = q.get("question", "")
        ground_truth = q.get("ground_truth", "")

        logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")

        # Query API
        response = query_api(api_url, question)

        if "error" in response:
            result = {
                "question": question,
                "answer": "",
                "contexts": [],
                "sources": [],
                "confidence": 0.0,
                "ground_truth": ground_truth,
                "success": False,
                "error": response["error"],
            }
        else:
            result = {
                "question": question,
                "answer": response.get("answer", ""),
                "contexts": response.get("contexts", []),
                "sources": response.get("sources", []),
                "confidence": response.get("confidence", 0.0),
                "ground_truth": ground_truth,
                "success": True,
            }

        results.append(result)

        # Small delay to not overwhelm the server
        time.sleep(0.5)

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Completed {successful}/{len(results)} questions successfully")

    return results


def run_ragas_evaluation(results: list, args) -> dict:
    """Run RAGAS evaluation on results."""
    try:
        from evaluation import EvaluationConfig, RAGASEvaluator

        # Filter successful results with contexts
        valid_results = [r for r in results if r["success"] and r.get("contexts")]

        if not valid_results:
            logger.warning("No valid results to evaluate with RAGAS")
            return {"error": "No valid results with contexts"}

        # Create evaluator config
        config = EvaluationConfig(
            llm_provider=args.provider,
            llm_model=args.llm_model,
            ollama_base_url=args.ollama_url,
        )

        # Initialize evaluator
        evaluator = RAGASEvaluator(config)

        # Prepare data
        questions = [r["question"] for r in valid_results]
        answers = [r["answer"] for r in valid_results]
        contexts = [r["contexts"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results if r["ground_truth"]]

        # Run RAGAS evaluation
        logger.info(f"Running RAGAS evaluation on {len(valid_results)} results...")
        scores = evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if len(ground_truths) == len(questions) else None,
        )

        return scores

    except ImportError as e:
        logger.warning(f"RAGAS not available: {e}")
        return {"error": "RAGAS not available"}
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}


def save_results(results: list, scores: dict, output_dir: str) -> Path:
    """Save evaluation results to files."""
    output_path = Path(output_dir)
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
    report = generate_report(results, scores, timestamp)
    report_file = output_path / f"evaluation_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Results saved to {output_path}")
    return output_path


def generate_report(results: list, scores: dict, timestamp: str) -> str:
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
    if "error" in scores:
        report += f"⚠️ {scores['error']}\n\n"
    else:
        for metric, score in scores.items():
            if metric not in ["error", "average"]:
                report += f"| {metric} | {score:.4f} |\n"
        if "average" in scores:
            report += f"\n**Average Score: {scores['average']:.4f}**\n"

    report += """

## Detailed Results

"""
    for i, result in enumerate(results[:10]):  # Show first 10
        status = "✅" if result["success"] else "❌"
        answer = result.get('answer', '')[:200]
        if len(result.get('answer', '')) > 200:
            answer += '...'

        report += f"""### Question {i+1} {status}

**Q:** {result['question']}

**A:** {answer}

**Confidence:** {result['confidence']:.3f}

---

"""

    return report


def main():
    """Main entry point."""
    args = parse_args()

    # Generate sample if requested
    if args.generate_sample:
        output_path = Path(args.output_dir) / "sample_questions.json"
        generate_sample_questions(str(output_path))
        return

    # Check questions file
    if not args.questions:
        print("Error: --questions is required (or use --generate-sample)")
        sys.exit(1)

    if not Path(args.questions).exists():
        print(f"Error: Questions file not found: {args.questions}")
        print("Use --generate-sample to create a sample file")
        sys.exit(1)

    print("=" * 60)
    print("🧪 RAG EVALUATION (via API)")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print(f"RAGAS LLM Provider: {args.provider.upper()}")
    print(f"RAGAS LLM Model: {args.llm_model}")
    print("=" * 60)

    # Check API health
    logger.info("Checking API health...")
    if not check_api_health(args.api_url):
        print(f"\n❌ Error: RAG API is not running at {args.api_url}")
        print("\nPlease start the server first:")
        print("  python main.py --persist-dir ../data/db/unified_chroma_db --port 5007")
        sys.exit(1)

    print("✅ API is healthy!")

    # Load questions
    questions = load_questions(args.questions)

    # Run RAG queries via API
    logger.info("Running RAG queries via API...")
    results = run_rag_via_api(args.api_url, questions)

    # Run RAGAS evaluation
    logger.info("Running RAGAS evaluation...")
    scores = run_ragas_evaluation(results, args)

    # Save results
    output_path = save_results(results, scores, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION COMPLETE")
    print("=" * 60)

    successful = sum(1 for r in results if r["success"])
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0

    print(f"\n📋 RAG Results:")
    print(f"  Questions: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Avg Confidence: {avg_conf:.3f}")

    print(f"\n📈 RAGAS Scores:")
    if "error" in scores:
        print(f"  ⚠️ {scores['error']}")
    else:
        for metric, score in scores.items():
            if metric not in ["error", "average"]:
                print(f"  {metric}: {score:.4f}")
        if "average" in scores:
            print(f"\n  📊 Average: {scores['average']:.4f}")

    print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
