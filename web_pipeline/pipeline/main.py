#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG LangGraph System - Entry Point
===================================

Modular RAG (Retrieval-Augmented Generation) system using LangGraph
with a Flask web interface.

This is the slim entry point that wires together all modular components.

Usage:
    python main.py --persist-dir /path/to/db --collection uantwerp_cs_web

For CLI mode:
    python main.py --persist-dir /path/to/db --question "Your question here"
"""

import argparse
import json
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig, RAGConfig, ServerConfig, AppConfig
from config.logging_config import setup_logging
from services.rag_service import RAGService
from api.app import create_app

# Setup logging
logger = setup_logging()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG LangGraph System with Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core system arguments
    parser.add_argument(
        "--persist-dir",
        required=True,
        help="Path to persisted Chroma DB"
    )
    parser.add_argument(
        "--collection",
        default="uantwerp_cs_web",
        help="Chroma collection name"
    )
    parser.add_argument(
        "--embed-model",
        default="Salesforce/SFR-Embedding-Mistral",
        help="HuggingFace embedding model name"
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:latest",
        help="Ollama model name"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for embeddings: cpu|cuda|cuda:N"
    )

    # RAG parameters
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=100,
        help="Number of documents to fetch before MMR"
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda parameter for diversity"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=2000,
        help="Token budget for context"
    )

    # Reranking arguments
    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        default=True,
        help="Enable cross-encoder reranking"
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name"
    )

    # Web server arguments
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Flask host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Flask port"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode"
    )

    # CLI/Testing arguments
    parser.add_argument(
        "--question",
        help="Single question to process (CLI mode)"
    )
    parser.add_argument(
        "--debug-metadata",
        action="store_true",
        help="Debug metadata fields in database"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke tests before starting web server"
    )

    return parser.parse_args()


def build_config(args) -> tuple:
    """Build configuration objects from arguments."""
    model_config = ModelConfig(
        ollama_model=args.ollama_model,
        embed_model=args.embed_model,
        embed_device=args.device,
        use_cross_encoder=args.use_cross_encoder,
        cross_encoder_model=args.cross_encoder_model,
    )

    rag_config = RAGConfig(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        k=args.k,
        fetch_k=args.fetch_k,
        mmr_lambda=args.mmr_lambda,
        token_budget=args.token_budget,
    )

    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    return model_config, rag_config, server_config


def run_cli_mode(rag_service: RAGService, question: str):
    """Run in CLI mode with a single question."""
    print(f"\nProcessing question: {question}")
    response = rag_service.process_query(question)

    result = {
        "question": question,
        "answer": response.answer,
        "generation_mode": response.generation_mode,
        "confidence": response.confidence,
        "processing_time": response.processing_time,
        "sources": response.sources,
        "metadata": response.metadata,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_smoke_tests(rag_service: RAGService):
    """Run smoke tests to verify system functionality."""
    test_queries = [
        "What are the admission requirements for the Master in Computer Science: Data Science and AI?",
        "If I have a bachelor's degree from the Benelux and I don't need a visa, do I have to apply via Mobility Online for the Data Science track?",
        "Who can apply for a tuition fee reduction in the Software Engineering track?",
        "What is the faculty contact email for questions about these master's programmes?"
    ]

    print("\n🧪 Running smoke tests:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: {query}")
        try:
            response = rag_service.process_query(query)
            print(f"   ✅ Success | Mode: {response.generation_mode} | Time: {response.processing_time:.2f}s")
            print(f"   Answer: {response.answer[:150]}{'...' if len(response.answer) > 150 else ''}")
            if response.sources:
                print(f"   Sources: {len(response.sources)} documents")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def run_web_server(rag_service: RAGService, server_config: ServerConfig):
    """Start the Flask web server."""
    app = create_app(rag_service, server_config)

    print("\n" + "=" * 60)
    print("🧠 RAG LangGraph System Ready!")
    print(f"Web Interface: http://{server_config.host}:{server_config.port}")
    print(f"API Endpoint: http://{server_config.host}:{server_config.port}/api/query")
    print(f"Health Check: http://{server_config.host}:{server_config.port}/api/health")
    print(f"Statistics: http://{server_config.host}:{server_config.port}/api/stats")
    print(f"System Info: http://{server_config.host}:{server_config.port}/api/system-info")
    print("=" * 60)

    try:
        app.run(
            host=server_config.host,
            port=server_config.port,
            debug=server_config.debug
        )
    except KeyboardInterrupt:
        print("\nShutting down RAG LangGraph System...")
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("🧠 Initializing RAG LangGraph System")
    print("=" * 60)

    # Build configuration
    model_config, rag_config, server_config = build_config(args)

    # Initialize RAG service
    logger.info("Building RAG LangGraph system...")
    try:
        rag_service = RAGService(
            model_config=model_config,
            rag_config=rag_config,
            debug_metadata=args.debug_metadata,
        )
        logger.info("RAG LangGraph system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise

    # CLI mode for single questions
    if args.question:
        run_cli_mode(rag_service, args.question)
        return

    # Optional smoke tests
    if args.smoke_test:
        run_smoke_tests(rag_service)

    # Start web server
    run_web_server(rag_service, server_config)


if __name__ == "__main__":
    main()
