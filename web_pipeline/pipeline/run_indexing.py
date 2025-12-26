#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Indexing Pipeline
==========================

Run the complete indexing pipeline: Scraping → Chunking → Ingestion

Uses configuration from config/settings.py for consistency with the main system.

Usage:
    # Full pipeline
    python run_indexing.py --full

    # Individual steps
    python run_indexing.py --scrape
    python run_indexing.py --chunk
    python run_indexing.py --ingest

    # Custom paths
    python run_indexing.py --full --output-dir /path/to/output --db-dir /path/to/db
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Import configuration
from config.settings import IndexingConfig
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger("indexing-pipeline")


def parse_args():
    """Parse command line arguments."""
    # Get defaults from config
    config = IndexingConfig()

    parser = argparse.ArgumentParser(
        description="Unified Indexing Pipeline for UAntwerp RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_indexing.py --full                    # Run complete pipeline
  python run_indexing.py --scrape --max-pages 50   # Only scrape, limit pages
  python run_indexing.py --chunk                   # Only chunk existing data
  python run_indexing.py --ingest --recreate       # Re-ingest, clear DB first
        """
    )

    # Pipeline steps
    step_group = parser.add_argument_group("Pipeline Steps")
    step_group.add_argument("--full", action="store_true", help="Run complete pipeline")
    step_group.add_argument("--scrape", action="store_true", help="Run scraping only")
    step_group.add_argument("--chunk", action="store_true", help="Run chunking only")
    step_group.add_argument("--ingest", action="store_true", help="Run ingestion only")

    # Paths (defaults from config)
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--output-dir", type=str, default=config.output_dir,
                           help=f"Output directory (default: {config.output_dir})")
    path_group.add_argument("--db-dir", type=str, default=config.db_dir,
                           help=f"ChromaDB directory (default: {config.db_dir})")
    path_group.add_argument("--collection", type=str, default=config.collection_name,
                           help=f"Collection name (default: {config.collection_name})")

    # Scraping options (defaults from config)
    scrape_group = parser.add_argument_group("Scraping Options")
    scrape_group.add_argument("--seeds", nargs="+", default=list(config.seeds),
                             help="Seed URLs for scraping")
    scrape_group.add_argument("--max-pages", type=int, default=config.max_pages_per_seed,
                             help=f"Max pages per seed (default: {config.max_pages_per_seed})")
    scrape_group.add_argument("--max-depth", type=int, default=config.max_depth,
                             help=f"Max crawl depth (default: {config.max_depth})")
    scrape_group.add_argument("--concurrency", type=int, default=config.concurrency,
                             help=f"Concurrent requests (default: {config.concurrency})")

    # Chunking options (defaults from config)
    chunk_group = parser.add_argument_group("Chunking Options")
    chunk_group.add_argument("--target-tokens", type=int, default=config.target_tokens,
                            help=f"Target tokens per chunk (default: {config.target_tokens})")
    chunk_group.add_argument("--overlap", type=int, default=config.overlap_pct,
                            help=f"Overlap percentage (default: {config.overlap_pct})")

    # Ingestion options (defaults from config)
    ingest_group = parser.add_argument_group("Ingestion Options")
    ingest_group.add_argument("--recreate", action="store_true", default=config.recreate,
                             help="Recreate collection (delete existing)")
    ingest_group.add_argument("--device", type=str, default=config.device,
                             help=f"Device for embeddings (default: {config.device})")
    ingest_group.add_argument("--embed-model", type=str, default=config.embed_model,
                             help=f"Embedding model (default: {config.embed_model})")
    ingest_group.add_argument("--batch-size", type=int, default=config.batch_size,
                             help=f"Batch size for ingestion (default: {config.batch_size})")

    return parser.parse_args()


async def run_scraping(args, output_dir: Path) -> Path:
    """Run the scraping step."""
    logger.info("=" * 60)
    logger.info("📥 STEP 1: SCRAPING")
    logger.info("=" * 60)

    from indexing.scraper import WebScraper

    output_file = output_dir / "scraped_content.json"

    logger.info(f"Seeds: {len(args.seeds)}")
    for seed in args.seeds:
        logger.info(f"  - {seed}")
    logger.info(f"Max pages per seed: {args.max_pages}")
    logger.info(f"Max depth: {args.max_depth}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Output: {output_file}")

    start_time = time.time()

    scraper = WebScraper(
        seeds=args.seeds,
        max_pages_per_seed=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        force_study_programme_once=True,
        force_other_essentials=True,
    )

    await scraper.run(output_file)

    elapsed = time.time() - start_time
    logger.info(f"✅ Scraping complete: {len(scraper.pages)} pages in {elapsed:.1f}s")

    return output_file


def run_chunking(args, output_dir: Path, scraped_file: Path) -> Path:
    """Run the chunking step."""
    logger.info("=" * 60)
    logger.info("📦 STEP 2: CHUNKING")
    logger.info("=" * 60)

    from indexing.chunker import HybridChunker

    chunks_file = output_dir / "chunks.jsonl"

    logger.info(f"Input: {scraped_file}")
    logger.info(f"Output: {chunks_file}")
    logger.info(f"Target tokens: {args.target_tokens}")
    logger.info(f"Overlap: {args.overlap}%")

    start_time = time.time()

    # Load scraped data
    with open(scraped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    logger.info(f"Loaded {len(pages)} pages")

    # Initialize chunker with config
    chunker = HybridChunker(
        default_target=args.target_tokens,
        default_overlap=args.overlap,
    )

    all_chunks = []

    for page in pages:
        markdown = page.get("markdown", "")
        if not markdown.strip():
            continue

        page_type = page.get("metadata", {}).get("page_type", "other")
        chunks = chunker.chunk(markdown, page_type=page_type)

        for i, chunk_text in enumerate(chunks):
            chunk = {
                "text": chunk_text,
                "url": page.get("url", ""),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_count": chunker.count_tokens(chunk_text),
                "word_count": len(chunk_text.split()),
                "metadata": page.get("metadata", {}),
            }
            all_chunks.append(chunk)

    # Save chunks
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    logger.info(f"✅ Chunking complete: {len(all_chunks)} chunks in {elapsed:.1f}s")

    return chunks_file


def run_ingestion(args, chunks_file: Path):
    """Run the ingestion step."""
    logger.info("=" * 60)
    logger.info("💾 STEP 3: INGESTION")
    logger.info("=" * 60)

    from indexing.ingestor import ChromaIngestor

    logger.info(f"Input: {chunks_file}")
    logger.info(f"Database: {args.db_dir}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Embedding model: {args.embed_model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Recreate: {args.recreate}")

    start_time = time.time()

    # Load chunks
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks")

    # Initialize ingestor with config
    ingestor = ChromaIngestor(
        persist_dir=args.db_dir,
        collection_name=args.collection,
        embed_model_name=args.embed_model,
        device=args.device,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    # Ingest
    count = ingestor.ingest(chunks)

    elapsed = time.time() - start_time
    logger.info(f"✅ Ingestion complete: {count} documents in {elapsed:.1f}s")


async def run_full_pipeline(args):
    """Run the complete pipeline."""
    logger.info("🚀 Starting Full Indexing Pipeline")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")
    logger.info("Configuration from: config/settings.py (IndexingConfig)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Step 1: Scraping
    scraped_file = await run_scraping(args, output_dir)

    # Step 2: Chunking
    chunks_file = run_chunking(args, output_dir, scraped_file)

    # Step 3: Ingestion
    run_ingestion(args, chunks_file)

    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info(f"🎉 PIPELINE COMPLETE in {total_elapsed:.1f}s")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Determine which steps to run
    if args.full:
        asyncio.run(run_full_pipeline(args))
    elif args.scrape:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_scraping(args, output_dir))
    elif args.chunk:
        output_dir = Path(args.output_dir)
        scraped_file = output_dir / "scraped_content.json"
        if not scraped_file.exists():
            logger.error(f"Scraped file not found: {scraped_file}")
            logger.error("Run --scrape first or specify correct --output-dir")
            sys.exit(1)
        run_chunking(args, output_dir, scraped_file)
    elif args.ingest:
        output_dir = Path(args.output_dir)
        chunks_file = output_dir / "chunks.jsonl"
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            logger.error("Run --chunk first or specify correct --output-dir")
            sys.exit(1)
        run_ingestion(args, chunks_file)
    else:
        print("No action specified. Use --full, --scrape, --chunk, or --ingest")
        print("Use --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
