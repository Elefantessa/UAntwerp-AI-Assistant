"""
Logging Configuration for RAG System
=====================================

Centralized logging setup with consistent formatting.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    logger_name: str = "rag_langgraph"
) -> logging.Logger:
    """
    Configure logging for the RAG system.

    Args:
        level: Logging level (default: INFO)
        log_format: Custom format string (optional)
        logger_name: Name for the logger

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Get or create named logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    return logger


def get_logger(name: str = "rag_langgraph") -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name (default: "rag_langgraph")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Module-level logger for convenience
logger = get_logger()
