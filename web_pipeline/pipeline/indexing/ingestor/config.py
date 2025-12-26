"""
Ingestor Configuration
======================

Default parameters for ChromaDB ingestion.
"""

# Default embedding model
DEFAULT_EMBED_MODEL = "Salesforce/SFR-Embedding-Mistral"

# Batching parameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_ENCODE_BATCH = 16
DEFAULT_MAX_SEQ_LENGTH = 8192

# CUDA settings
DEFAULT_MIN_FREE_GB = 0
