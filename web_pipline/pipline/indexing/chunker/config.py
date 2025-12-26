"""
Chunker Configuration
=====================

Default parameters for text chunking.
"""

# Default chunking parameters
DEFAULT_TARGET = 350          # target tokens per chunk
DEFAULT_OVERLAP = 15          # overlap percentage (0-50)
DEFAULT_MIN_WORDS = 30        # minimum words per chunk
DEFAULT_DROP_SHORT = 25       # drop chunks below this token count
BUDGET_TOKENS = 500           # maximum tokens per chunk

# Page-type specific parameters
PAGE_TYPE_PARAMS = {
    "programme": {"target": 320, "overlap": 18},
    "landing": {"target": 280, "overlap": 15},
    "study-programme": {"target": 350, "overlap": 20},
    "admission-and-enrolment": {"target": 240, "overlap": 18},
    "contact": {"target": 180, "overlap": 10},
    "other": {"target": 260, "overlap": 18},
}

# Index text prefix settings
META_FIELDS_FOR_PREFIX = ["title", "program", "page_type"]
