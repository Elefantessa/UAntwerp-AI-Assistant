"""
JSON Utilities
==============

Safe JSON parsing and handling utilities.
"""

import json
import re
from typing import Any, Dict, Optional


def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse JSON from a string, handling various formats.

    Tries to find a JSON object within possible non-JSON wrappers
    like markdown code blocks.

    Args:
        s: String that may contain JSON

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not s:
        return None

    s = s.strip()

    # Try to find a JSON object within possible wrappers
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        s = match.group(0)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def safe_json_loads(s: str, default: Any = None) -> Any:
    """
    Safely parse JSON with a default fallback.

    Args:
        s: String to parse as JSON
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.

    Handles cases where JSON is embedded in markdown code blocks
    or surrounded by other text.

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted and parsed JSON or None
    """
    if not text:
        return None

    # Try to find JSON in code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find standalone JSON object
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
