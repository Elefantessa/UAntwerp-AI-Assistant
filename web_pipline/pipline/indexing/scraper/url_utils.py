"""
URL Utilities
=============

URL normalization, filtering, and link extraction functions.
"""

import hashlib
import re
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

from bs4 import BeautifulSoup

from .config import DOMAIN, ALLOW_PATTERNS, BLOCK_PATTERNS, QUERY_WHITELIST


def sha256_hex(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str) -> str:
    """
    Normalize URLs: enforce http(s), drop fragments, and sanitize query.
    - Keep only whitelisted query params globally.
    - For any '/study-programme/' path: drop ALL query params (avoid `?id=` explosions).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    path = parsed.path

    # For study-programme pages: drop all query
    if "/study-programme/" in path:
        query_items = {}
    else:
        incoming = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_items = {k: v for k, v in incoming.items() if k in QUERY_WHITELIST}

    query = urlencode(sorted(query_items.items()))
    norm = parsed._replace(fragment="", query=query)
    return urlunparse(norm)


def is_allowed_url(url: str) -> bool:
    """
    Allow if:
      - domain matches
      - allow-pattern matches
      - not containing any blocked substrings
    Also: explicitly reject study-programme links that carry an `id=` (even if normalization would drop it).
    """
    if not url:
        return False

    # quick raw guard for '?id=' on study-programme
    if "/study-programme/" in url and "id=" in url:
        return False

    url = normalize_url(url)
    if not url:
        return False
    p = urlparse(url)
    if p.netloc != DOMAIN:
        return False
    for blk in BLOCK_PATTERNS:
        if blk in p.path:
            return False
    if any(pat in p.path for pat in ALLOW_PATTERNS):
        return True
    return False


def extract_links(base_url: str, html: str) -> List[str]:
    """
    Extract, normalize, and filter links from HTML.
    - Collapse any study-programme deep paths to the base '/study-programme/' once.
    - De-duplicate while preserving order.
    """
    links: List[str] = []
    soup = BeautifulSoup(html, "lxml")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        abs_url = urljoin(base_url, href)
        url = normalize_url(abs_url)

        if url and is_allowed_url(url):
            # Collapse any study-programme deep paths to '/study-programme/'
            p = urlparse(url)
            if "/study-programme/" in p.path:
                base_path = p.path.split("/study-programme/")[0] + "/study-programme/"
                url = urlunparse(p._replace(path=base_path, query=""))
            links.append(url)

    # De-duplicate while preserving order
    seen: Set[str] = set()
    uniq = []
    for u in links:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def infer_program_and_page_type(path: str) -> Tuple[str, str]:
    """
    Infer (program, page_type) from the URL path.
    program ∈ {DS, SE, CN, UNKNOWN}
    page_type ∈ {landing, profile, programme-info, study-programme,
                 admission-and-enrolment, job-opportunities, contact, other}
    """
    program = "UNKNOWN"
    if "/master-data-science" in path:
        program = "DS"
    elif "/master-software-engineering" in path:
        program = "SE"
    elif "/master-computer-networks" in path or "/master-computernetworks" in path:
        program = "CN"

    # after the program slug, the next segment indicates page type
    page_type = "landing"
    segments = [seg for seg in path.strip("/").split("/") if seg]
    slug_idx = -1
    for i, seg in enumerate(segments):
        if seg.startswith("master-data-science") or \
           seg.startswith("master-software-engineering") or \
           seg.startswith("master-computer-networks") or \
           seg.startswith("master-computernetworks"):
            slug_idx = i
            break
    if slug_idx != -1 and slug_idx + 1 < len(segments):
        candidate = segments[slug_idx + 1]
        allowed = {
            "profile", "programme-info", "study-programme",
            "admission-and-enrolment", "job-opportunities", "contact"
        }
        page_type = candidate if candidate in allowed else "other"

    return program, page_type
