"""
Text Utilities for Chunking
============================

Section parsing, heading normalization, and text consolidation.
"""

import re
from typing import Dict, List, Any, Optional

from bs4 import BeautifulSoup


def words_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def has_md_headings(md: str) -> bool:
    """Check if markdown has headings."""
    return bool(re.search(r'^#{1,6}\s', md, re.MULTILINE))


def make_soup(html: str) -> BeautifulSoup:
    """Create BeautifulSoup from HTML."""
    return BeautifulSoup(html or "", "lxml")


def inject_headings_from_html(html_clean: str, md: str) -> str:
    """Inject headings from HTML into markdown if missing."""
    if has_md_headings(md):
        return md
    soup = make_soup(html_clean)
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        txt = h.get_text(" ", strip=True)
        if txt:
            level = int(h.name[1])
            md = f"{'#'*level} {txt}\n\n" + md
    return md


def normalize_headings(md: str) -> str:
    """Normalize heading levels to start from H1."""
    levels = []
    for line in md.splitlines():
        m = re.match(r'^(#{1,6})\s', line)
        if m:
            levels.append(len(m.group(1)))

    if not levels:
        return md

    min_level = min(levels)
    if min_level == 1:
        return md

    shift = min_level - 1

    def fix(line: str) -> str:
        m = re.match(r'^(#{1,6})\s(.*)$', line)
        if not m:
            return line
        old = len(m.group(1))
        new = max(1, old - shift)
        return f"{'#'*new} {m.group(2)}"

    return "\n".join(fix(l) for l in md.splitlines())


def parse_md_sections(md: str) -> List[Dict[str, Any]]:
    """
    Parse markdown into sections based on headings.

    Returns list of dicts with keys: level, heading, body
    """
    lines = md.splitlines()
    sections = []
    current = None

    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            if current:
                current["body"] = "\n".join(current["body_lines"]).strip()
                del current["body_lines"]
                sections.append(current)

            level = len(m.group(1))
            heading = m.group(2).strip()
            current = {"level": level, "heading": heading, "body_lines": []}
        elif current:
            current["body_lines"].append(line)
        else:
            # Content before any heading goes to level 0
            if not sections or sections[-1]["level"] != 0:
                current = {"level": 0, "heading": "", "body_lines": [line]}
            else:
                sections[-1]["body_lines"].append(line)

    if current:
        current["body"] = "\n".join(current.get("body_lines", [])).strip()
        if "body_lines" in current:
            del current["body_lines"]
        sections.append(current)

    return sections


def parent_key_for_section(sec: Dict[str, Any], level: int) -> str:
    """Generate parent key for section identification."""
    return f"L{level}:{sec.get('heading', '')[:20]}"


def consolidate_sections(
    sections: List[Dict[str, Any]],
    min_section_words: int,
    parent_level: int
) -> List[Dict[str, Any]]:
    """
    Consolidate small sections under parent headings.

    Args:
        sections: List of section dicts
        min_section_words: Minimum words for a section to stand alone
        parent_level: Heading level to consolidate under

    Returns:
        Consolidated sections
    """
    if not sections:
        return []

    result = []
    buffer = []
    buffer_words = 0

    for sec in sections:
        sec_words = words_count(sec.get("body", ""))

        if sec["level"] <= parent_level or sec_words >= min_section_words:
            # Flush buffer
            if buffer:
                merged_body = "\n\n".join(s.get("body", "") for s in buffer if s.get("body"))
                if result and result[-1]["level"] <= parent_level:
                    result[-1]["body"] = result[-1].get("body", "") + "\n\n" + merged_body
                else:
                    result.append({
                        "level": parent_level + 1,
                        "heading": "",
                        "body": merged_body,
                    })
                buffer = []
                buffer_words = 0
            result.append(sec)
        else:
            buffer.append(sec)
            buffer_words += sec_words

    # Flush remaining buffer
    if buffer:
        merged_body = "\n\n".join(s.get("body", "") for s in buffer if s.get("body"))
        if result:
            result[-1]["body"] = result[-1].get("body", "") + "\n\n" + merged_body
        else:
            result.append({
                "level": 1,
                "heading": "",
                "body": merged_body,
            })

    return result


def effective_min_section_words(target: int, base_min: int) -> int:
    """Calculate effective minimum section words based on target."""
    return max(base_min, target // 3)


def consolidate_with_escalation(
    sections: List[Dict[str, Any]],
    base_min_words: int,
    parent_level: int,
    target_for_page: int
) -> List[Dict[str, Any]]:
    """
    Consolidate sections with escalating parent levels.

    Tries progressively higher parent levels until sections meet minimum.
    """
    def ok(b):
        return all(words_count(s.get("body", "")) >= base_min_words for s in b)

    result = sections
    for level in range(parent_level, 0, -1):
        result = consolidate_sections(result, base_min_words, level)
        if ok(result):
            break

    return result
