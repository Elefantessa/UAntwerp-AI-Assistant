"""
HTML Cleaner
============

HTML cleaning and structure-preserving content extraction.
"""

from typing import List, Optional
from bs4 import BeautifulSoup, Comment, Tag

from .config import MAIN_CONTENT_SELECTORS, NOISE_PATTERNS, HARD_REMOVE_TAGS


def _node_is_noisy(node: Tag) -> bool:
    """Check if a node should be removed based on attribute substrings (lowercased)."""
    attr_vals = " ".join(
        str(v) for k, v in node.attrs.items()
        if k in ("id", "class", "role", "data-cmp")
    ).lower()
    return any(key in attr_vals for key in NOISE_PATTERNS)


def extract_breadcrumbs(soup: BeautifulSoup) -> List[str]:
    """Attempt to extract breadcrumb-like structure (site-specific heuristics)."""
    crumbs: List[str] = []
    navs = soup.select("nav.navBreadcrumb ul li, .navBreadcrumb ul li")
    if not navs:
        navs = soup.select("ul.breadcrumb li, nav[aria-label*=breadcrumb] li")
    for li in navs:
        txt = li.get_text(" ", strip=True)
        if txt:
            crumbs.append(txt)
    out = []
    for c in crumbs:
        if c not in out:
            out.append(c)
    return out


def select_main_container(soup: BeautifulSoup) -> Tag:
    """
    Choose the most relevant main content container.
    Strategy:
      1) Try known CSS selectors (first match wins).
      2) Fallback to <body>.
    """
    for sel in MAIN_CONTENT_SELECTORS:
        found = soup.select(sel)
        if found:
            return found[0]
    return soup.body or soup


def clean_html_keep_structure(html: str) -> str:
    """
    Remove obvious noise but keep important structure.
    - Drop scripts/styles/forms/iframes/noscripts entirely.
    - Remove blocks whose id/class/role/data-cmp contain noise markers (lowercased).
    - Select a main-content container and return its HTML only (not the entire <body>).
    - Inside the main container, strip nav/aside/figure/svg/video/picture/source.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # Remove hard tags globally
    for tag in soup.find_all(HARD_REMOVE_TAGS):
        tag.decompose()

    # Remove noisy blocks globally by attributes
    for node in list(soup.find_all(True)):
        try:
            if _node_is_noisy(node):
                node.decompose()
        except Exception:
            continue

    # Select strict main content container
    container = select_main_container(soup)

    # Inside the main container, drop non-textual chrome
    for tag in container.find_all(["nav", "aside", "figure", "svg", "video", "source", "picture"]):
        tag.decompose()

    # Second pass for noisy attributes (container-level)
    for node in list(container.find_all(True)):
        try:
            if _node_is_noisy(node):
                node.decompose()
        except Exception:
            continue

    return str(container)


def extract_active_study_programme(html: str) -> str:
    """
    From a UAntwerp study-programme page, keep ONLY the active academic year pane (e.g., 2025-2026)
    and drop older years. The active tab is indicated by:
      <li class="stateActive" data-desttab="..."> in the header.tabHeader
    and a matching <section class="pane" id="..."> inside <div class="main tabMain">.
    If parsing fails, return the original HTML unchanged.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        active_li = soup.select_one("header.tabHeader li.stateActive[data-desttab]")
        if not active_li:
            return html
        dest_id = (active_li.get("data-desttab") or "").strip()
        if not dest_id:
            return html
        active_pane = soup.select_one(f"div.tabMain section.pane#{dest_id}")
        if not active_pane:
            # Fallback: sometimes the active pane itself has stateActive
            active_pane = soup.select_one("div.tabMain section.pane.stateActive")
            if not active_pane:
                return html
        container = soup.new_tag("div", **{"class": "study-programme-active"})
        heading = active_pane.find(["h3", "h2"], class_="heading")
        if heading:
            container.append(heading)
        pane_content = active_pane.select_one(".paneContent") or active_pane
        container.append(pane_content)
        return str(container)
    except Exception:
        return html


class HtmlCleaner:
    """
    High-level HTML cleaning interface.

    Provides methods for cleaning HTML while preserving structure.
    """

    def __init__(self, selectors: Optional[List[str]] = None):
        """
        Initialize HTML cleaner.

        Args:
            selectors: Custom main content selectors (optional)
        """
        self.selectors = selectors or MAIN_CONTENT_SELECTORS

    def clean(self, html: str) -> str:
        """Clean HTML and return main content."""
        return clean_html_keep_structure(html)

    def extract_study_programme(self, html: str) -> str:
        """Extract active study programme content."""
        return extract_active_study_programme(html)

    def get_breadcrumbs(self, html: str) -> List[str]:
        """Extract breadcrumbs from HTML."""
        soup = BeautifulSoup(html, "lxml")
        return extract_breadcrumbs(soup)
