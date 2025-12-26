# Scraper Module
"""Web scraping components with HTML cleaning and Markdown conversion."""

from .config import (
    DOMAIN,
    ALLOW_PATTERNS,
    BLOCK_PATTERNS,
    QUERY_WHITELIST,
    MAIN_CONTENT_SELECTORS,
    NOISE_PATTERNS,
    HARD_REMOVE_TAGS,
    # Link Hygiene Config
    REDIRECTOR_DOMAINS,
    TRACKING_PREFIXES,
    TRACKING_KEYS,
    DEFAULT_ALLOW_PARAMS,
    DEFAULT_MAX_LABEL_LEN,
    DEFAULT_HREF_PLACEHOLDER_BASE,
)
from .url_utils import normalize_url, is_allowed_url, extract_links
from .html_cleaner import HtmlCleaner, select_main_container, clean_html_keep_structure
from .markdown_converter import MarkdownConverter, html_to_markdown
from .link_hygiene import (
    LinkInfo,
    LinkHygieneProcessor,
    normalize_url_advanced,
    unwrap_redirector,
    strip_tracking_params,
    remove_default_port,
    make_short_label,
    rewrite_html_links,
)
from .scraper import WebScraper

__all__ = [
    # Config
    "DOMAIN",
    "ALLOW_PATTERNS",
    "BLOCK_PATTERNS",
    "QUERY_WHITELIST",
    "MAIN_CONTENT_SELECTORS",
    "NOISE_PATTERNS",
    "HARD_REMOVE_TAGS",
    # Link Hygiene Config
    "REDIRECTOR_DOMAINS",
    "TRACKING_PREFIXES",
    "TRACKING_KEYS",
    "DEFAULT_ALLOW_PARAMS",
    "DEFAULT_MAX_LABEL_LEN",
    "DEFAULT_HREF_PLACEHOLDER_BASE",
    # Utilities
    "normalize_url",
    "is_allowed_url",
    "extract_links",
    # Link Hygiene
    "LinkInfo",
    "LinkHygieneProcessor",
    "normalize_url_advanced",
    "unwrap_redirector",
    "strip_tracking_params",
    "remove_default_port",
    "make_short_label",
    "rewrite_html_links",
    # Cleaners
    "HtmlCleaner",
    "select_main_container",
    "clean_html_keep_structure",
    # Converters
    "MarkdownConverter",
    "html_to_markdown",
    # Main
    "WebScraper",
]

