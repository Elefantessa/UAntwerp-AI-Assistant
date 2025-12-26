"""
Link Hygiene Module
===================

Advanced URL normalization, cleaning, and HTML link rewriting utilities.
This module provides comprehensive link hygiene for the scraper stage,
including redirector unwrapping, tracking parameter removal, and
optional href suppression for token budget optimization.

Features:
- Resolves relative links against the page URL
- Unwraps common redirectors (facebook, google, t.co, lnkd.in)
- Strips tracking params (utm_*, gclid, fbclid, etc.) with an allow-list
- Percent-decodes/normalizes paths; collapses //; removes default ports
- Rewrites <a href> to the normalized URL
- Optional: shorten only the *display label* when it is a raw URL or too long
- Optional: suppress href in text — replace long hrefs with short placeholders
"""

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode, unquote

from bs4 import BeautifulSoup

from .config import (
    REDIRECTOR_DOMAINS,
    TRACKING_PREFIXES,
    TRACKING_KEYS,
    DEFAULT_ALLOW_PARAMS,
    DEFAULT_MAX_LABEL_LEN,
    DEFAULT_HREF_PLACEHOLDER_BASE,
)


# ---------------- Data Classes ----------------

@dataclass
class LinkInfo:
    """Metadata about a processed link."""
    href_full: str
    href_normalized: str
    placeholder: str
    suppressed_href: bool
    domain: str
    was_redirector: bool
    stripped_tracking: bool
    decoded: bool
    removed_fragment: bool
    orig_len: int
    norm_len: int
    anchor_text_old: str
    anchor_text_new: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ---------------- Core URL Helpers ----------------

def unwrap_redirector(url: str) -> Tuple[str, bool]:
    """
    Unwrap common redirector URLs (Facebook, Google, t.co, LinkedIn).

    Args:
        url: The URL to potentially unwrap

    Returns:
        Tuple of (unwrapped_url, was_redirector)
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    if any(host.endswith(d) for d in REDIRECTOR_DOMAINS) and ("url=" in url or "q=" in url):
        qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
        target = qs.get("url") or qs.get("q")
        if target and target.startswith(("http://", "https://")):
            return target, True

    return url, False


def strip_tracking_params(
    qs_items: List[Tuple[str, str]],
    allow_params: Tuple[str, ...] = DEFAULT_ALLOW_PARAMS
) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Remove tracking parameters from query string items.

    Args:
        qs_items: List of (key, value) tuples from query string
        allow_params: Tuple of parameter names to always keep

    Returns:
        Tuple of (filtered_items, was_stripped)
    """
    out: List[Tuple[str, str]] = []
    allow_lower = {a.lower() for a in allow_params or ()}
    stripped = False

    for k, v in qs_items:
        lk = k.lower()
        # Check if it's a tracking param
        if any(lk.startswith(p) for p in TRACKING_PREFIXES) or lk in TRACKING_KEYS:
            stripped = True
            continue
        out.append((k, v))

    return out, stripped


def remove_default_port(netloc: str) -> str:
    """
    Remove default port numbers (80 for HTTP, 443 for HTTPS).

    Args:
        netloc: Network location string (host:port)

    Returns:
        Network location without default ports
    """
    if netloc.endswith(":80"):
        return netloc[:-3]
    if netloc.endswith(":443"):
        return netloc[:-4]
    return netloc


def normalize_url_advanced(
    url: str,
    drop_fragments: bool = True,
    allow_params: Tuple[str, ...] = DEFAULT_ALLOW_PARAMS
) -> Tuple[str, Dict[str, bool]]:
    """
    Advanced URL normalization with full link hygiene.

    This function performs comprehensive URL normalization including:
    - Unwrapping redirectors (Facebook, Google, etc.)
    - Stripping tracking parameters
    - Percent-decoding paths
    - Removing default ports
    - Collapsing multiple slashes
    - Optionally dropping fragments

    Args:
        url: The URL to normalize
        drop_fragments: Whether to remove URL fragments (#section)
        allow_params: Query parameters to preserve

    Returns:
        Tuple of (normalized_url, flags_dict)
        flags_dict contains: was_redirector, stripped_tracking, decoded, removed_fragment
    """
    info = {
        "was_redirector": False,
        "stripped_tracking": False,
        "decoded": False,
        "removed_fragment": False
    }

    if not url or not url.startswith(("http://", "https://")):
        return url, info

    # Unwrap redirectors
    url, was_redirector = unwrap_redirector(url)
    info["was_redirector"] = was_redirector

    parsed = urlparse(url)

    # Percent-decode path (once)
    path_dec = unquote(parsed.path)
    if path_dec != parsed.path:
        info["decoded"] = True

    # Strip tracking parameters
    qs_items = parse_qsl(parsed.query, keep_blank_values=True)
    qs_filtered, was_stripped = strip_tracking_params(qs_items, allow_params)
    info["stripped_tracking"] = was_stripped
    query = urlencode(qs_filtered, doseq=True)

    # Drop fragment optionally
    fragment = "" if drop_fragments else (parsed.fragment or "")
    info["removed_fragment"] = bool(parsed.fragment) and drop_fragments

    # Normalize scheme/host, remove default ports, collapse //
    scheme = (parsed.scheme or "http").lower()
    netloc = remove_default_port((parsed.netloc or "").lower())
    path = re.sub(r"/{2,}", "/", path_dec)

    normalized = urlunparse((scheme, netloc, path, "", query, fragment))
    return normalized, info


def make_short_label(url: str, maxlen: int = DEFAULT_MAX_LABEL_LEN) -> str:
    """
    Create a shortened display label for a URL.

    Args:
        url: The full URL
        maxlen: Maximum length for the label

    Returns:
        Shortened label string
    """
    try:
        parsed = urlparse(url)
        base = (parsed.netloc or "").lower() + (parsed.path or "")
        if not base:
            base = url
        return (base[: maxlen - 1] + "…") if len(base) > maxlen else base
    except Exception:
        return (url[: maxlen - 1] + "…") if len(url) > maxlen else url


# ---------------- HTML Rewriting ----------------

def rewrite_html_links(
    html: str,
    page_url: str,
    *,
    rewrite_labels: bool = True,
    max_label_len: int = DEFAULT_MAX_LABEL_LEN,
    drop_fragments: bool = True,
    allow_params: Tuple[str, ...] = DEFAULT_ALLOW_PARAMS,
    suppress_href_in_text: bool = False,
    href_placeholder_base: str = DEFAULT_HREF_PLACEHOLDER_BASE,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rewrite anchor hrefs in HTML and optionally their display labels.

    If suppress_href_in_text=True, replaces the href with a short placeholder
    (e.g., #ref-1) and stores the original normalized href in data-orig-href.
    This prevents huge URLs from blowing up token budgets when HTML is
    converted to Markdown.

    Args:
        html: HTML content to process
        page_url: Base URL of the page (for resolving relative links)
        rewrite_labels: Whether to shorten display labels
        max_label_len: Maximum length for display labels
        drop_fragments: Whether to remove URL fragments
        allow_params: Query parameters to preserve
        suppress_href_in_text: Whether to use placeholders instead of full URLs
        href_placeholder_base: Prefix for placeholder hrefs

    Returns:
        Tuple of (clean_html, links_list, page_metrics)
    """
    soup = BeautifulSoup(html or "", "lxml") if html else BeautifulSoup("", "lxml")

    links: List[Dict[str, Any]] = []
    counts = Counter()
    placeholder_idx = 1

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue

        # Resolve relative against page_url
        abs_url = urljoin(page_url or "", href)
        if not abs_url.startswith(("http://", "https://")):
            # Keep mailto:, tel:, javascript:, data: as-is (but don't record)
            continue

        orig_len = len(abs_url)
        norm, flags = normalize_url_advanced(
            abs_url,
            drop_fragments=drop_fragments,
            allow_params=allow_params
        )
        norm_len = len(norm)

        # Optional label rewrite — only if label is missing or is itself a (raw) URL or very long
        text_old = (a.get_text(" ", strip=True) or "")
        text_new = text_old
        if rewrite_labels:
            if (not text_old) or text_old.startswith(("http://", "https://")) or len(text_old) > max_label_len:
                text_new = make_short_label(norm, maxlen=max_label_len)
                if text_new != text_old:
                    a.string = text_new

        suppressed = False
        placeholder = ""
        if suppress_href_in_text:
            placeholder = f"{href_placeholder_base}{placeholder_idx}"
            a["data-orig-href"] = norm
            a["href"] = placeholder
            suppressed = True
            placeholder_idx += 1
        else:
            a["href"] = norm

        # Collect link info
        domain = urlparse(norm).netloc.lower()
        info = LinkInfo(
            href_full=abs_url,
            href_normalized=norm,
            placeholder=placeholder,
            suppressed_href=suppressed,
            domain=domain,
            was_redirector=flags["was_redirector"],
            stripped_tracking=flags["stripped_tracking"],
            decoded=flags["decoded"],
            removed_fragment=flags["removed_fragment"],
            orig_len=orig_len,
            norm_len=norm_len,
            anchor_text_old=text_old,
            anchor_text_new=text_new,
        )
        links.append(info.to_dict())

        # Metrics
        counts["total"] += 1
        counts["pct_encoded"] += int("%" in abs_url)
        counts["was_redirector"] += int(flags["was_redirector"])
        counts["stripped_tracking"] += int(flags["stripped_tracking"])
        counts["suppressed_href"] += int(suppressed)
        counts["len_ge_200_before"] += int(orig_len >= 200)
        counts["len_ge_200_after"] += int(norm_len >= 200)

    # Page-level metrics
    metrics = {
        "total_links": counts["total"],
        "share_pct_encoded_before": (counts["pct_encoded"] / counts["total"]) if counts["total"] else 0.0,
        "share_redirector_unwrapped": (counts["was_redirector"] / counts["total"]) if counts["total"] else 0.0,
        "share_tracking_stripped": (counts["stripped_tracking"] / counts["total"]) if counts["total"] else 0.0,
        "share_len_ge_200_before": (counts["len_ge_200_before"] / counts["total"]) if counts["total"] else 0.0,
        "share_len_ge_200_after": (counts["len_ge_200_after"] / counts["total"]) if counts["total"] else 0.0,
        "share_suppressed_href": (counts["suppressed_href"] / counts["total"]) if counts["total"] else 0.0,
    }

    return str(soup), links, metrics


# ---------------- Processor Class ----------------

class LinkHygieneProcessor:
    """
    High-level processor for cleaning links in scraped page data.

    This class wraps the link hygiene functions for easy integration
    with the scraper pipeline.
    """

    def __init__(
        self,
        rewrite_labels: bool = True,
        max_label_len: int = DEFAULT_MAX_LABEL_LEN,
        drop_fragments: bool = True,
        allow_params: Tuple[str, ...] = DEFAULT_ALLOW_PARAMS,
        suppress_href_in_text: bool = False,
        href_placeholder_base: str = DEFAULT_HREF_PLACEHOLDER_BASE,
    ):
        """
        Initialize the processor with configuration options.

        Args:
            rewrite_labels: Whether to shorten display labels
            max_label_len: Maximum length for display labels
            drop_fragments: Whether to remove URL fragments
            allow_params: Query parameters to preserve
            suppress_href_in_text: Whether to use placeholders instead of full URLs
            href_placeholder_base: Prefix for placeholder hrefs
        """
        self.rewrite_labels = rewrite_labels
        self.max_label_len = max_label_len
        self.drop_fragments = drop_fragments
        self.allow_params = allow_params
        self.suppress_href_in_text = suppress_href_in_text
        self.href_placeholder_base = href_placeholder_base

    def process_page(
        self,
        html: str,
        page_url: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single page's HTML, cleaning all links.

        Args:
            html: HTML content to process
            page_url: URL of the page (for resolving relative links)

        Returns:
            Tuple of (clean_html, links_list, page_metrics)
        """
        return rewrite_html_links(
            html,
            page_url,
            rewrite_labels=self.rewrite_labels,
            max_label_len=self.max_label_len,
            drop_fragments=self.drop_fragments,
            allow_params=self.allow_params,
            suppress_href_in_text=self.suppress_href_in_text,
            href_placeholder_base=self.href_placeholder_base,
        )

    def process_pages(
        self,
        pages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process multiple pages, cleaning links in each.

        Args:
            pages: List of page dictionaries with 'url' and 'html_clean'/'html_content'/'html' keys

        Returns:
            Tuple of (processed_pages, aggregate_metrics)
        """
        out_pages: List[Dict[str, Any]] = []
        agg = Counter()
        per_domain: Counter = Counter()

        for p in pages:
            url = p.get("url") or p.get("metadata", {}).get("normalized_url") or ""
            html = p.get("html_clean") or p.get("html_content") or p.get("html") or ""

            clean_html, links, m = self.process_page(html, url)

            # Update page
            p_out = dict(p)
            p_out["html_clean"] = clean_html
            p_out["links"] = links
            p_out.setdefault("metadata", {})["url_hygiene"] = m
            out_pages.append(p_out)

            # Aggregate
            agg["pages"] += 1
            agg["links_total"] += m["total_links"]
            agg["links_len_ge_200_before"] += int(m["share_len_ge_200_before"] * m["total_links"]) if m["total_links"] else 0
            agg["links_len_ge_200_after"] += int(m["share_len_ge_200_after"] * m["total_links"]) if m["total_links"] else 0
            agg["links_suppressed"] += int(m["share_suppressed_href"] * m["total_links"]) if m["total_links"] else 0

            for li in links:
                per_domain[li["domain"]] += 1

        # Aggregate metrics
        aggregate_metrics = {
            "pages": agg["pages"],
            "links_total": agg["links_total"],
            "share_len_ge_200_before": (agg["links_len_ge_200_before"] / agg["links_total"]) if agg["links_total"] else 0.0,
            "share_len_ge_200_after": (agg["links_len_ge_200_after"] / agg["links_total"]) if agg["links_total"] else 0.0,
            "share_suppressed_href": (agg["links_suppressed"] / agg["links_total"]) if agg["links_total"] else 0.0,
            "top_domains": sorted(per_domain.items(), key=lambda kv: -kv[1])[:20],
            "config": {
                "rewrite_labels": self.rewrite_labels,
                "max_label_len": self.max_label_len,
                "drop_fragments": self.drop_fragments,
                "allow_params": list(self.allow_params),
                "suppress_href_in_text": self.suppress_href_in_text,
                "href_placeholder_base": self.href_placeholder_base,
            },
        }

        return out_pages, aggregate_metrics
