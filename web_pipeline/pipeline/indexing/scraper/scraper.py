"""
Web Scraper
===========

Async web scraper for UAntwerp pages with URL filtering,
robots.txt compliance, and content deduplication.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from bs4 import BeautifulSoup

try:
    from urllib.robotparser import RobotFileParser
except ImportError:
    RobotFileParser = None  # type: ignore

from .config import (
    DOMAIN,
    DEFAULT_MAX_PAGES_PER_SEED,
    DEFAULT_MAX_DEPTH,
    DEFAULT_CONCURRENCY,
    DEFAULT_USER_AGENT,
)
from .url_utils import normalize_url, is_allowed_url, extract_links, sha256_hex, infer_program_and_page_type
from .html_cleaner import (
    clean_html_keep_structure,
    extract_active_study_programme,
    extract_breadcrumbs,
)
from .markdown_converter import html_to_markdown, remove_personalpage_urls_from_markdown
from .link_hygiene import rewrite_html_links
from .config import (
    DEFAULT_ALLOW_PARAMS,
    DEFAULT_MAX_LABEL_LEN,
    DEFAULT_HREF_PLACEHOLDER_BASE,
)

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result of scraping a single page."""
    url: str
    title: str
    html_content: str
    html_clean: str
    markdown: str
    metadata: Dict[str, str]
    scraped_at: str
    extraction_method: str
    content_hash: str
    source_type: str = "web"
    depth: int = 0
    seed_index: int = 0


async def fetch_robots(session, base_url: str):
    """Fetch and parse robots.txt."""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        async with session.get(robots_url, timeout=ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                text = await resp.text()
                rp.parse(text.splitlines())
                logger.info("Robots.txt loaded: %s", robots_url)
            else:
                rp.parse(["User-agent: *", "Disallow:"])
                logger.warning("Robots.txt unavailable (%s): allowing by default.", resp.status)
    except Exception as e:
        logger.warning("Failed to fetch robots.txt (%s): allowing by default.", e)
        rp.parse(["User-agent: *", "Disallow:"])
    return rp


class WebScraper:
    """
    Async web scraper for UAntwerp pages.

    Features:
    - URL filtering/normalization
    - Robots.txt compliance
    - Depth limits and concurrency control
    - Content-based deduplication per program
    - Per-seed page quotas
    - Dual outputs (HTML + Markdown)
    """

    def __init__(
        self,
        seeds: List[str],
        max_pages_per_seed: int = DEFAULT_MAX_PAGES_PER_SEED,
        max_depth: int = DEFAULT_MAX_DEPTH,
        concurrency: int = DEFAULT_CONCURRENCY,
        keep_raw_html: bool = False,
        force_study_programme_once: bool = True,
        force_other_essentials: bool = False,
        extract_links_from: str = "raw",
        # Link Hygiene options
        enable_link_hygiene: bool = True,
        rewrite_labels: bool = True,
        max_label_len: int = DEFAULT_MAX_LABEL_LEN,
        drop_fragments: bool = True,
        allow_params: Tuple[str, ...] = DEFAULT_ALLOW_PARAMS,
        suppress_href_in_text: bool = False,
        href_placeholder_base: str = DEFAULT_HREF_PLACEHOLDER_BASE,
    ):
        """
        Initialize web scraper.

        Args:
            seeds: List of seed URLs to start crawling from
            max_pages_per_seed: Maximum pages per seed URL
            max_depth: Maximum crawl depth
            concurrency: Number of concurrent requests
            keep_raw_html: Whether to keep raw HTML
            force_study_programme_once: Force visit study-programme once per seed
            force_other_essentials: Force visit other essential pages
            extract_links_from: Source for link extraction ("raw" or "clean")
            enable_link_hygiene: Enable link hygiene processing
            rewrite_labels: Shorten long link display labels
            max_label_len: Maximum length for link labels
            drop_fragments: Remove URL fragments (#section)
            allow_params: Query parameters to keep
            suppress_href_in_text: Replace hrefs with placeholders (#ref-1)
            href_placeholder_base: Prefix for placeholder hrefs
        """
        self.seeds = [normalize_url(s) for s in seeds if s]
        self.max_pages_per_seed = max_pages_per_seed
        self.max_depth = max_depth
        self.sem = asyncio.Semaphore(concurrency)
        self.keep_raw_html = keep_raw_html
        self.force_study_programme_once = force_study_programme_once
        self.force_other_essentials = force_other_essentials
        self.extract_links_from = extract_links_from.lower().strip()

        # Link Hygiene settings
        self.enable_link_hygiene = enable_link_hygiene
        self.rewrite_labels = rewrite_labels
        self.max_label_len = max_label_len
        self.drop_fragments = drop_fragments
        self.allow_params = allow_params
        self.suppress_href_in_text = suppress_href_in_text
        self.href_placeholder_base = href_placeholder_base

        # Queue for URLs to visit
        self.to_visit: deque = deque()

        # Tracking
        self.visited: Set[str] = set()
        self.failed_urls: List[str] = []
        self.blocked_urls: List[str] = []
        self.depth_distribution: Dict[int, int] = {}
        self.pages: List[PageResult] = []

        # Per-program deduplication
        self.content_hash_by_program: Set[Tuple[str, str]] = set()
        self.duplicate_map: Dict[str, List[str]] = {}

        # Per-seed tracking
        self.per_seed_counts: Dict[int, int] = {}
        self.per_seed_roots: Dict[int, str] = {}
        self.per_seed_study_prog_seen: Dict[int, bool] = {}

        # Link hygiene metrics
        self.link_hygiene_metrics: Dict[str, Any] = {}

        self._started = time.time()

    async def run(self, output_path: Path) -> None:
        """Run the scraper and save results to output path."""
        if not self.seeds:
            logger.error("No seeds provided.")
            return

        # Initialize per-seed tracking
        for idx, s in enumerate(self.seeds):
            self.per_seed_counts[idx] = 0
            self.per_seed_roots[idx] = s
            self.per_seed_study_prog_seen[idx] = False

            if is_allowed_url(s):
                self.to_visit.append((s, 0, idx))
            else:
                self.blocked_urls.append(s)

        timeout = ClientTimeout(total=30, connect=10, sock_read=20)
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            # Fetch robots.txt
            rp = await fetch_robots(session, self.seeds[0])

            # Force visit essential pages
            for idx, s in enumerate(self.seeds):
                if not is_allowed_url(s):
                    continue
                base = s if s.endswith("/") else s + "/"

                if self.force_study_programme_once:
                    forced_sp = normalize_url(urljoin(base, "study-programme/"))
                    if is_allowed_url(forced_sp):
                        self.to_visit.append((forced_sp, 1, idx))

                if self.force_other_essentials:
                    essentials = [
                        "profile/", "programme-info/", "admission-and-enrolment/",
                        "job-opportunities/", "contact/"
                    ]
                    for suf in essentials:
                        u = normalize_url(urljoin(base, suf))
                        if is_allowed_url(u):
                            self.to_visit.append((u, 1, idx))

            # Main crawl loop
            while self.to_visit:
                url, depth, seed_idx = self.to_visit.popleft()

                if self.per_seed_counts.get(seed_idx, 0) >= self.max_pages_per_seed:
                    continue

                # Robots check
                if not rp.can_fetch(headers["User-Agent"], url):
                    logger.info("Blocked by robots.txt: %s", url)
                    self.blocked_urls.append(url)
                    continue

                await self._fetch_and_process(session, url, depth, seed_idx)

                if all(self.per_seed_counts[i] >= self.max_pages_per_seed for i in self.per_seed_counts):
                    break

        self._write_output(output_path)

    async def _safe_get(self, session, url: str, attempt: int = 1, max_attempts: int = 3):
        """Safe GET request with retries."""
        try:
            async with self.sem:
                async with session.get(url, allow_redirects=True) as resp:
                    text = await resp.text(errors="ignore")
                    last_mod = resp.headers.get("Last-Modified", "")
                    final_url = str(resp.url)
                    return text, final_url, resp.status, last_mod
        except Exception:
            if attempt < max_attempts:
                await asyncio.sleep(0.8 * attempt)
                return await self._safe_get(session, url, attempt + 1, max_attempts)
            return "", url, 599, ""

    async def _fetch_and_process(self, session, url: str, depth: int, seed_idx: int) -> None:
        """Fetch and process a single URL."""
        if depth > self.max_depth:
            return
        if self.per_seed_counts.get(seed_idx, 0) >= self.max_pages_per_seed:
            return

        try:
            html, final_url, status, last_modified = await self._safe_get(session, url)
            if status != 200 or not html:
                logger.warning("Fetch failed [%s]: %s", status, url)
                self.failed_urls.append(url)
                return

            # Extract metadata before cleaning
            soup_raw = BeautifulSoup(html, "lxml")
            title = (soup_raw.title.get_text(strip=True) if soup_raw.title else "") or ""
            html_lang = soup_raw.html.get("lang", "").strip().lower() if soup_raw.html else ""
            canonical_link = soup_raw.find("link", rel=lambda v: v and "canonical" in v.lower())
            canonical = canonical_link["href"].strip() if canonical_link and canonical_link.has_attr("href") else ""
            if canonical:
                canonical = urljoin(final_url or url, canonical)
            breadcrumbs = extract_breadcrumbs(soup_raw)

            # Determine canonical URL key
            key_url = canonical if (canonical and urlparse(canonical).netloc == DOMAIN) else (final_url or url)
            url_key = normalize_url(key_url)

            # Skip if already visited
            if url_key in self.visited:
                return
            self.visited.add(url_key)

            # Study-programme handling
            page_path = urlparse(final_url or url).path
            if "/study-programme/" in urlparse(url_key).path and self.per_seed_study_prog_seen.get(seed_idx, False):
                logger.info("Study-programme already visited for seed %d, skipping: %s", seed_idx, url_key)
                return

            # Extract active study programme if applicable
            sliced_html = html
            if "/study-programme/" in page_path:
                sliced_html = extract_active_study_programme(html)

            # Clean and convert
            html_clean = clean_html_keep_structure(sliced_html)

            # Apply Link Hygiene if enabled
            page_links = []
            link_metrics = {}
            if self.enable_link_hygiene:
                html_clean, page_links, link_metrics = rewrite_html_links(
                    html_clean,
                    final_url or url,
                    rewrite_labels=self.rewrite_labels,
                    max_label_len=self.max_label_len,
                    drop_fragments=self.drop_fragments,
                    allow_params=self.allow_params,
                    suppress_href_in_text=self.suppress_href_in_text,
                    href_placeholder_base=self.href_placeholder_base,
                )

            markdown = html_to_markdown(html_clean)
            if not markdown:
                markdown = BeautifulSoup(html_clean, "lxml").get_text("\n", strip=True) + "\n"
            else:
                markdown = remove_personalpage_urls_from_markdown(markdown)

            # Deduplicate by content hash per program
            path_only = urlparse(final_url or url).path
            program_code, page_type = infer_program_and_page_type(path_only)
            hash_val = sha256_hex(html_clean)
            hash_key = (hash_val, program_code)
            if hash_key in self.content_hash_by_program:
                logger.info("Duplicate content within same program skipped: %s", url_key)
                self.duplicate_map.setdefault(hash_val, []).append(url_key)
                return
            self.content_hash_by_program.add(hash_key)

            # Update study-programme flag
            if "/study-programme/" in urlparse(url_key).path:
                self.per_seed_study_prog_seen[seed_idx] = True

            # Compose metadata
            metadata = {
                "url": final_url or url,
                "normalized_url": url_key,
                "domain": urlparse(final_url or url).netloc,
                "path": path_only,
                "title": title,
                "lang": html_lang or "en",
                "canonical": canonical,
                "breadcrumbs": " > ".join(breadcrumbs) if breadcrumbs else "",
                "last_modified": last_modified or "",
                "seed_index": seed_idx,
                "seed_root": self.per_seed_roots.get(seed_idx, ""),
                "program": program_code,
                "page_type": page_type,
            }

            # Add link hygiene info to metadata if enabled
            if self.enable_link_hygiene:
                metadata["link_hygiene"] = link_metrics
                metadata["links"] = page_links

            # Build PageResult
            page = PageResult(
                url=final_url or url,
                title=title,
                html_content=html_clean,
                html_clean=html_clean,
                markdown=markdown,
                metadata=metadata,
                scraped_at=datetime.utcnow().isoformat(),
                extraction_method="aiohttp-html",
                content_hash=hash_val,
                depth=depth,
                seed_index=seed_idx,
            )
            self.pages.append(page)
            self.depth_distribution[depth] = self.depth_distribution.get(depth, 0) + 1

            # Update counters
            self.per_seed_counts[seed_idx] = self.per_seed_counts.get(seed_idx, 0) + 1

            logger.info("Fetched [%d for seed %d/%d] %s (depth=%d)",
                       self.per_seed_counts[seed_idx], seed_idx, self.max_pages_per_seed,
                       page.metadata["normalized_url"], depth)

            # Enqueue child links
            if self.per_seed_counts[seed_idx] < self.max_pages_per_seed and depth < self.max_depth:
                if "/study-programme/" in page_path:
                    link_source = sliced_html if self.extract_links_from == "raw" else html_clean
                else:
                    link_source = html if self.extract_links_from == "raw" else html_clean
                child_links = extract_links(page.metadata["normalized_url"], link_source)
                for link in child_links:
                    if "/study-programme/" in urlparse(link).path and self.per_seed_study_prog_seen.get(seed_idx, False):
                        continue
                    if is_allowed_url(link):
                        self.to_visit.append((link, depth + 1, seed_idx))

        except Exception as e:
            logger.exception("Error processing %s: %s", url, e)
            self.failed_urls.append(url)

    def _write_output(self, output_path: Path) -> None:
        """Write output to JSON file."""
        per_seed_summary = {
            str(idx): {
                "seed_root": self.per_seed_roots.get(idx, ""),
                "unique_pages": self.per_seed_counts.get(idx, 0),
            }
            for idx in sorted(self.per_seed_roots.keys())
        }

        out = {
            "scraped_at": datetime.utcnow().isoformat(),
            "total_pages": len(self.pages),
            "per_seed_summary": per_seed_summary,
            "failed_urls": self.failed_urls,
            "blocked_urls": self.blocked_urls,
            "duplicates": {h: urls for h, urls in self.duplicate_map.items()},
            "extraction_methods": {"aiohttp-html": len(self.pages)},
            "depth_distribution": {str(k): v for k, v in sorted(self.depth_distribution.items())},
            "pages": [
                {
                    "url": p.url,
                    "title": p.title,
                    "html_content": p.html_content,
                    "html_clean": p.html_clean,
                    "markdown": p.markdown,
                    "metadata": p.metadata,
                    "scraped_at": p.scraped_at,
                    "extraction_method": p.extraction_method,
                    "content_hash": p.content_hash,
                    "source_type": p.source_type,
                    "depth": p.depth,
                    "seed_index": p.seed_index,
                }
                for p in self.pages
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - self._started
        logger.info(
            "Saved %d unique pages to %s (%.1fs). Duplicates skipped: %d",
            len(self.pages),
            str(output_path),
            elapsed,
            sum(len(v) for v in self.duplicate_map.values()),
        )


def scrape_cli():
    """CLI entry point for scraping."""
    import sys

    # Default seeds
    seeds = [
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-data-science",
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-software-engineering",
        "https://www.uantwerpen.be/en/study/programmes/all-programmes/master-computer-networks",
    ]

    output = "/project_antwerp/web_pipline/data/raw/scraped_content.json"

    async def _run():
        scraper = WebScraper(
            seeds=seeds,
            max_pages_per_seed=100,
            max_depth=4,
            concurrency=8,
            force_study_programme_once=True,
            force_other_essentials=True,
        )
        await scraper.run(Path(output))

    asyncio.run(_run())


if __name__ == "__main__":
    scrape_cli()
