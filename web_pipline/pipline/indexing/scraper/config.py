"""
Scraper Configuration
=====================

URL filtering patterns and HTML cleaning configuration for UAntwerp web scraper.
"""

# -----------------------
# URL Filtering (tunable)
# -----------------------
DOMAIN = "www.uantwerpen.be"

# Allowlist patterns relevant to Master of CS programme pages
ALLOW_PATTERNS = (
    "/en/study/programmes/all-programmes/master-",
    "/en/study/programmes/all-programmes/master-data-science",
    "/en/study/programmes/all-programmes/master-software-engineering",
    "/en/study/programmes/all-programmes/master-computer-networks",
    "/en/study/programmes/all-programmes/master-computernetworks",  # legacy slug
)

# Disallow common noise (expand as needed)
BLOCK_PATTERNS = (
    "/calendar/",
    "/jobs/",
    "/staff/",
    "/library/",
    "/research/",
    "/life-in-antwerp/",
    "/about-uantwerp/",
    "/privacy-policy",
    "/cookie-policy",
    "/terms-of-use",
    "/search/",
    "javascript:",
)

# Query parameters to keep globally
QUERY_WHITELIST = {"lang"}


# -----------------------
# HTML Cleaning Config
# -----------------------

# Prefer strict main-content area; avoid selecting the outer page wrapper
MAIN_CONTENT_SELECTORS = [
    ".managedContent.singleMain",
    ".managedContent",
    "#content.pageMain",
    ".pageMain#content",
    "#content",
    ".pageMain",
    "article.wrap",
    "main",
]

# Lowercase noise markers; compared on lowercase attributes
NOISE_PATTERNS = (
    "cookie", "consent", "overlay", "modal", "newsletter",
    "cmpcta", "cmpoverlay", "cmpcookie", "cmpsearch",
    "sitefooter", "siteheader",
    # general site-specific noise observed in UAntwerp pages
    "keepintouch", "navsection", "navbreadcrumb", "navlinklist",
    "pageaside", "breadcrumb",
    "video", "player", "thankyoumessage", "errormessage",
)

# Tags to remove entirely with their content early
HARD_REMOVE_TAGS = {"script", "style", "noscript", "iframe", "form"}


# -----------------------
# Scraper Settings
# -----------------------
DEFAULT_MAX_PAGES_PER_SEED = 100
DEFAULT_MAX_DEPTH = 2
DEFAULT_CONCURRENCY = 8
DEFAULT_USER_AGENT = "UAntwerp-MCS-RAG-Scraper/1.2 (+research; contact: student@uantwerpen.be)"


# -----------------------
# Link Hygiene Config
# -----------------------

# Domains that are commonly used as redirectors (will be unwrapped)
REDIRECTOR_DOMAINS = (
    "l.facebook.com",
    "lm.facebook.com",
    "lnkd.in",
    "t.co",
    "google.com",
    "www.google.com",
)

# Tracking parameter prefixes to strip
TRACKING_PREFIXES = ("utm_",)

# Specific tracking parameter keys to strip
TRACKING_KEYS = {"gclid", "fbclid", "igshid", "mc_eid", "mc_cid", "si"}

# Default allowed query parameters (kept after normalization)
DEFAULT_ALLOW_PARAMS = ("id", "q", "lang")

# Default max length for shortened link labels
DEFAULT_MAX_LABEL_LEN = 80

# Default placeholder base for suppressed hrefs
DEFAULT_HREF_PLACEHOLDER_BASE = "#ref-"
