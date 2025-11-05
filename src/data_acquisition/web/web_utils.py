# src/data_acquisition/web/web_utils.py
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from src.data_acquisition.web.web_constants import (
    REQUEST_TIMEOUT, VIDEO_DOMAINS, USER_AGENT
)

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def normalize_url(base_url: str, href: str) -> str:
    """Resolve relative links and strip trailing slashes."""
    joined = urljoin(base_url, href)
    u = urlparse(joined)
    return f"{u.scheme}://{u.netloc}{u.path}".rstrip("/")

def is_html_link(url: str) -> bool:
    non_html_ext = (".pdf", ".jpg", ".png", ".gif", ".zip", ".mp4")
    return not url.lower().endswith(non_html_ext)

def can_fetch(url: str) -> bool:
    """Check if crawling is allowed by robots.txt."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        headers = {"User-Agent": USER_AGENT}
        res = requests.get(domain, headers=headers, timeout=REQUEST_TIMEOUT)
        if res.status_code >= 400:
            # When you get a 4xx or 5xx error on /robots.txt, it means
            # the site doesn’t have a robots.txt file (404 Not Found), or
            # it is temporarily unavailable (500, 502, 503, etc.).
            # If robots.txt not accessible, assume allowed
            return True
        
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(res.text.splitlines())
        return rp.can_fetch("*", url)
    except requests.exceptions.Timeout:
        print(f"[WARN] robots.txt request timed out for {domain}, assuming allowed.")
        return True
    except Exception as e:
        print(f"[WARN] Error checking robots.txt for {domain}: {e}")
        return True
    
def is_medium_url(url: str) -> bool:
     return "medium.com" in urlparse(url).netloc

def is_video_url(url: str) -> bool:
     domain = urlparse(url).netloc
     return any(vd in domain for vd in VIDEO_DOMAINS)

def is_github_repo(url: str) -> bool:
    parsed = urlparse(url)
    return "github.com" in parsed.netloc and re.match(r"^/[^/]+/[^/]+/?$", parsed.path)
    
def resolve_redirect(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Resolve the final destination URL by following redirects.
    Uses a lightweight HEAD request first, falling back to GET if needed.
    Returns the final URL or None if it fails.
    """
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",  # prevents gzip overhead
            }
        res = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if res.status_code in (405, 403):  # HEAD not allowed, try GET
            res = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        final_url = res.url
        if final_url.rstrip('/') != url.rstrip('/'):
            print(f"[REDIRECT] {url} to {final_url}")
        return final_url.rstrip('/')
    except requests.RequestException:
        return None
    except Exception:
        return None
        

def remove_ui_noise(soup, debug=False):
    """
    Remove non-content UI elements like navbars, footers, ads, etc.
    Fully safe version that handles decomposed tags gracefully.
    """

    if soup is None:
        return soup

    # Step 1: remove globally safe tags (these never hold main content)
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "button","iframe"]):
        tag.decompose()

    # Step 2: remove noise *based on class/id*, but safely
    UI_NOISE_PATTERNS = [
        r"footer", r"navbar", r"advert", r"ads?", r"sponsor",
        r"subscribe", r"login", r"signup", r"comment", r"feedback",
        r"share", r"cookie", r"popup", r"banner", r"modal", r"promo"
    ]
    for t in soup.find_all(True):
        try:
            # Avoid removing structural containers
            if t.find(["article", "main", "section"]):
                continue

            classes = " ".join(t.get("class", []))
            cid = t.get("id", "")
            label = f"{classes} {cid}".strip()

            if not label:
                continue

            # If class/id looks like noise — remove it
            if any(re.search(p, label, re.I) for p in UI_NOISE_PATTERNS):
                # double-check it doesn’t contain headings or paragraphs
                if t.find(["h1", "h2", "h3", "p"]):
                    continue
                t.decompose()

        except Exception:
            continue

    return soup


def get_canonical_url(soup, fetched_url):
    link_tag = soup.find("link", rel="canonical")
    if link_tag and link_tag.get("href"):
        canonical = link_tag["href"].strip()
        if canonical.startswith("/"):
            # relative canonical URL — resolve it
            from urllib.parse import urljoin
            canonical = urljoin(fetched_url, canonical)
        return canonical.rstrip('/')
    return fetched_url.rstrip('/')