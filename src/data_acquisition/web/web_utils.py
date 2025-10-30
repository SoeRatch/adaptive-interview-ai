# src/data_acquisition/web/web_utils.py
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser
import requests
from urllib.parse import urlparse

from src.data_acquisition.web.web_constants import (
    REQUEST_TIMEOUT
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
        res = requests.get(domain, timeout=REQUEST_TIMEOUT)
        if res.status_code >= 400:
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

def is_github_repo(url: str) -> bool:
    parsed = urlparse(url)
    return "github.com" in parsed.netloc and re.match(r"^/[^/]+/[^/]+/?$", parsed.path)