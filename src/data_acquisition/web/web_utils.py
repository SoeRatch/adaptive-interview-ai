# src/data_acquisition/web/web_utils.py
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser

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
    rp = urllib.robotparser.RobotFileParser()
    domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
    rp.set_url(domain)
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except Exception:
        return True  # assume allowed if robots.txt fails
    
def is_medium_url(url: str) -> bool:
     return "medium.com" in urlparse(url).netloc

def is_github_repo(url: str) -> bool:
    parsed = urlparse(url)
    return "github.com" in parsed.netloc and re.match(r"^/[^/]+/[^/]+/?$", parsed.path)