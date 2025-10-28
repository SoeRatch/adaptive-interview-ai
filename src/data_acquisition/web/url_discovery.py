# src/data_acquisition/web/url_discovery.py
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse
from pathlib import Path

from src.data_acquisition.web.web_utils import (
    is_valid_url, normalize_url, is_html_link, can_fetch
)

from src.data_acquisition.web.web_constants import (
    SEED_URLS, KEYWORDS, RELEVANT_KEYWORDS,
    URL_DISCOVERY_OUTPUT, MAX_LINKS_PER_SOURCE,
    REQUEST_TIMEOUT, USER_AGENT
)

# Set the proper output path
RAW_WEB_DIR = Path(__file__).parents[3] / "data" / "raw" / "web"
RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RAW_WEB_DIR / URL_DISCOVERY_OUTPUT

class URLDiscovery:
    """Generic reusable crawler for discovering topic-related URLs."""

    def __init__(self, seed_urls=None, keywords=None, relevant_keywords=None):
        self.seed_urls = seed_urls or SEED_URLS
        self.keywords = [kw.lower() for kw in (keywords or KEYWORDS)]
        self.relevant_keywords = [kw.lower() for kw in (relevant_keywords or RELEVANT_KEYWORDS)]

    def contains_keyword(self, url: str, anchor_text: str = "") -> bool:
        """Check if either the URL or its anchor text matches topic keywords."""
        url_lower = url.lower()
        text_lower = anchor_text.lower()
        return (
            any(kw in url_lower for kw in self.keywords) or
            any(kw in text_lower for kw in self.relevant_keywords)
        )

    def scrape_links(self, seed_url: str):
        """Scrape relevant links from one seed page."""
        if not can_fetch(seed_url):
            print(f"[SKIP] Disallowed by robots.txt: {seed_url}")
            return []

        try:
            headers = {"User-Agent": USER_AGENT}
            res = requests.get(seed_url, headers=headers, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Could not fetch {seed_url}: {e}")
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        results = []

        for a_tag in soup.find_all("a", href=True):
            href = normalize_url(seed_url, a_tag["href"])
            text = a_tag.get_text(strip=True) or "N/A"

            if not is_valid_url(href) or not is_html_link(href):
                continue

            if self.contains_keyword(href, text):
                results.append({
                    "title": text[:150],
                    "url": href,
                    "source": urlparse(seed_url).netloc
                })

        return results

    def crawl_sources(self, max_links_per_source=MAX_LINKS_PER_SOURCE):
        """Run the crawler across all seed URLs."""
        all_links = []
        for seed in self.seed_urls:
            print(f"Scraping: {seed}")
            links = self.scrape_links(seed)
            print(f" -> Found {len(links)} relevant links")
            all_links.extend(links[:max_links_per_source])
            time.sleep(random.uniform(1.5, 3.5))  # polite delay
        return all_links

    def save_to_csv(self, results, output_file=OUTPUT_PATH):
        df = pd.DataFrame(results).drop_duplicates(subset=["url"])
        df.to_csv(output_file, index=False)

        domain_counts = Counter(df["source"])
        print(f"\n✅ Saved {len(df)} unique links to {output_file}")
        print("Links per source:")
        for domain, count in domain_counts.items():
            print(f"   {domain}: {count}")

if __name__ == "__main__":
    crawler = URLDiscovery()
    results = crawler.crawl_sources()
    crawler.save_to_csv(results)


# Can reuse the same class for another topic like “machine learning system design”:
# <
# from src.data_acquisition.web.url_discovery import URLDiscovery

# ml_keywords = ["machine-learning", "ai", "mlops", "model-serving"]
# ml_seeds = ["https://www.oreilly.com/ai/", "https://towardsdatascience.com/"]

# crawler = URLDiscovery(seed_urls=ml_seeds, keywords=ml_keywords)
# results = crawler.crawl_sources()
# crawler.save_to_csv(results, output_file=RAW_WEB_DIR / "ml_system_design_sources.csv")
# >


# Run it like this - 
# python -m src.data_acquisition.web.url_discovery
