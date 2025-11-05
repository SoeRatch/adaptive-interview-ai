# src/data_acquisition/web/url_discovery.py
import time
import random
import json
import requests
from datetime import datetime, timezone
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse
from pathlib import Path

from src.data_acquisition.web.web_utils import (
    is_valid_url, normalize_url, is_html_link, can_fetch, resolve_redirect
)

from src.data_acquisition.web.web_constants import (
    SEED_URLS, KEYWORDS, RELEVANT_KEYWORDS,
    URL_PROGRESS_FILE, URL_DISCOVERY_OUTPUT, MAX_LINKS_PER_SOURCE,
    REQUEST_TIMEOUT, USER_AGENT
)

# Set the proper output path
RAW_WEB_DIR = Path(__file__).parents[3] / "data" / "raw" / "web"
RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)

class URLDiscovery:
    """Generic reusable crawler for discovering topic-related URLs."""

    def __init__(self, seed_urls=None, keywords=None, relevant_keywords=None,output_filename=URL_DISCOVERY_OUTPUT,progress_filename=URL_PROGRESS_FILE):
        self.seed_urls = seed_urls or SEED_URLS

        self.keywords = [kw.lower() for kw in (keywords or KEYWORDS)]
        self.relevant_keywords = [kw.lower() for kw in (relevant_keywords or RELEVANT_KEYWORDS)]
        
        self.output_path = RAW_WEB_DIR / output_filename
        self.progress_path = RAW_WEB_DIR / progress_filename

        self.results = []
        self.redirect_cache = {}
        self.progress = self._load_progress()

    def _load_progress(self):
        """Load progress file to skip completed seeds."""
        if self.progress_path.exists():
            with open(self.progress_path, "r") as f:
                return json.load(f)
        return {}

    
    def _save_progress(self, seed_url, status="success", links_count=0, error_msg=None):
        """Mark a seed as completed with metadata."""
        self.progress[seed_url] = {
            "status": status,
            "links": links_count,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "error": error_msg
        }
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)


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
        print(f"Allowed by robots.txt: {seed_url}")
        
        headers = {"User-Agent": USER_AGENT}

        for attempt in range(3):
            try:
                res = requests.get(seed_url, headers=headers, timeout=REQUEST_TIMEOUT)
                res.raise_for_status()
                break  # success
            except requests.exceptions.RequestException as e:
                wait_time = random.uniform(10 * (attempt + 1), 30 * (attempt + 1))
                print(f"[WARN] Attempt {attempt+1} failed: {e}")
                print(f"[BACKOFF] Waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time)
            except Exception as e:
                # Handle anything unexpected gracefully
                print(f"[ERROR] Unexpected error fetching {seed_url}: {e}")
                return []
        else:
            # If all retries fail - Only runs if loop completed all 3 attempts without a 'break'
            print(f"[SKIP] Giving up on {seed_url}")
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        results = []

        for a_tag in soup.find_all("a", href=True):
            href = normalize_url(seed_url, a_tag["href"])
            text = a_tag.get_text(strip=True) or "N/A"

            if not is_valid_url(href) or not is_html_link(href):
                continue

            # Resolve final canonical URL (lightweight HEAD)
            if href in self.redirect_cache:
                final_url = self.redirect_cache[href]
            else:
                final_url = resolve_redirect(href)
                self.redirect_cache[href] = final_url
            
            if not final_url:
                continue

            if self.contains_keyword(final_url, text):
                results.append({
                    "title": text[:150],
                    "url": final_url,
                    "source": urlparse(seed_url).netloc
                })

        return results
    
    def save_to_csv_incremental(self, results):
        """Append seed results to CSV immediately and dedupe."""
        # If expectation is under 10k links in total then current implementation is simple and safe.
        # But later if we expect >10k links,then switch to append mode and dedupe once at the end.
        df_new = pd.DataFrame(results)
        if df_new.empty:
            return 0
        
        # Write in append mode; only write header if file doesn't exist
        df_new.to_csv(
            self.output_path,
            mode="a",
            header=not self.output_path.exists(),
            index=False
        )

        return len(df_new)
    
    def deduplicate_csv(self):
        """Run this once after full crawl to remove duplicate URLs."""
        if not self.output_path.exists():
            print("[INFO] No output file to deduplicate.")
            return

        df = pd.read_csv(self.output_path)
        before = len(df)
        df = df.drop_duplicates(subset=["url"])
        after = len(df)
        df.to_csv(self.output_path, index=False)
        print(f"[CLEANUP] Removed {before - after} duplicate URLs. Final count: {after}")

    def crawl_sources(self, max_links_per_source=MAX_LINKS_PER_SOURCE):
        """Run the crawler across all seed URLs with resume support."""
        for seed in self.seed_urls:
            if seed in self.progress:
                print(f"[SKIP] Already completed: {seed}")
                continue
            print(f"\nScraping: {seed}")
            links = self.scrape_links(seed)
            if links:
                saved_count = self.save_to_csv_incremental(links[:max_links_per_source])
                print(f" -> Found {saved_count} relevant links")
                self._save_progress(seed, status="success", links_count=saved_count, error_msg=None)
            else:
                # mark as processed even if 0 links, but with status 'success_zero' or similar
                self._save_progress(seed, status="success_zero", links_count=0)

            # Polite delay between sources
            time.sleep(random.uniform(1.5, 3.5))
        
        # Deduplicate at the end once
        self.deduplicate_csv()

        print(f"\n✅ Completed all seed URLs.\nData saved to: {self.output_path}")

if __name__ == "__main__":
    crawler = URLDiscovery()
    results = crawler.crawl_sources()


# Can reuse the same class for another topic like “machine learning system design”:
# <
# ml_keywords = ["machine-learning", "ai", "mlops", "model-serving"]
# ml_seeds = ["https://www.oreilly.com/ai/", "https://towardsdatascience.com/"]

# crawler = URLDiscovery(
#     seed_urls=ml_seeds,
#     keywords=ml_keywords,
#     output_filename="ml_system_design_links.csv",
#     progress_filename="ml_system_design_progress.json"
# )
# crawler.crawl_sources()
# >


# Run it like this - 
# python -m src.data_acquisition.web.url_discovery
