# src/data_acquisition/web/web_scrapper.py
import os
import re
import json
import hashlib
import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
import time
import random
from tqdm import tqdm
from pathlib import Path

from src.data_acquisition.web.web_utils import (
    can_fetch, is_medium_url, is_github_repo
)

from src.data_acquisition.web.web_constants import (
    URL_DISCOVERY_OUTPUT, WEB_SCRAPPER_HISTORY,WEB_SCRAPER_OUTPUT, REQUEST_TIMEOUT, USER_AGENT
)

# Set the proper output path
RAW_WEB_DIR = Path(__file__).parents[3] / "data" / "raw" / "web"
RAW_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"

RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
RAW_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class WebContentExtractor:
    def __init__(
            self,
            filter_fn,
            request_timeout=REQUEST_TIMEOUT,
            delay_range=(0.5, 1.5),
            recrawl_days=7,
            input_filename=URL_DISCOVERY_OUTPUT,
            history_filename= WEB_SCRAPPER_HISTORY,
            output_filename=WEB_SCRAPER_OUTPUT):
        """
        filter_fn: function that takes a string and returns filtered string
        recrawl_days: re-crawl a URL only if older than this many days
        """
        self.filter_fn = filter_fn
        self.request_timeout = request_timeout
        self.delay_range = delay_range
        self.recrawl_days = recrawl_days

        self.input_path = RAW_WEB_DIR / input_filename
        self.history_path = RAW_WEB_DIR / history_filename
        self.output_path = RAW_PROCESSED_DIR / output_filename

        self.history = self._load_history()
    
    def _load_history(self):
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                return json.load(f)
        return {}

    def _save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def get_content_hash(self,content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def is_recently_crawled(self, url):
        """Check if the URL was crawled recently"""
        if url not in self.history:
            return False
        try:
            last_crawled = datetime.fromisoformat(self.history[url]["last_crawled"])
            return (datetime.now(timezone.utc) - last_crawled).days < self.recrawl_days
        except Exception:
            return False

    def get_raw_markdown_urls(self, repo_url):
        user_repo = "/".join(urlparse(repo_url).path.strip("/").split("/")[:2])
        branches = ["main", "master"]
        filenames = ["README.md", "readme.md", "docs/README.md", "src/README.md"]
        return [f"https://raw.githubusercontent.com/{user_repo}/{branch}/{filename}"
                for branch in branches for filename in filenames]

    def fetch_markdown(self, url):
        try:
            headers = {"User-Agent": USER_AGENT}
            res = requests.get(url, headers=headers, timeout=self.request_timeout)
            if res.status_code == 200 and res.text.strip():
                return res.text
        except Exception:
            return ""
        return ""

    def fetch_html(self, url):
        """Fetch HTML content using plain requests (no JS rendering)."""
        headers = {"User-Agent": USER_AGENT}
        html = ""
        try:
            res = requests.get(url, headers=headers, timeout=self.request_timeout)
            res.raise_for_status()
            html = res.text
        except Exception as e:
            print(f"[ERROR] Could not fetch {url}: {e}")

        return html

    # -------------------- Extraction ---------------------
    def extract_text(self, html):
        """Extract clean text from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else ""
        texts = [title] if title else []
        for tag in soup.find_all(["h1", "h2", "h3", "p"]):
            text = tag.get_text(strip=True)
            if text:
                texts.append(text)

        clean_text = "\n".join(texts)
        # replaces any sequence of multiple newlines (\n+) with just one newline (\n)
        return re.sub(r'\n+', '\n', clean_text).strip()

    def process_url(self, url, source,title=""):
        """Fetch, extract, and filter content from a URL"""
        if is_medium_url(url):
            print(f"[SKIP] Skipping Medium URL: {url}")
            return None

        if not can_fetch(url):
            print(f"[SKIP] Disallowed by robots.txt: {url}")
            return None
        
        if is_github_repo(url):
            print(f"Fetching GitHub Markdown: {url}")
            for raw_url in self.get_raw_markdown_urls(url):
                content = self.fetch_markdown(raw_url)
                filtered = self.filter_fn(content)

                if filtered.strip():
                    return {"url": raw_url, "source": source, "title": title, "content": filtered}
            return None

        print(f"Fetching: {url}")
        html = self.fetch_html(url)
        if not html:
            return None
        
        content = self.extract_text(html)
        filtered = self.filter_fn(content)
        if not filtered.strip():
            print(f"[SKIP] No relevant content for {url}")
            return None

        title = filtered.split("\n")[0] if not title else title
        return {"url": url, "source": source, "title": title, "content": filtered}

    # Incremental Extraction to CSV
    def extract_from_csv(self):
        df_urls = pd.read_csv(self.input_path)
        corpus = []
        now = datetime.now(timezone.utc).isoformat()

        for idx, row in tqdm(df_urls.iterrows(), total=len(df_urls), desc="Crawling Progress", unit="url"):
            url = row["url"]
            source = row.get("source", "")
            title = row.get("title", "")

            if self.history.get(url) and self.is_recently_crawled(url):
                self.history[url] = self.history.get(url, {})
                self.history[url]["last_crawled"] = now
                self.history[url]["status"] = "recent"
                tqdm.write(f"[SKIP] Recently crawled: {url}")
                continue

            result = self.process_url(url,source,title)
            if result:
                new_hash = self.get_content_hash(result["content"])
                old_hash = self.history.get(url, {}).get("content_hash")

                if new_hash == old_hash:
                    self.history[url].update({
                        "last_crawled": now,
                        "status": "no_change"
                        })
                    tqdm.write(f"[SKIP] No content change for {url}")
                    continue
            
                self.history[url] = {
                    "last_crawled": now,
                    "content_hash": new_hash,
                    "status": "success"
                    }
                corpus.append(result)
            else:
                # Failed or skipped crawl
                self.history[url] = {
                    "last_crawled": now,
                    "content_hash": None,
                    "status": "skipped"
                }
                tqdm.write(f"[SKIP] Failed or skipped crawling for {url}")
            
            # self._save_history()
            # Save progress periodically
            if idx % 10 == 0:
                self._save_history()

            time.sleep(random.uniform(*self.delay_range))

        # Incremental save to avoid overwriting
        if corpus:
            self._save_history()
            df_new = pd.DataFrame(corpus).drop_duplicates(subset=["url"]) # Rare case with current setup
            df_new.to_csv(self.output_path, mode='a', index=False, header=not os.path.exists(self.output_path))

        tqdm.write(f"✅ Completed crawling. Total new content: {len(corpus)}")


# ---------- Main ----------
if __name__ == "__main__":
    from src.data_acquisition.web.content_filter import filter_system_design_content
    extractor = WebContentExtractor(filter_fn=filter_system_design_content)
    try:
        extractor.extract_from_csv()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        extractor._save_history()
    


# Can reuse the same class for other files and other filter function”:
# <
# from src.data_acquisition.web.content_filter import filter_cloud_architecture_content
# extractor = WebContentExtractor(
#     filter_fn=filter_cloud_architecture_content,
#     input_filename="cloud_architecture_urls.csv",
#     output_filename="cloud_architecture_corpus.csv",
#     history_filename="cloud_architecture_history.json",
# )
# extractor.extract_from_csv()
# >

# Run it like this - 
# python -m src.data_acquisition.web.web_scrapper