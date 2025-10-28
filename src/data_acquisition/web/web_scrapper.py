# src/data_acquisition/web/web_scrapper.py
import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
from requests_html import HTMLSession
import urllib.robotparser
import time
import random
from pathlib import Path

from src.data_acquisition.web.web_utils import (
    can_fetch, is_medium_url, is_github_repo
)

from src.data_acquisition.web.web_constants import (
    URL_DISCOVERY_OUTPUT, WEB_SCRAPER_OUTPUT, JS_REQUIRED_DOMAINS , REQUEST_TIMEOUT, USER_AGENT
)

# Set the proper output path
RAW_WEB_DIR = Path(__file__).parents[3] / "data" / "raw" / "web"
RAW_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"

RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)
RAW_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = RAW_WEB_DIR / URL_DISCOVERY_OUTPUT
OUTPUT_PATH = RAW_PROCESSED_DIR / WEB_SCRAPER_OUTPUT


class WebContentExtractor:
    def __init__(self, filter_fn, request_timeout=REQUEST_TIMEOUT, delay_range=(1.5, 3.5)):
        """
        filter_fn: function that takes a string and returns filtered string
        """
        self.filter_fn = filter_fn
        self.request_timeout = request_timeout
        self.delay_range = delay_range


    def get_raw_markdown_urls(self, repo_url):
        user_repo = "/".join(urlparse(repo_url).path.strip("/").split("/")[:2])
        branches = ["main", "master"]
        filenames = ["README.md", "readme.md"]
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
        """Fetch HTML with fallback to JS rendering if needed."""
        domain = urlparse(url).netloc
        headers = {"User-Agent": USER_AGENT}
        html = ""

        # --- Step 1: Try normal request ---
        try:
            res = requests.get(url, headers=headers, timeout=self.request_timeout)
            res.raise_for_status()
            html = res.text
        except Exception as e:
            print(f"[ERROR] Could not fetch {url}: {e}")

        # --- Step 2: Decide if JS rendering is needed ---
        js_render_needed = (
            domain in JS_REQUIRED_DOMAINS or len(html) < 1000
        )

        if js_render_needed:
            print(f"[INFO] Using JS rendering for {domain}")
            try:
                session = HTMLSession()
                r = session.get(url, timeout=self.request_timeout)
                r.html.render(timeout=25, sleep=2)
                html = r.html.html
            except Exception as e:
                print(f"[WARN] JS rendering unavailable or failed for {url}: {e}")
            finally:
                try:
                    session.close()
                except:
                    pass
        return html

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
        return "\n".join(texts)

    def process_url(self, url, source):
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
                    return {"url": raw_url, "source": source, "title": filtered.split("\n")[0], "content": filtered}
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

        return {"url": url, "source": source, "title": filtered.split("\n")[0], "content": filtered}

    def extract_from_csv(self, input_file, output_file):
        df_urls = pd.read_csv(input_file)
        corpus = []

        for idx, row in df_urls.iterrows():
            result = self.process_url(row['url'], row['source'])
            if result:
                corpus.append(result)
            time.sleep(random.uniform(*self.delay_range))

        df = pd.DataFrame(corpus)
        df.to_csv(output_file, index=False)
        # df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        print(f"\n✅ Saved {len(df)} filtered pages to {output_file}")


# ---------- Main ----------
if __name__ == "__main__":
    from src.data_acquisition.web.content_filter import filter_system_design_content
    extractor = WebContentExtractor(filter_fn=filter_system_design_content)
    extractor.extract_from_csv(INPUT_PATH, OUTPUT_PATH)

# Run it like this - 
# python -m src.data_acquisition.web.web_scrapper