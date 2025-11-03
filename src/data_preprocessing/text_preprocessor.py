# src/data_preprocessing/text_preprocessor.py

"""
text_preprocessor.py
----------------------
Stage 1: Text Cleaning & Normalization

Purpose:
- Clean and normalize scraped documents
- Remove duplicate URLs (keep latest)
- Remove non-English or too-short content
- Preserve paragraph structure for semantic chunking
- Output one clean document per row (ready for embedding/chunking)

Input:  data/processed/web_scraper_output.csv
Output: data/processed/clean_corpus.csv
"""

import os
import re
import pandas as pd
import html
from langdetect import detect, DetectorFactory
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from src.data_preprocessing.preprocess_constants import (
    WEB_SCRAPER_OUTPUT, CLEAN_CORPUS_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Setup ---
DetectorFactory.seed = 0  # make language detection deterministic

class TextPreprocessor:
    """Clean and normalize scraped documents for downstream modeling."""

    def __init__(self, min_word_count=300, input_filename=WEB_SCRAPER_OUTPUT, output_filename=CLEAN_CORPUS_OUTPUT):
        self.min_word_count = min_word_count
        self.input_path = PROCESSED_DIR / input_filename
        self.output_path = PROCESSED_DIR / output_filename

    def extract_clean_text(self, raw_text: str) -> str:
        """
        Normalize and clean extracted text (for web, PDF, or other sources).
        Use this AFTER format-specific extraction (like BeautifulSoup, pdfminer, etc.).
        """
        if not raw_text:
            return ""

        # 1. Decode HTML entities (&amp; → &)
        text = html.unescape(raw_text)

        # 2. Normalize line endings across different operating systems.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove excessive blank lines but preserve paragraph separation
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Remove non-printable characters (e.g., control chars)
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)

        # 5. Remove URLs and special characters
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 6. Remove unwanted chars but keep common punctuation
        # text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"[^\w\s.,!?%\-]", " ", text)

        # 7. Fix space before punctuation for ex - "word , something" instead of "word, something"
        text = re.sub(r"\s+([.,!?%\-])", r"\1", text)

        # 8. Collapse multiple spaces within lines
        text = re.sub(r"[ \t]{2,}", " ", text)

        # 9. Normalize per-line whitespace but preserve blank lines
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        rebuilt = "\n".join(lines)
        return rebuilt

    
    def is_meaningful_content(self, text: str) -> bool:
        """
        Returns True if text has sufficient meaningful content after normalization.
        Used to filter out boilerplate, short, or non-informative pages.
        """
        cleaned = re.sub(r'\s+', ' ', text).strip()
        words = re.findall(r"\b[\w'-]+\b", cleaned)
        # words = [w for w in text.split() if w.isalpha() and len(w) > 2]
        return len(words) >= self.min_word_count


    def is_english(self, text: str) -> bool:
        """Check if text is English using language detection."""
        try:
            sample = text[:2000]
            return detect(sample) == "en"
        except:
            return False

    # Main Pipeline
    def process(self):
        """Run the full cleaning + deduplication pipeline."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")

        print(f"Loading scraped data from {self.input_path}")
        df = pd.read_csv(self.input_path)

        # Deduplicate by URL — keep the last occurrence (latest crawl)
        df = df.drop_duplicates(subset=["url"], keep="last")
        
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            url = row["url"]
            source = row.get("source", "")
            title = row.get("title", "")
            content = row.get("content", "")

            # Skip empty or trivially short raw content
            if not isinstance(content, str) or len(content) < 100:
                tqdm.write(f"[SKIP] Skipped short content: {url}")
                continue
            
            # 1. Clean HTML and preserve paragraphs
            text = self.extract_clean_text(content)
            if not text:
                tqdm.write(f"[SKIP] Empty text after cleaning: {url}")
                continue
            
            # 2. Filter short documents i.e filtering by meaningful content length
            if not self.is_meaningful_content(text):
                tqdm.write(f"[SKIP] Too short (<{self.min_word_count} words): {url}")
                continue  # skip texts with <300 words

            # 3. Filter non-English documents
            if not self.is_english(text):
                tqdm.write(f"[SKIP] Non-English content skipped: {url}")
                continue

            records.append({
                "url": url,
                "source": source,
                "title": title,
                "text": text
                })

        if not records:
            print("No valid documents after cleaning. Check content length or filters.")
            return
        
        # Save final clean corpus
        df_clean = pd.DataFrame(records)
        df_clean.reset_index(drop=True, inplace=True)
        df_clean["id"] = df_clean.index + 1
        df_clean = df_clean[["id", "url", "source","title","text"]]
        df_clean.to_csv(self.output_path, index=False)

        print(f"\n✅ Clean corpus saved in: {self.output_path}")
        print(f"Total cleaned documents: {len(df_clean)}")


if __name__ == "__main__":
    preprocessor = TextPreprocessor(min_word_count=300)
    preprocessor.process()

# Can reuse the same class for other files”:
# <
# preprocessor = TextPreprocessor(
#     min_word_count=300,
#     input_filename="cloud_architecture_web_scraper_output.csv",
#     output_filename="cloud_architecture_cleaned_corpus.csv",
#     )
# preprocessor.process()
# >

# Run it like this - 
# python -m src.data_preprocessing.text_preprocessor