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
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
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


# def download_nltk_resources():
#     resources = {
#         "punkt": "tokenizers/punkt",
#         "punkt_tab": "tokenizers/punkt_tab",
#         "stopwords": "corpora/stopwords",
#         "wordnet": "corpora/wordnet"
#     }
#     for name, path in resources.items():
#         try:
#             nltk.data.find(path)
#             print(f"NLTK resource '{name}' already exists, skipping download.")
#         except LookupError:
#             print(f"Downloading NLTK resource: {name} ...")
#             nltk.download(name)

# download_nltk_resources()


class TextPreprocessor:
    """Clean and normalize scraped documents for downstream modeling."""

    def __init__(self, min_word_count=300, input_filename=WEB_SCRAPER_OUTPUT, output_filename=CLEAN_CORPUS_OUTPUT):
        self.min_word_count = min_word_count
        # self.stop_words = set(stopwords.words("english"))
        # self.lemmatizer = WordNetLemmatizer()

        self.input_path = PROCESSED_DIR / input_filename
        self.output_path = PROCESSED_DIR / output_filename

    # Cleaning Helpers
    def extract_clean_text(self, text: str) -> str:
        """Remove HTML tags, scripts, URLs, and non-alphabetic noise while preserving paragraphs."""
        soup = BeautifulSoup(text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        # Extract text with newlines to mark paragraph breaks
        text = soup.get_text(separator="\n")

        # Remove URLs and special characters
        text = re.sub(r"http\S+", "", text)
        # text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"[^\w\s.,!?%\-]", " ", text) # keeps common punctuation


        # Collapse 3+ newlines to 2 (to represent paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize internal spaces within lines only
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        text = "\n".join([line for line in lines if line])

        return text.strip()
    
    def is_meaningful_content(self, text: str, min_word_count: int = 300) -> bool:
        """
        Returns True if text has sufficient meaningful content after normalization.
        Used to filter out boilerplate, short, or non-informative pages.
        """
        text = re.sub(r'\s+', ' ', text.lower())
        tokens = [w for w in text.split() if w.isalpha() and len(w) > 2]
        return len(tokens) >= min_word_count


    def is_english(self, text: str) -> bool:
        """Check if text is English using language detection."""
        try:
            return detect(text) == "en"
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
            
            # 2. Filter non-English documents
            if not self.is_english(text):
                tqdm.write(f"[SKIP] Non-English content skipped: {url}")
                continue
            
            
            # 2. Filter short documents i.e filtering by meaningful content length
            if not self.is_meaningful_content(text, self.min_word_count):
                tqdm.write(f"[SKIP] Too short (<{self.min_word_count} words): {url}")
                continue  # skip texts with <300 words

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