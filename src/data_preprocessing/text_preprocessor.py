# src/data_preprocessing/text_preprocessor.py

"""
text_preprocessor.py
----------------------
End-to-end preprocessing pipeline:
- Cleans and normalizes scraped corpus
- Removes duplicates and non-English content
- Deduplicates semantically similar docs
- Chunks long texts for topic modeling
"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import nltk
import torch
from tqdm import tqdm
tqdm.pandas()

from src.data_preprocessing.preprocess_constants import (
    WEB_SCRAPER_OUTPUT, CLEAN_CORPUS_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = PROCESSED_DIR / WEB_SCRAPER_OUTPUT
OUTPUT_PATH = PROCESSED_DIR / CLEAN_CORPUS_OUTPUT

# --- Setup ---
DetectorFactory.seed = 0  # make language detection deterministic


def download_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
            print(f"✔ NLTK resource '{name}' already exists, skipping download.")
        except LookupError:
            print(f"Downloading NLTK resource: {name} ...")
            nltk.download(name)

download_nltk_resources()


class CleanCorpus:
    def __init__(self, chunk_size=400, similarity_threshold=0.92):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Cleaning Helpers
    def clean_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2
        ]
        return " ".join(cleaned)

    def chunk_text(self, text: str):
        words = text.split()
        return [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

    def is_english(self, text: str) -> bool:
        try:
            return detect(text) == "en"
        except:
            return False

    # Deduplication
    # def semantic_deduplicate(self, texts: list[str]) -> list[str]:
    #     """Remove near-duplicate documents using cosine similarity."""
    #     if len(texts) < 2:
    #         return texts

    #     # Compute embeddings
    #     embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    #     keep_indices = []
    #     seen = set()

    #     for i in range(len(texts)):
    #         if i in seen:
    #             continue
    #         keep_indices.append(i)

    #         # Only compare with subsequent texts to avoid repeated checks
    #         sims = util.cos_sim(embeddings[i], embeddings[i + 1 :]).squeeze(0)
    #         for offset, score in enumerate(sims):
    #             j = i + 1 + offset
    #             if score > self.similarity_threshold:
    #                 seen.add(j)  # Mark duplicate

    #     return [texts[i] for i in keep_indices]


    # Deduplication
    def semantic_deduplicate_vectorized(self, texts: list[str]) -> list[str]:
        """
        Remove near-duplicate documents using cosine similarity.
        Fully vectorized version using PyTorch.
        """
        if len(texts) < 2:
            return texts

        # Compute embeddings
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

        # Compute full pairwise cosine similarity matrix
        sims = util.cos_sim(embeddings, embeddings)

        # Mask upper triangle and diagonal to avoid self-comparison
        mask = torch.triu(torch.ones_like(sims), diagonal=0).bool()
        sims = sims.masked_fill(mask, 0.0)

        # Keep track of duplicates
        duplicates = torch.any(sims > self.similarity_threshold, dim=0)

        # Keep texts that are not duplicates
        keep_indices = [i for i, is_dup in enumerate(duplicates) if not is_dup]

        return [texts[i] for i in keep_indices]


    # Pipeline
    def process(self, input_file=INPUT_PATH, output_file=OUTPUT_PATH):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"❌ File not found: {input_file}")

        print(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            url, source, content = row.get("url"), row.get("source"), row.get("content", "")

            # skip empty or trivially short raw content
            if not isinstance(content, str) or len(content) < 100:
                continue
            
            # 1. Clean HTML
            text = self.clean_html(content)
            if not text or not self.is_english(text):
                continue
            
            # 2. Normalize text (lowercase, lemmatize, remove stopwords)
            normalized = self.normalize_text(text)
            if not normalized:
                continue
            
            # 3. Filter by meaningful content length
            if len(normalized.split()) < 300:
                continue  # skip texts with <300 words after cleaning & normalization

            # 4. Chunk long texts
            chunks = self.chunk_text(normalized)
            for i, chunk in enumerate(chunks, 1):
                records.append({
                    "url": url,
                    "source": source,
                    "chunk_id": i,
                    "text": chunk
                })

        print(f"Total text chunks before deduplication: {len(records)}")

        df_clean = pd.DataFrame(records)
        # dedup_texts = self.semantic_deduplicate(df_clean["text"].tolist())
        dedup_texts = self.semantic_deduplicate_vectorized(df_clean["text"].tolist())
        df_dedup = df_clean[df_clean["text"].isin(dedup_texts)]

        df_dedup.to_csv(output_file, index=False)
        print(f"\n✅ Final cleaned corpus saved to {output_file}")
        print(f"Total unique chunks: {len(df_dedup)}")


if __name__ == "__main__":
    cleaner = CleanCorpus(chunk_size=400, similarity_threshold=0.92)
    cleaner.process(INPUT_PATH, OUTPUT_PATH)

# Run it like this - 
# python -m src.data_preprocessing.text_preprocessor