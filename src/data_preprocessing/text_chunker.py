# src/data_preprocessing/text_chunker.py
"""
text_chunker.py
----------------------
Stage 2: Text Chunking

Text Chunker — Split large cleaned documents into smaller chunks
for embedding and topic modeling.

Input:  data/processed/cleaned_corpus.csv
Output: data/processed/chunked_corpus.csv
"""

import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data_preprocessing.preprocess_constants import (
    CLEAN_CORPUS_OUTPUT,CHUNKED_CORPUS_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=150, input_filename=CLEAN_CORPUS_OUTPUT, output_filename=CHUNKED_CORPUS_OUTPUT):
        """
        Parameters
        ----------
        chunk_size : int
            Maximum number of characters per chunk.
        chunk_overlap : int
            Overlap between consecutive chunks (to retain context).
        input_filename : str
            preprocessed corpus file.
        output_filename : str
            Filename to save the chunked data.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.input_path = PROCESSED_DIR / input_filename
        self.output_path = PROCESSED_DIR / output_filename

        # Include structure tokens as separators
        self.splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",               # paragraph
                "\n- ",
                "\n",
                "-",
                "\n• ",               # bullet lists (in case any survived)
                ". ", "? ", "! ",     # sentence ends
                " "                   # fallback
            ],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def chunk_texts(self):
        """
        Reads cleaned documents, splits each into chunks,
        and saves as a new CSV for embedding & topic modeling.
        """
        df = pd.read_csv(self.input_path)
        all_chunks = []

        print(f"Loaded {len(df)} cleaned documents.")
        print(f"Chunking with size={self.chunk_size}, overlap={self.chunk_overlap}...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
            text = str(row.get("text", "")).strip()
            if not text:
                tqdm.write(f"[SKIP] Empty text at index: {idx}, skipping.")
                continue

            # Normalize leftover bullets
            text = text.replace("•", "\n- ").replace("\n• ", "\n- ")

            chunks = self.splitter.split_text(text)
            for c in chunks:
                c = c.strip()
                if not c:
                    continue
                all_chunks.append({
                    "source_id": row.get("id", idx),
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "title": row.get("title", ""),
                    "text": c
                })

        chunked_df = pd.DataFrame(all_chunks)
        chunked_df["id"] = range(1, len(chunked_df) + 1) #chunk_id
        chunked_df = chunked_df[["id", "source_id", "url", "source", "title", "text"]] # reorder
        chunked_df.to_csv(self.output_path, index=False)

        print(f"\n✅ Chunking complete.")
        print(f"Documents processed: {len(df)}")
        print(f"Chunks generated: {len(chunked_df)}")
        print(f"Average chunks per document: {len(chunked_df) / len(df):.2f}")
        print(f"Saved to: {self.output_path}")

        return chunked_df


if __name__ == "__main__":
    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=200,
        input_filename=CLEAN_CORPUS_OUTPUT,
        output_filename=CHUNKED_CORPUS_OUTPUT
    )
    chunker.chunk_texts()

# Run it like this - 
# python -m src.data_preprocessing.text_chunker