# src/topic_modeling/embedding_generator.py
"""
Embedding Generator — Stage 1 of Topic Modeling
================================================
Generates dense vector embeddings for text documents.

Responsibilities:
- Load pre-trained SentenceTransformer model.
- Generate normalized embeddings for each document.
- Cache results to avoid redundant computation.
- Persist embeddings + metadata for downstream tasks.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.topic_modeling.constants import (
    CHUNKED_CORPUS_OUTPUT,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_OUTPUT,
    EMBEDDING_MODEL_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

EMBEDDINGS_DIR = Path(__file__).parents[2] / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingGenerator:
    """Generate and cache embeddings for topic modeling pipeline."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        input_filename: str = CHUNKED_CORPUS_OUTPUT,
        embeddings_filename: str = EMBEDDINGS_OUTPUT,
        model_filename: str = EMBEDDING_MODEL_OUTPUT,
        batch_size: int = 32,
        use_cache: bool = True
    ):
        self.input_path = PROCESSED_DIR / input_filename
        self.embeddings_path = EMBEDDINGS_DIR / embeddings_filename
        self.model_path = MODEL_DIR / model_filename
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.embedding_model_name = embedding_model_name

        print(f"\n[Init] Loading embedding model - {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)

    def generate_embeddings(self) -> np.ndarray:
        """Generate or load cached embeddings for text documents."""
        if self.use_cache and self.embeddings_path.exists():
            tqdm.write(f"[Cache] Found existing embeddings at {self.embeddings_path}. Loading...")
            data = np.load(self.embeddings_path, allow_pickle=True)
            return data["embeddings"]

        # Load input data
        df = pd.read_csv(self.input_path)

        # sanity guard
        if df["text"].isnull().any():
            print("[Warning] Null text values detected — dropping them.")
            df = df.dropna(subset=["text"]).reset_index(drop=True)
            
        documents = df["text"].astype(str).tolist()
        
        if not documents:
            raise ValueError("No valid documents found for embedding generation.")

        tqdm.write(f"[Encoding] Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )

        # Save both embeddings + document IDs together
        np.savez_compressed(
            self.embeddings_path,
            embeddings=embeddings,
            ids=df["id"].values,
            urls=df["url"].values
        )

        # np.savez_compressed() creates a compressed .npz that saves multiple NumPy arrays into one file.
        # It applies ZIP compression internally, resulting in smaller file size than a plain .npy.
        # Instead of saving embeddings.npy, ids.npy, urls.npy, etc. separately, save just one file

        tqdm.write(f"[Done] Saved {len(embeddings)} embeddings to {self.embeddings_path}")
        return embeddings

    def save_model(self) -> str:
        """Save SentenceTransformer model locally for reproducibility."""
        joblib.dump(self.model, self.model_path)
        tqdm.write(f"[Saved] Model serialized to {self.model_path}")
        return self.model_path

    def summary(self, embeddings: np.ndarray, df: pd.DataFrame):
        """Print useful summary for sanity check."""
        print("\n=== Embedding Summary ===")
        print(f"Model Name      : {self.embedding_model_name}")
        print(f"Documents       : {len(df)}")
        print(f"Embedding Dim   : {embeddings.shape[1] if embeddings.ndim > 1 else 'N/A'}")
        print(f"Embeddings File : {self.embeddings_path}")
        print(f"Model File      : {self.model_path}")
        print("==========================\n")


if __name__ == "__main__":
    # Stage entry point for pipeline
    emb_gen = EmbeddingGenerator(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        input_filename=CHUNKED_CORPUS_OUTPUT,
        embeddings_filename=EMBEDDINGS_OUTPUT,
        model_filename=EMBEDDING_MODEL_OUTPUT,
        batch_size=32,
        use_cache=True
    )

    # Generate & save
    embeddings = emb_gen.generate_embeddings()
    emb_gen.save_model()

    # Optional: show quick summary
    df = pd.read_csv(emb_gen.input_path)
    emb_gen.summary(embeddings, df)

# Run it like this - 
# python -m src.topic_modeling.embedding_generator