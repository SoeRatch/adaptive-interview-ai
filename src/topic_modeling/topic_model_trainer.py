# src/topic_modeling/topic_model_trainer.py
"""
Topic Model Trainer — Stage 2 of Topic Modeling
--------------------
Trains a BERTopic model on document embeddings and stores the trained model.

Responsibilities:
- Load documents and embeddings.
- Fit BERTopic.
- Save model artifacts for inference or metadata extraction.
"""

import os
import re
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from pathlib import Path
from nltk.corpus import stopwords
import nltk

from src.topic_modeling.embedding_storage import EmbeddingStorage
from src.topic_modeling.constants import (
    CHUNKED_CORPUS_OUTPUT,
    TOPIC_MODEL_OUTPUT
)

# Ensure NLTK stopwords available
try:
    stop_words = stopwords.words("english")
except LookupError:
    nltk.download("stopwords") # stores in global cache directory
    stop_words = stopwords.words("english")

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path(__file__).parents[2] / "data" / "embeddings"

class TopicModelTrainer:
    def __init__(
            self,
            input_filename: str = CHUNKED_CORPUS_OUTPUT,
            model_filename: str = TOPIC_MODEL_OUTPUT,
            language: str = "english",
            verbose: bool = True,
            calculate_probabilities: bool = True,
            min_topic_size: int = 10,
            use_cache: bool = True
            ):
        """
        Args:
            input_filename (str): Preprocessed document chunks.
            model_filename (str): Output model file.
            language (str): Language for topic modeling.
        """
        self.input_path = PROCESSED_DIR / input_filename
        self.model_path = MODEL_DIR / model_filename
        self.language = language
        self.verbose = verbose
        self.use_cache = use_cache
        self.storage = EmbeddingStorage(data_dir=EMBEDDINGS_DIR, mode= "npz")

        # Setup vectorizer (for topic labeling only)
        self.vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # only alphabetic tokens
            max_features=8000
        )

        # UMAP configuration for dimensionality reduction control
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        # Load trained model or initialize model
        if self.model_path.exists() and self.use_cache:
            print(f"[Init] Loading existing BERTopic model from {self.model_path}")
            self.model = BERTopic.load(self.model_path)
            self.is_trained = True
        else:
            if self.model_path.exists() and not self.use_cache:
                backup_path = self.model_path.with_suffix(".bak")
                self.model_path.rename(backup_path)
                print(f"[Backup] Previous model backed up at {backup_path}")
                print(f"[Info] use_cache=False — existing model at {self.model_path} will be retrained and overwritten.")

            print(f"[Init] Creating new BERTopic model...")
            self.model = BERTopic(
                embedding_model=None, # embeddings provided manually
                vectorizer_model=self.vectorizer_model,
                umap_model=self.umap_model,
                verbose=self.verbose,
                calculate_probabilities=calculate_probabilities,
                # min_topic_size = min_topic_size 
            )
            self.is_trained = False

        print("[Init] BERTopic model initialized successfully.")


    def train_model(self) -> BERTopic:
        """
        Train the BERTopic model on documents and embeddings.
        """
        if self.is_trained and self.use_cache:
            print("[Cache] Model already trained. Skipping retraining.")
            return self.model
                
        # Load input data
        df = pd.read_csv(self.input_path)

        # sanity guard
        if df["text"].isnull().any():
            print("[Warning] Null text values detected — dropping them.")
            df = df.dropna(subset=["text"]).reset_index(drop=True)

        documents = df["text"].astype(str).tolist()
        if not documents:
            raise ValueError("No valid documents found for embedding generation.")
        
        # Load precomputed embeddings
        embeddings_data = self.storage.load()
        embeddings = embeddings_data.get("embeddings")

        if embeddings is None:
            raise ValueError("Embeddings not found in loaded data.")
        
        if len(documents) != embeddings.shape[0]: # Alignment check
            raise ValueError(f"Mismatch: {len(documents)} documents vs {embeddings.shape[0]} embeddings")
        
        # Train BERTopic model
        print(f"[Training] BERTopic model on {len(documents)} documents...")
        topics, probs = self.model.fit_transform(documents, embeddings)
        # The embedding_model inside BERTopic is never invoked during training if embeddings are explicitly provided.
        print(f"[Done] Model trained. Unique topics: {len(np.unique(topics))}")
        print("[Summary] Example topics:")
        for topic_num in list(set(topics))[:5]:
            print(f"  {topic_num}: {self.model.get_topic(topic_num)}")


        self.is_trained = True
        return self.model

    def save_model(self):
        """
        Save the trained BERTopic model to disk.
        """
        self.model.save(self.model_path)
        print(f"[Saved] BERTopic model in {self.model_path}")
        return self.model_path

if __name__ == "__main__":
    import time
    start = time.perf_counter()

    trainer = TopicModelTrainer(
        input_filename = CHUNKED_CORPUS_OUTPUT,
        model_filename = TOPIC_MODEL_OUTPUT,
        language= "english",
        verbose=True,
        use_cache=False
    )
    topic_model = trainer.train_model()
    model_path = trainer.save_model()
    print(f"[Info] Model saved at: {model_path}")

    total = time.perf_counter() - start
    print(f"[Total] End-to-end execution time: {total/60:.2f} minutes")

# Run it like this - 
# python -m src.topic_modeling.topic_model_trainer