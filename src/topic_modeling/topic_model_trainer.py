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
nltk.download('stopwords')

from src.topic_modeling.embedding_storage import EmbeddingStorage
from src.topic_modeling.constants import (
    CHUNKED_CORPUS_OUTPUT,
    EMBEDDING_MODEL_NAME,
    TOPIC_MODEL_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path(__file__).parents[2] / "data" / "embeddings"

class TopicModelTrainer:
    def __init__(
            self,
            input_filename: str = CHUNKED_CORPUS_OUTPUT,
            embedding_model_name: str = EMBEDDING_MODEL_NAME,
            model_filename: str = TOPIC_MODEL_OUTPUT,
            language: str = "english",
            verbose: bool = True,
            ):
        """
        Args:
            input_filename (str): Preprocessed document chunks.
            embedding_model_name (str): Name of the embedding model
            model_filename (str): Output model file.
            language (str): Language for topic modeling.
        """
        self.input_path = PROCESSED_DIR / input_filename
        self.embedding_model_name = embedding_model_name
        self.model_path = MODEL_DIR / model_filename
        self.language = language
        self.verbose = verbose

        self.storage = EmbeddingStorage(data_dir=EMBEDDINGS_DIR, mode= "npz")

        # Setup embedding model (semantic)
        print(f"[Init] Setting up semantic embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # In later stage i.e in production or reproducible MLOps pipelines, change to this
        # if (MODEL_DIR / EMBEDDING_MODEL_OUTPUT).exists():
        #     self.embedding_model = BERTopic.load(MODEL_DIR / EMBEDDING_MODEL_OUTPUT)
        # else:
        #     self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Setup vectorizer (for topic labeling only)
        stop_words = stopwords.words(self.language)
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

        # Initialize BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            umap_model=self.umap_model,
            verbose=self.verbose,
        )
        print("[Init] BERTopic model initialized successfully.")
        
    
    def update_documents_with_topics(self,df: pd.DataFrame, topics: list[int], probs: np.ndarray):
        """
        Update the documents with their assigned topics and probabilities.
        """

        # Guard: ensure consistent lengths
        if len(df) != len(topics):
            raise ValueError(
            f"Mismatch: {len(df)} documents vs {len(topics)} topic assignments."
        )

        # Handle existing columns safely
        if "topic" in df.columns or "probability" in df.columns:
            print("[Warning] Existing 'topic' or 'probability' columns detected — overwriting.")
            df = df.drop(columns=["topic", "probability"], errors="ignore")

        df["topic"] = topics
        df["probability"] = probs

        # Overwrite the same input file
        temp_path = self.input_path.with_suffix(".tmp.csv")
        df.to_csv(temp_path, index=False)
        temp_path.replace(self.input_path)
        
        print(f"[Updated] Added topic assignments in {self.input_path}")

    def train_model(self) -> tuple[BERTopic, list[int], np.ndarray]:
        """
        Train the BERTopic model on documents and embeddings.
        """
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
        
        # Alignment check
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(documents)} documents vs {embeddings.shape[0]} embeddings")
        
        # Train BERTopic model
        print(f"[Training] BERTopic model on {len(documents)} documents...")
        topics, probs = self.model.fit_transform(documents, embeddings)
        # The embedding_model inside BERTopic is never invoked during training if embeddings are explicitly provided.
        print(f"[Done] Model trained. Unique topics: {len(set(topics))}")

        # Handle None probabilities(for unassigned documents)
        probs = np.array([p if p is not None else 0.0 for p in probs])

        # Update the same input documents with topic assignments
        self.update_documents_with_topics(df, topics, probs)
        return self.model, topics, probs

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
        embedding_model_name = EMBEDDING_MODEL_NAME,
        model_filename = TOPIC_MODEL_OUTPUT,
        language= "english",
        verbose=True,
    )
    model, topics, probs = trainer.train_model()
    trainer.save_model()

    total = time.perf_counter() - start
    print(f"[Total] End-to-end execution time: {total/60:.2f} minutes")

# Run it like this - 
# python -m src.topic_modeling.topic_model_trainer