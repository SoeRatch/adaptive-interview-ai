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
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

from src.topic_modeling.constants import (
    CHUNKED_CORPUS_OUTPUT,
    EMBEDDINGS_OUTPUT,
    TOPIC_MODEL_OUTPUT
)

# Set the proper output path
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
EMBEDDINGS_DIR = Path(__file__).parents[2] / "data" / "embeddings"

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class TopicModelTrainer:
    def __init__(
            self,
            input_filename: str = CHUNKED_CORPUS_OUTPUT,
            embeddings_filename: str = EMBEDDINGS_OUTPUT,
            model_filename: str = TOPIC_MODEL_OUTPUT,
            language: str = "english",
            verbose: bool = True,
            ):
        """
        Args:
            output_dir (str): Directory to save the trained BERTopic model.
            language (str): Language parameter for BERTopic.
        """
        self.input_path = PROCESSED_DIR / input_filename
        self.embeddings_path = EMBEDDINGS_DIR / embeddings_filename
        self.model_path = MODEL_DIR / model_filename

        self.language = language
        self.verbose = verbose

        print(f"[Init] Initializing BERTopic model for language: {language}")
        self.model = BERTopic(language=self.language, verbose=self.verbose)

    def update_documents_with_topics(self,df: pd.DataFrame, topics: list[int], probs: np.ndarray):
        """
        Update the documents with their assigned topics and probabilities.
        """
        df["topic"] = topics
        df["probability"] = probs

        # Overwrite the same input file
        df.to_csv(self.input_path, index=False)

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
        
        # Load embeddings
        data = np.load(self.embeddings_path, allow_pickle=False)
        embeddings = data["embeddings"]
        if embeddings is None:
            raise ValueError(f"Embeddings not found in file {self.embeddings_path}.")
        
        # Critical check - ensure documents and embeddings align
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {embeddings.shape[0]} embeddings"
                )
        
        # Train BERTopic model
        print(f"[Training] BERTopic model on {len(documents)} documents...")
        topics, probs = self.model.fit_transform(documents, embeddings)
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
        input_filename=CHUNKED_CORPUS_OUTPUT,
        embeddings_filename=EMBEDDINGS_OUTPUT,
        model_filename=TOPIC_MODEL_OUTPUT,
        language= "english",
        verbose=True,
    )
    model, topics, probs = trainer.train_model()
    trainer.save_model()

    total = time.perf_counter() - start
    print(f"[Total] End-to-end execution time: {total/60:.2f} minutes")

# Run it like this - 
# python -m src.topic_modeling.topic_model_trainer