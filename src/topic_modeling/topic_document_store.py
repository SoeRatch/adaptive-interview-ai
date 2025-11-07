# src/topic_modeling/topic_document_store.py
"""
Store document-level info:
- Metadata (PostgreSQL)
- Embeddings (Vector DB)
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from pathlib import Path
from src.database.postgres_handler import PostgresHandler
# from src.database.vector_db_factory import get_vector_db_handler
from src.database.langchain_vector_db_factory import get_langchain_vector_db_handler
from src.topic_modeling.embedding_storage import EmbeddingStorage

from src.topic_modeling.constants import (
    TOPIC_METADATA_TABLE,
    TOPIC_DOCUMENT_TABLE,
    TOPIC_MODEL_OUTPUT
)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
EMBEDDINGS_DIR = Path(__file__).parents[2] / "data" / "embeddings"
MODEL_DIR = Path(__file__).parents[2] / "data" / "models"


class TopicDocumentStore:
    def __init__(
        self,
        pg_handler: PostgresHandler,
        backend: str = "faiss",
        topic_metadata_table: str = TOPIC_METADATA_TABLE,
        topic_document_table: str = TOPIC_DOCUMENT_TABLE,
        config=None,
        model_filename: str = TOPIC_MODEL_OUTPUT,
    ):
        """
        Args:
            pg_handler (PostgresHandler): Database handler for PostgreSQL.
            backend (str): Vector DB backend ('faiss', 'pinecone', or 'qdrant').
            topic_metadata_table (str): Table name for topic metadata.
            topic_document_table (str): Table name for topic-doc mapping.
            config (dict): Backend-specific configuration.
        """
        self.pg = pg_handler
        self.vector_db = get_langchain_vector_db_handler(backend, config)
        self.topic_metadata_table = topic_metadata_table
        self.topic_document_table = topic_document_table
        self.model_path = MODEL_DIR / model_filename
        self.storage = EmbeddingStorage(data_dir=EMBEDDINGS_DIR, mode= "npz")

        self.model = self._load_model()
    
    def _load_model(self):
        """Load BERTopic model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model = BERTopic.load(self.model_path)
        print(f"Loaded BERTopic model from {self.model_path}")
        return model

    def store_documents(self):
        """
        Store document-topic mapping in Postgres and embeddings in vector DB.
        """

        print("Loading embedding data...")
        data = self.storage.load()

        docs = data["documents"]
        embeddings = data["embeddings"]
        urls = data.get("urls", [None] * len(docs))
        sources = data.get("sources", [None] * len(docs))
        titles = data.get("titles", [None] * len(docs))

        print("Fetching topic mapping from metadata table...")
        rows = self.pg.fetch_all(
            f"SELECT id, model_topic_id, name FROM {self.topic_metadata_table}"
        )
        topic_map = {r["model_topic_id"]: (r["id"], r["name"]) for r in rows}


        print(f"Predicting topic assignments for {len(docs)} documents...")
        topics, _ = self.model.transform(docs, embeddings)

        # Build DataFrame
        df = pd.DataFrame({
            "topic_model_id": topics,
            "text": docs,
            "url": urls,
            "source": sources,
            "title": titles,
        })

        # Map to topic_metadata.id and topic name
        df["topic_id"] = df["topic_model_id"].map(lambda t: topic_map.get(t, (None, None))[0])
        df["topic_name"] = df["topic_model_id"].map(lambda t: topic_map.get(t, (None, None))[1])
        

        # Prepare for PostgreSQL
        insert_df = df[["topic_id", "topic_name", "text", "url", "source", "title"]]

        print(f"Cleaning table '{self.topic_document_table}' before insert...")
        self.pg.execute_query(
            f"TRUNCATE TABLE {self.topic_document_table} RESTART IDENTITY CASCADE;"
        )

        # Step 1: Insert documents
        print("Inserting document-topic data into PostgreSQL...")
        self.pg.insert_dataframe(insert_df, self.topic_document_table)
        print(f"✅ Stored {len(insert_df)} documents in table '{self.topic_document_table}'.")
        
        # Step 2: Fetch document IDs
        rows = self.pg.fetch_all("SELECT document_id FROM topic_documents ORDER BY document_id;")
        document_ids = [r["document_id"] for r in rows]
        print(f"Sample inserted IDs: {document_ids[:5]}")

        # Step 3: Insert embeddings in Vector DB
        print("Storing embeddings in Vector DB...")

        metadatas = [
            {
                "document_id": doc_id,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "title": title,
                "url": url
                }
                for doc_id, topic_id, topic_name, title, url in zip(
                    document_ids, df["topic_id"], df["topic_name"], df["title"], df["url"]
                    )
        ]
        self.vector_db.build_index(docs, metadatas=metadatas)

        # self.vector_db.upsert_embeddings(ids=document_ids, vectors=embeddings)
        print(f"✅ Stored {len(document_ids)} embeddings in Vector DB ({self.vector_db.__class__.__name__}).")


if __name__ == "__main__":
    import os
    import time
    from dotenv import load_dotenv

    load_dotenv()

    conn_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD")
    }

    print("Starting topic document storage...")
    start_time = time.time()

    pg_handler = PostgresHandler(conn_params)

    doc_store = TopicDocumentStore(
        pg_handler=pg_handler,
        backend="faiss",  # or "qdrant"/"pinecone"
    )
    doc_store.store_documents()

    elapsed = time.time() - start_time
    print(f"✅ Topic document storage completed in {elapsed:.2f} seconds.")

# Run it like this - 
# python -m src.topic_modeling.topic_document_store