# src/topic_modeling/topic_metadata_store.py

"""
Store topic-level metadata into PostgreSQL.
"""

import pandas as pd
from bertopic import BERTopic  # assuming the model is BERTopic
from pathlib import Path

from src.database.postgres_handler import PostgresHandler
from src.topic_modeling.constants import (
    TOPIC_MODEL_OUTPUT,
    TOPIC_METADATA_TABLE
)

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"

class TopicMetadataStore:
    def __init__(
            self,
            db_handler: PostgresHandler,
            model_filename: str = TOPIC_MODEL_OUTPUT,
            topic_metadata_table: str = TOPIC_METADATA_TABLE
            ):
        """
        Args:
            db_handler (PostgresHandler): Database handler for PostgreSQL.
            model_filename (str): Saved topic model filename (pickle or serialized model).
            topic_metadata_table (str): Target table name to store topic metadata.
        """
        self.db = db_handler
        self.model_path = MODEL_DIR / model_filename
        self.topic_metadata_table = topic_metadata_table
        self.topic_model = self._load_model()
    
    def _load_model(self):
        """Load the trained topic model from disk using BERTopic's loader."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Topic model not found at {self.model_path}")

        try:
            model = BERTopic.load(self.model_path)
            print(f"✅ Loaded BERTopic model from {self.model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load topic model: {e}")

    def store_topic_metadata(self):
        """
        Store metadata for each discovered topic from the topic model.

        Schema:
        - id: SERIAL PRIMARY KEY
        - model_topic_id: INTEGER (from topic_model, can be -1)
        - name: TEXT
        - representation: TEXT
        - count: INTEGER
        """
        try:
            topics_info = self.topic_model.get_topic_info()
            if topics_info is None or len(topics_info) == 0:
                print("No topics found in the model — nothing to store.")
                return
            
            df = pd.DataFrame(topics_info)

            if "Topic" not in df.columns:
                raise ValueError("[Error]: Model output missing 'Topic' column — check model format.")

            # Standardize columns
            df.rename(columns={"Topic": "model_topic_id"}, inplace=True)
            df = df.rename(columns={col: col.lower() for col in df.columns})

            # Ensure all expected columns exist
            for col in ["model_topic_id", "name", "representation", "count"]:
                if col not in df.columns:
                    df[col] = None

            df = df[["model_topic_id", "name", "representation", "count"]]

            # Insert data into PostgreSQL
            # Clean existing metadata
            if self.db.table_exists(self.topic_metadata_table):
                 print(f"Cleaning table '{self.topic_metadata_table}' before insert...")
                 self.db.execute_query(f"TRUNCATE TABLE {self.topic_metadata_table} RESTART IDENTITY CASCADE;")
            self.db.insert_dataframe(df, self.topic_metadata_table)
            print(f"✅ Stored {len(df)} topics in table '{self.topic_metadata_table}'.")

        except Exception as e:
            print(f"[Error]: Error storing topic metadata: {e}")
            raise

if __name__ == "__main__":
    # Build DB connection from .env
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

    start_time = time.time()
    print("Starting topic metadata storage...")

    # Initialize handler and store class
    pg_handler = PostgresHandler(conn_params)
    metadata_store = TopicMetadataStore(
        db_handler=pg_handler,
        model_filename= TOPIC_MODEL_OUTPUT,
        topic_metadata_table= TOPIC_METADATA_TABLE
        )

    # Run the storage process
    metadata_store.store_topic_metadata()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Topic metadata successfully stored in {elapsed:.2f} seconds.")

# Run it like this - 
# python -m src.topic_modeling.topic_metadata_store