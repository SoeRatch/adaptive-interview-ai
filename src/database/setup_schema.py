# src/database/setup_schema.py

import os
import logging
from dotenv import load_dotenv
from src.database.postgres_handler import PostgresHandler

load_dotenv()

def get_connection_params() -> dict:
    """Fetch database connection parameters from environment variables."""
    return {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432))
    }


def setup_schema():
    """Creates database tables if they do not exist."""
    conn_params = get_connection_params()
    pg_handler = PostgresHandler(conn_params)

    table_schemas = [
        (
            "topic_metadata",
            {
                "id": "SERIAL PRIMARY KEY",
                "model_topic_id": "INTEGER",
                "name": "TEXT",
                "representation": "TEXT",
                "count": "INTEGER"
            },
        ),
        (
            "topic_documents",
            {
                "document_id": "SERIAL PRIMARY KEY",
                "topic_id": "INTEGER REFERENCES topic_metadata(id)",
                "url": "TEXT",
                "source": "TEXT",
                "title": "TEXT",
                "text": "TEXT"
            },
        ),
        (
            "schema_version",
            {
                "version": "INTEGER PRIMARY KEY",
                "applied_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
        ),
    ]

    try:
        print("Starting database schema setup...")
        pg_handler.create_tables_in_order(table_schemas)
        print("✅ Database schema setup complete.")
    except Exception as e:
        print(f"Schema setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_schema()

# Run it like this - 
# python -m src.database.setup_schema