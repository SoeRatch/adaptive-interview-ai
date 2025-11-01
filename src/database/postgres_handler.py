# src/database/postgres_handler.py
"""
PostgreSQL handler for topic modeling metadata.
Handles table creation, insertion, and simple queries.
"""

import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from typing import List, Dict, Tuple


class PostgresHandler:
    """
    Handles PostgreSQL operations including schema setup and data insertion.
    """

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize with database configuration.

        Example db_config:
        {
            "host": "localhost",
            "port": 5432,
            "dbname": "dbname",
            "user": "user",
            "password": "password"
        }
        """
        self.db_config = db_config

    def execute_query(self, query: str, params: tuple = None):
        """
        Execute a raw SQL query (useful for deletes, updates, or maintenance ops).
        Automatically commits changes.
        """
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                print(f"✅ Executed query: {query[:80]}{'...' if len(query) > 80 else ''}")
        except Exception as e:
            print(f"[ERROR] Failed to execute query: {e}")
            raise
    
    def drop_table(self, table_name: str, cascade: bool = False):
        """
        Drop a table safely. Optionally use CASCADE to remove dependent objects.
        """
        query = f"DROP TABLE IF EXISTS {table_name} {'CASCADE' if cascade else ''};"
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
            print(f"✅ Dropped table '{table_name}'{' (with CASCADE)' if cascade else ''}.")
        except Exception as e:
            print(f"[ERROR] Failed to drop table '{table_name}': {e}")
            raise
    
    def drop_tables(self, tables: list[str], cascade: bool = False):
        """
        Drop multiple tables safely in order.
        """
        for table in tables:
            self.drop_table(table, cascade=cascade)

    # Connection Management
    def _get_connection(self):
        """Establish and return a PostgreSQL connection."""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"[Connection Error] Failed to connect to PostgreSQL: {e}")
            raise


    # Table Creation
    def create_table_if_not_exists(self, table_name: str, schema: Dict[str, str]):
        """
        Creates a table if it doesn't exist.
        Example schema = {"topic_id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        """
        columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
            print(f"✅ Table '{table_name}' created (if not exists).")
        except Exception as e:
            print(f"[ERROR] Error creating table '{table_name}': {e}")
            if "referenced table" in str(e):
                print("Hint: Ensure the referenced parent table exists before creating this table.")
            raise

    def create_tables_in_order(self, table_schemas: List[Tuple[str, Dict[str, str]]]):
        """
        Create multiple tables in the order they are defined.
        """
        print("Creating tables in dependency order...")
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                for table_name, schema in table_schemas:
                    columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});")
                    print(f"Created table: {table_name}")
                conn.commit()
            print("✅ All tables created successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to create tables: {e}")
            raise


    # Data Insertion
    def insert_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Efficiently bulk insert a DataFrame into PostgreSQL.
        """
        if df.empty:
            print(f"[WARN] No records to insert into '{table_name}'.")
            return

        columns = list(df.columns)
        values = [tuple(x) for x in df.to_numpy()]
        insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s;"

        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                execute_values(cur, insert_query, values)
                conn.commit()
            print(f"✅ Inserted {len(df)} rows into '{table_name}'.")
        except Exception as e:
            print(f"[ERROR] Failed to insert data into '{table_name}': {e}")
            raise


    # Utility
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the current database."""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
        """
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute(query, (table_name,))
                return cur.fetchone()[0]
        except Exception as e:
            print(f"[ERROR] Error checking if table exists: {e}")
            return False