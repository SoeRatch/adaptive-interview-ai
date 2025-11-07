# src/database/langchain_vector_db_factory.py
from .vector_db.langchain_faiss_handler import LangChainFAISSHandler
from src.database.constants import (
    FAISS_INDEX_FILEPATH, EMBEDDING_MODEL_NAME
)

def get_langchain_vector_db_handler(backend="faiss", config=None):
    config = config or {}
    backend = backend.lower()

    if backend == "faiss":
        return LangChainFAISSHandler(
            index_path = FAISS_INDEX_FILEPATH,
            embedding_model_name=config.get("embedding_model", EMBEDDING_MODEL_NAME)
            )
    else:
        raise ValueError(f"Unsupported LangChain backend '{backend}'")
