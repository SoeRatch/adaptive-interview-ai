# src/database/vector_db_factory.py

from .vector_db.faiss_handler import FAISSHandler
from .vector_db.pinecone_handler import PineconeHandler
# from .vector_db.qdrant_handler import QdrantHandler

def get_vector_db_handler(backend="faiss", config=None):
    config = config or {}
    backend = backend.lower()

    if backend == "faiss":
        return FAISSHandler(dim=config.get("dim", 384))
    elif backend == "pinecone":
        return PineconeHandler(**config)
    else:
        raise ValueError("Unsupported backend.")