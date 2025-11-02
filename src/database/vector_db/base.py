# src/database/vector_db/base.py

from abc import ABC, abstractmethod

class VectorDBBase(ABC):
    """Abstract base class for all VectorDB backends."""

    @abstractmethod
    def upsert_embeddings(self, ids, vectors, metadata=None):
        """
        Insert or update embeddings in the vector database.
        
        Args:
            ids (list): Unique IDs for each embedding.
            vectors (np.ndarray): Embedding matrix of shape (n, dim).
            metadata (list[dict], optional): Metadata for each embedding.
        """
        pass

    @abstractmethod
    def search(self, query_vector, top_k=5):
        """
        Search for the most similar embeddings.

        Args:
            query_vector (np.ndarray): The query embedding.
            top_k (int): Number of nearest neighbors to return.
        """
        pass
