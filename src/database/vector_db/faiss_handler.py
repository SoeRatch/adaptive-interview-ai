# src/database/vector_db/faiss_handler.py

import os
import numpy as np
import faiss
import joblib
from pathlib import Path

from .base import VectorDBBase
from src.database.constants import (
    FAISS_INDEX_FILEPATH, FAISS_INDEX_DIMENENSIONS
)

class FAISSHandler(VectorDBBase):
    def __init__(self, index_path: str = FAISS_INDEX_FILEPATH, dim: int = FAISS_INDEX_DIMENENSIONS):
        self.dim = dim
        self.index_path = Path(index_path)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        self.index = None
        self.id_map = None
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load existing index if available, else initialize a new one."""
        if self.index_path.exists():
            print(f"Loading FAISS index from {self.index_path}")
            self.id_map = faiss.read_index(str(self.index_path))
        else:
            print(f"Creating new FAISS index (dim={self.dim})")
            base_index = faiss.IndexFlatL2(self.dim)  # L2 = cosine alternative
            self.id_map = faiss.IndexIDMap(base_index)

    def upsert_embeddings(self, ids: list[int], vectors: np.ndarray, metadata=None):
        """
        Insert or update embeddings in FAISS along with metadata.
        Args:
            ids (list[int]): Document IDs (from PostgreSQL)
            vectors (np.ndarray): Embeddings, shape (n, dim)
            metadata (list[dict]): Optional metadata for each embedding, For FAISS, metadata is stored externally (e.g. PostgreSQL).
        """
        ids = np.array(ids).astype(np.int64)
        vectors = np.array(vectors, dtype="float32")

        assert len(ids) == vectors.shape[0], "Number of IDs must match number of vectors"
        assert vectors.shape[1] == self.dim, f"Embedding dimensions mismatch (expected {self.dim})"


        # Add embeddings to FAISS index

        # FAISS assigns internal IDs automatically i.e sequentially (0, 1, 2, ...).
        # self.index.add(vectors)

        # Allows you to specify your own integer IDs for each vector.
        self.id_map.add_with_ids(vectors, ids)

        # Persist index and metadata
        faiss.write_index(self.id_map, str(self.index_path))
        print(f"Saved FAISS index at {str(self.index_path)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Search for top_k most similar vectors.

        Args:
            query_vector (np.ndarray): Single embedding vector.
            top_k (int): Number of nearest neighbors to return.
        
        Returns:
            list[int]: Top document IDs (FAISS returns these IDs).
            list[float]: Corresponding distances.
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.id_map.search(query_vector, top_k)
        return indices[0].tolist(), distances[0].tolist()
    
    def delete_embeddings(self, ids: list[int]):
        """Delete specific document IDs from FAISS index."""
        ids = np.array(ids).astype(np.int64)
        self.id_map.remove_ids(ids)
        faiss.write_index(self.id_map, str(self.index_path))
        print(f"Deleted {len(ids)} vectors (requested) from FAISS index.")

    def count(self):
        """Return number of vectors stored."""
        return self.id_map.ntotal
