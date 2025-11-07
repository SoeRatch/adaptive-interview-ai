# src/database/vector_db/langchain_base.py

from abc import ABC, abstractmethod

class LangChainVectorDBBase(ABC):
    """Abstract base class for LangChain-based VectorDB backends."""

    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass

    @abstractmethod
    def build_index(self, texts, metadatas=None):
        """Build or rebuild the FAISS index from text data."""
        pass

    @abstractmethod
    def get_retriever(self, top_k=5):
        """Return a LangChain-compatible retriever."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass

