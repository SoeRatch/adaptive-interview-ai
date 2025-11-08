# src/database/vector_db/langchain_faiss_handler.py

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # or OpenAIEmbeddings
from .langchain_base import LangChainVectorDBBase
from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings

class CachedHFEmbeddings(Embeddings):
    """LangChain-compatible wrapper for a preloaded SentenceTransformer."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class EmbeddingSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, embedding_model_name):
        if cls._instance is None:
            print(f"Initializing HuggingFaceEmbeddings: {embedding_model_name}")
            save_model_dir = f"data/models/{embedding_model_name}/base"
            model = SentenceTransformer(str(save_model_dir))
            cls._instance = CachedHFEmbeddings(model)
        return cls._instance


class LangChainFAISSHandler(LangChainVectorDBBase):
    """Handler for LangChain-based FAISS vector store."""
    def __init__(
            self,
            index_path: str,
            embedding_model_name: str
            ):
        self.index_path = Path(index_path)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.embedding_model = EmbeddingSingleton.get_instance(embedding_model_name)
        self.vector_store = None
        self.load()
    
    def getdb(self):
        return self.vector_store
    
    def load(self):
        """Load existing FAISS index if available; otherwise defer build until later."""
        if self.index_path.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            print("No FAISS index found. You can create one using build_index().")
            self.vector_store = None
    
    def build_index(self, texts, metadatas=None):
        """Build or rebuild the FAISS index from a list of texts."""
        print("Building FAISS index from corpus...")
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )
        self.save(self.index_path)

    def get_retriever(self, top_k=5):
        """Return LangChain retriever."""
        if not self.vector_store:
            raise ValueError("FAISS index not loaded or built.")
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})
    
    def save(self, path: str | Path = None):
        """Persist FAISS index to local storage."""
        if not self.vector_store:
            raise ValueError("No FAISS vector store to save.")
        path = Path(path or self.index_path)
        self.vector_store.save_local(str(path))
        print(f"✅ FAISS index saved to {path}")
