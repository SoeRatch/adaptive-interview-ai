# src/topic_modeling/embedding_storage.py
"""
Unified Embedding Storage System
--------------------------------
Supports:
  - Local storage (npz / split)
  - Vector DB storage (faiss / pinecone / qdrant)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.topic_modeling.constants import (
    EMBEDDINGS_OUTPUT_NPZ,
    EMBEDDINGS_OUTPUT_NPY,
    EMBEDDINGS_OUTPUT_METADATA_PARQUET
)

class EmbeddingStorage:
    def __init__(self, data_dir: str = "data/embeddings"):
        """
        Args:
            data_dir (str): Local path to store embeddings or metadata.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # SAVE
    def save(
        self,
        embeddings: np.ndarray,
        documents: list[str],
        urls: list[str] = None,
        sources: list[str] = None,
        titles: list[str] = None,
        mode: str = "npz"
    ):
        """
        Save embeddings and metadata.
        Args:
            embeddings: np.ndarray of shape (n_docs, n_dim)
            documents: list of text docs
            urls, sources, titles: optional metadata
            mode: 'npz', 'split', or 'vector_db'
        """
        if mode == "npz":
            np.savez_compressed(
                self.data_dir / EMBEDDINGS_OUTPUT_NPZ,
                embeddings=embeddings,
                documents=np.array(documents, dtype=object),
                urls=np.array(urls, dtype=object) if urls is not None else None,
                sources=np.array(sources, dtype=object) if sources is not None else None,
                titles=np.array(titles, dtype=object) if titles is not None else None,
                )

            print(f"✅ Saved all embeddings data in {self.data_dir / EMBEDDINGS_OUTPUT_NPZ}")
            # np.savez_compressed() creates a compressed .npz that saves multiple NumPy arrays into one file.
            # It applies ZIP compression internally, resulting in smaller file size than a plain .npy.
            # Instead of saving embeddings.npy, ids.npy, urls.npy, etc. separately, save just one file

        elif mode == "split":
            np.save(self.data_dir / EMBEDDINGS_OUTPUT_NPY, embeddings)
            meta_df = pd.DataFrame({
                "text": documents,
                "url": urls if urls is not None else [None] * len(documents),
                "source": sources if sources is not None else [None] * len(documents),
                "title": titles if titles is not None else [None] * len(documents),
            })
            meta_df.to_parquet(self.data_dir / EMBEDDINGS_OUTPUT_METADATA_PARQUET, index=False)
            print(f"✅ Saved {EMBEDDINGS_OUTPUT_NPY} and {EMBEDDINGS_OUTPUT_METADATA_PARQUET} in {self.data_dir}")

        else:
            raise ValueError("Invalid mode. Choose 'npz' or 'split'.")


    # LOAD
    def load(self):
        """
        Auto-detect and load embeddings + metadata.
        Returns: dict with keys: embeddings, documents, urls, sources, titles
        """
        npz_path = self.data_dir / EMBEDDINGS_OUTPUT_NPZ
        split_embeddings = self.data_dir / EMBEDDINGS_OUTPUT_NPY
        split_meta = self.data_dir / EMBEDDINGS_OUTPUT_METADATA_PARQUET

        if npz_path.exists():
            emb_data = np.load(npz_path, allow_pickle=True)
            print(f"Loaded data from {npz_path}")
            return {
                "embeddings": emb_data["embeddings"],
                "documents": emb_data["documents"],
                "urls": emb_data["urls"] if "urls" in emb_data.files else None,
                "sources": emb_data["sources"] if "sources" in emb_data.files else None,
                "titles": emb_data["titles"] if "titles" in emb_data.files else None,
            }

        elif split_embeddings.exists() and split_meta.exists():
            embeddings = np.load(split_embeddings, mmap_mode="r")
            meta_df = pd.read_parquet(split_meta)
            print(f"Loaded embeddings.npy and metadata.parquet from {self.data_dir}")
            return {
                "embeddings": embeddings,
                "documents": meta_df["text"].tolist(),
                "urls": meta_df["url"].tolist() if "url" in meta_df else None,
                "sources": meta_df.get("source"),
                "titles": meta_df.get("title"),
            }

        else:
            raise FileNotFoundError(f"No embedding data found in {self.data_dir}")


