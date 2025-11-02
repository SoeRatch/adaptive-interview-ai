# src/database/vector_db/pinecone_handler.py

from pinecone import Pinecone, ServerlessSpec
from .base import VectorDBBase


class PineconeHandler(VectorDBBase):
    def __init__(self, api_key, index_name, dim=384):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            print(f"Creating Pinecone index '{index_name}'...")
            self.pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(index_name)
        print(f"Connected to Pinecone index '{index_name}'")

    def upsert_embeddings(self, ids, embeddings, metadata=None):
        """Insert embeddings with metadata into Pinecone."""
        vectors = [
            {
                "id": str(i),
                "values": emb.tolist(),
                "metadata": metadata[idx] if metadata else {},
            }
            for idx, (i, emb) in enumerate(zip(ids, embeddings))
        ]
        self.index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors into Pinecone index.")

    def search(self, query_embedding, k=5):
        """Search Pinecone index for top-k similar embeddings."""
        query_embedding = query_embedding.flatten().tolist()
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in results.matches
        ]
