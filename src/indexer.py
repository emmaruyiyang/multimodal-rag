"""Qdrant indexer: embed chunks and store with metadata."""

import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_SIZE = 1536  # text-embedding-3-small
COLLECTION = "multimodal_rag"
EMBED_BATCH_SIZE = 100


class Indexer:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.openai = OpenAI()

    def _ensure_collection(self, collection: str = COLLECTION) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            response = self.openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            embeddings.extend(e.embedding for e in response.data)
        return embeddings

    def close(self) -> None:
        self.client.close()

    def index_chunks(self, chunks: list[dict], collection: str = COLLECTION) -> int:
        """Embed and upsert chunks into Qdrant. Returns number of points stored."""
        if not chunks:
            return 0

        self._ensure_collection(collection)

        texts = [c["chunk_text"] for c in chunks]
        embeddings = self._embed(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk['doc_name']}:{chunk['chunk_index']}")),
                vector=embedding,
                payload={
                    "doc_name": chunk["doc_name"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["chunk_text"],
                    "chunk_pages": chunk.get("chunk_pages", []),
                    "titles_context": chunk.get("titles_context", ""),
                    "type": chunk.get("type", "text"),
                    "image_path": chunk.get("image_path"),
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.client.upsert(collection_name=collection, points=points)
        return len(points)
