"""Dense retriever backed by Qdrant."""

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

COLLECTION = "multimodal_rag"


class Retriever:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.openai = OpenAI()

    def _embed_query(self, query: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        doc_name: str | None = None,
        top_k: int = 5,
        collection: str = COLLECTION,
    ) -> list[dict]:
        """Return top_k chunks most relevant to query, optionally filtered by doc_name."""
        query_vector = self._embed_query(query)

        search_filter = None
        if doc_name:
            search_filter = Filter(
                must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
            )

        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
        ).points

        return [{**r.payload, "score": r.score} for r in results]
