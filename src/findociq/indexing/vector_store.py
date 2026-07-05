"""Stage 2c - Vector store: Qdrant collections for both retrieval paths.

Visual collection uses Qdrant's native multivector support with MAX_SIM
comparator - late interaction scoring happens inside the database, no
client-side reranking loop needed. Text collection is a standard dense
cosine index.

Free options: local Docker (docker compose up -d qdrant) or a Qdrant Cloud
free 1 GB cluster.
"""

import uuid

from qdrant_client import QdrantClient, models

from findociq.config import get_settings
from findociq.ingestion.pdf_processor import PageRecord


def _point_id(page_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, page_id))


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def ensure_collections(self, text_dim: int) -> None:
        if not self.client.collection_exists(self.settings.visual_collection):
            self.client.create_collection(
                collection_name=self.settings.visual_collection,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
        if not self.client.collection_exists(self.settings.text_collection):
            self.client.create_collection(
                collection_name=self.settings.text_collection,
                vectors_config=models.VectorParams(
                    size=text_dim, distance=models.Distance.COSINE
                ),
            )

    def upsert_pages(
        self,
        records: list[PageRecord],
        visual_embeddings: list[list[list[float]]],
        text_embeddings: list[list[float]],
    ) -> None:
        payloads = [
            {
                "page_id": r.page_id,
                "doc_name": r.doc_name,
                "page_number": r.page_number,
                "image_path": str(r.image_path),
                "text": r.text[:2000],
            }
            for r in records
        ]
        self.client.upsert(
            collection_name=self.settings.visual_collection,
            points=[
                models.PointStruct(id=_point_id(r.page_id), vector=vec, payload=payload)
                for r, vec, payload in zip(records, visual_embeddings, payloads)
            ],
        )
        self.client.upsert(
            collection_name=self.settings.text_collection,
            points=[
                models.PointStruct(id=_point_id(r.page_id), vector=vec, payload=payload)
                for r, vec, payload in zip(records, text_embeddings, payloads)
            ],
        )

    def search_visual(self, query_multivector: list[list[float]], limit: int) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.settings.visual_collection,
            query=query_multivector,
            limit=limit,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in result.points]

    def search_text(self, query_vector: list[float], limit: int) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.settings.text_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in result.points]
