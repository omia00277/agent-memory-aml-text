import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024  # BAAI/bge-m3 dense dimension


class QdrantMemoryStore:
    def __init__(
        self,
        collection: str = settings.qdrant_collection,
    ):
        self.collection = collection
        if settings.qdrant_url:
            logger.info(f"Connecting to remote Qdrant at {settings.qdrant_url}")
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            logger.info(f"Using local Qdrant storage at {settings.qdrant_path}")
            os.makedirs(settings.qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=settings.qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    def upsert(
        self,
        user_id: str,
        session_id: str,
        content: str,
        vector: List[float],
        point_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        unit_type: str = "fact",
        source_ts: Optional[int] = None,
        source_request_id: Optional[str] = None,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        value: Optional[str] = None,
        valid: bool = True,
    ) -> str:
        point_id = point_id or str(uuid.uuid4())
        created_at = created_at or datetime.now(timezone.utc)
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "content": content,
            "unit_type": unit_type,
            "created_at": created_at.isoformat(),
            "valid": valid,
        }
        if source_ts is not None:
            payload["source_ts"] = source_ts
        if source_request_id is not None:
            payload["source_request_id"] = source_request_id
        if entity is not None:
            payload["entity"] = entity
        if attribute is not None:
            payload["attribute"] = attribute
        if value is not None:
            payload["value"] = value
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int = 100,
        limit: Optional[int] = None,
        valid_only: bool = False,
    ) -> List[dict]:
        limit = limit or top_k
        must_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        ]
        if valid_only:
            must_conditions.append(
                FieldCondition(
                    key="valid",
                    match=MatchValue(value=True),
                )
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": str(point.id),
                "content": point.payload.get("content"),
                "score": point.score,
                "created_at": point.payload.get("created_at"),
                "unit_type": point.payload.get("unit_type", "fact"),
                "source_ts": point.payload.get("source_ts"),
                "source_request_id": point.payload.get("source_request_id"),
                "valid": point.payload.get("valid", True),
                "entity": point.payload.get("entity"),
                "attribute": point.payload.get("attribute"),
                "value": point.payload.get("value"),
            }
            for point in response.points
        ]


qdrant_store = QdrantMemoryStore()
