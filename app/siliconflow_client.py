import json
import logging
from typing import List, Optional

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class SiliconFlowClient:
    """Client for embedding, reranking, chat, and JSON-schema chat outputs."""

    def __init__(self):
        self.api_key = settings.siliconflow_api_key
        self.base_url = settings.siliconflow_base_url
        self.embed_client = OpenAI(
            api_key=settings.aliyun_api_key or self.api_key,
            base_url=settings.aliyun_base_url or self.base_url,
            timeout=60.0,
        )
        self.chat_client = OpenAI(
            api_key=settings.openai_api_key or self.api_key,
            base_url=settings.openai_base_url or self.base_url,
            timeout=60.0,
        )
        self.http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), 10):
            response = self.embed_client.embeddings.create(
                model=settings.embedding_model,
                input=texts[i : i + 10],
                encoding_format="float",
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        if not documents:
            return []
        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": True,
        }
        try:
            resp = self.http_client.post("/rerank", json=payload)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, messages: List[dict], temperature: float = 0.1, max_tokens: int = 512) -> str:
        response = self.chat_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat_structured(
        self,
        messages: List[dict],
        schema_name: str,
        schema: dict,
        max_tokens: int = 1024,
    ) -> dict:
        """Use OpenAI structured outputs, with JSON-object fallback for compatible APIs."""
        kwargs = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            response = self.chat_client.chat.completions.create(
                **kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
        except Exception as structured_error:
            logger.warning(f"Structured output unavailable; using JSON fallback: {structured_error}")
            response = self.chat_client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


sf_client = SiliconFlowClient()
