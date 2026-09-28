import json
import logging
from typing import List, Optional
import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

logger = logging.getLogger(__name__)


class SiliconFlowClient:
    """Thin client for SiliconFlow embedding, reranker and chat APIs."""

    def __init__(self):
        self.api_key = settings.siliconflow_api_key
        self.base_url = settings.siliconflow_base_url
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the configured embedding model."""
        if not texts:
            return []
        try:
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """Rerank documents with respect to query."""
        if not documents:
            return []
        top_n = top_n or len(documents)
        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        try:
            resp = self.http_client.post("/rerank", json=payload)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # Fallback: return empty rerank signal so caller can keep original order
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, messages: List[dict], temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Call SiliconFlow chat completions. Used for async fact/entity extraction."""
        response = self.openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


sf_client = SiliconFlowClient()
