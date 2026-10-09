from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SiliconFlow API (used for reranker)
    siliconflow_api_key: str
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # Alibaba Cloud (Bailian) API (used for text-embedding-v4; required for AML academic track)
    aliyun_api_key: str = ""
    aliyun_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # OpenAI-compatible API (used for LLM consolidation; required for AML academic track)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Model names
    embedding_model: str = "text-embedding-v4"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"

    # Qdrant configuration
    qdrant_url: str = ""
    qdrant_path: str = "./qdrant_storage"
    qdrant_collection: str = "agent_memory"

    # Database configuration (SQLite for local dev, PostgreSQL for Docker deployment)
    database_url: str = "sqlite:///./aml_memory.db"

    # Memory consolidation
    enable_llm_consolidation: bool = True
    llm_consolidation_max_tokens: int = 1024

    # Cross-message association / relation extraction
    enable_relation_classification: bool = True
    relation_classification_batch_size: int = 10
    relation_confidence_threshold: int = 70
    related_candidates_max: int = 15
    dense_recall_top_k: int = 20
    keyword_recall_top_k: int = 20
    structured_recall_limit: int = 20
    equivalent_similarity_threshold: float = 0.92

    # Direction A: soft demotion factor for facts superseded by a newer value.
    # Kept >0 so the audit trail stays retrievable (no hard filtering).
    superseded_score_penalty: float = 0.5

    # Hybrid retrieval weights
    dense_weight: float = 0.50
    keyword_weight: float = 0.30
    recency_weight: float = 0.20
    dense_recall_multiplier: int = 8

    # Keyword scoring enhancement
    keyword_ngram_max: int = 3
    keyword_phrase_weight: float = 0.7

    # Memory storage fallback
    enable_raw_sentence_fallback: bool = True

    # Search result deduplication
    search_dedup_threshold: float = 0.85

    # API authentication for AML platform calls (Token / Bearer / X-Api-Key).
    # If empty, no authentication is required (useful for public smoke tests).
    memory_system_key: str = ""

    # Raw sentence fallback controls
    raw_fallback_per_chunk: int = 8
    raw_fallback_min_score: float = 0.3
    temporal_priority_boost: float = 0.25

    # App behavior
    log_level: str = "INFO"
    top_k_default: int = 100
    max_add_timeout_seconds: int = 1800
    max_search_timeout_seconds: int = 1800


settings = Settings()
