from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SiliconFlow API
    siliconflow_api_key: str
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # Model names
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Qdrant configuration
    qdrant_url: str = ""
    qdrant_path: str = "./qdrant_storage"
    qdrant_collection: str = "agent_memory"

    # Database configuration (SQLite for local dev, PostgreSQL for Docker deployment)
    database_url: str = "sqlite:///./aml_memory.db"

    # Memory consolidation
    enable_llm_consolidation: bool = True
    llm_consolidation_max_tokens: int = 1024

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

    # App behavior
    log_level: str = "INFO"
    top_k_default: int = 100
    max_add_timeout_seconds: int = 1800
    max_search_timeout_seconds: int = 1800


settings = Settings()
