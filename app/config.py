from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    siliconflow_api_key: str
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    aliyun_api_key: str = ""
    aliyun_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    embedding_model: str = "text-embedding-v4"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"

    qdrant_url: str = ""
    qdrant_path: str = "./qdrant_storage"
    qdrant_collection: str = "agent_memory"
    database_url: str = "sqlite:///./aml_memory.db"

    enable_llm_consolidation: bool = True
    llm_consolidation_max_tokens: int = 8192

    dense_weight: float = 0.50
    keyword_weight: float = 0.30
    recency_weight: float = 0.20
    dense_recall_multiplier: int = 8
    keyword_ngram_max: int = 3
    keyword_phrase_weight: float = 0.7
    search_dedup_threshold: float = 0.85
    # bge 风格检索指令：仅加在 embedding 环节的 query 侧（短查询×长文档时提升区分度），
    # 不用于 rerank / 关键词打分（交叉编码器与词法匹配都不需要前缀）。
    search_query_prefix: str = "Represent this sentence for searching relevant passages: "

    enable_raw_sentence_fallback: bool = True
    raw_fallback_per_chunk: int = 8
    raw_fallback_min_score: float = 0.3
    temporal_priority_boost: float = 0.25

    enable_episode_building: bool = True
    episode_candidate_limit: int = 12
    episode_fact_recall_multiplier: int = 3
    episode_max_facts: int = 20
    episode_group_bonus_per_fact: float = 0.03
    episode_group_bonus_cap: float = 0.15
    # episode 归属：text_similarity 低于该阈值且候选>=2 才直接新建（明确的"无关"）；
    # 高于该阈值但 <0.6（合并阈值）则进入模糊带走 LLM 判定。
    # 原值 0.25 会让词面 Jaccard 失真的情况（如语义 cosine 0.68 但 Jaccard 0.09）被误判为"新主题"。
    episode_create_sim_threshold: float = 0.05

    memory_system_key: str = ""
    log_level: str = "INFO"
    top_k_default: int = 100
    max_add_timeout_seconds: int = 1800
    max_search_timeout_seconds: int = 1800


settings = Settings()
