from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_api_key_fallback: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "low"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "data/chroma"
    raw_dir: str = "data/raw"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    retriever_k: int = 20
    fused_k: int = 20
    context_k: int = 5
    rrf_k: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
