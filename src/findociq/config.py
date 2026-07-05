from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    visual_collection: str = "findociq_pages_visual"
    text_collection: str = "findociq_pages_text"

    # Generation
    vlm_provider: str = "gemini"  # gemini | ollama | anthropic
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"

    # Embedding models
    visual_model: str = "vidore/colqwen2.5-v0.2"
    text_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval
    top_k: int = 5
    rrf_k: int = 60

    # Storage
    pages_dir: Path = PROJECT_ROOT / "data" / "pages"
    render_dpi: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()
