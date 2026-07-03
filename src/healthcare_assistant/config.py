import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiConfig(BaseModel):
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"


class RetrievalConfig(BaseModel):
    top_k: int = 3


class StorageConfig(BaseModel):
    db_path: Path = Path("data/healthcare.db")
    embeddings_cache_path: Path = Path("data/embeddings_cache.json")


class AssistantConfig(BaseModel):
    max_function_call_rounds: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    gemini: GeminiConfig = GeminiConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    storage: StorageConfig = StorageConfig()
    assistant: AssistantConfig = AssistantConfig()


def load_settings(toml_path: Path = Path("configs/app.toml")) -> Settings:
    overrides = tomllib.loads(toml_path.read_text()) if toml_path.exists() else {}
    return Settings(**overrides)
