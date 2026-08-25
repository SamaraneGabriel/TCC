from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()

    
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "gemini"

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "No Key Provided")
    gemini_model: str = "gemini-3.1-flash-lite"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    embedding_provider: str = "gemini"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768

    qdrant_local_path: str = ".qdrant_storage"

    debug: bool = False


settings = Settings()
