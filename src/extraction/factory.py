from .base import BaseExtractor
from ..config import Settings


def create_extractor(settings: Settings) -> BaseExtractor:
    """Instantiate the correct LLM provider based on settings.llm_provider."""
    if settings.llm_provider == "gemini":
        from .providers.gemini import GeminiExtractor
        return GeminiExtractor(api_key=settings.gemini_api_key, model=settings.gemini_model)

    if settings.llm_provider == "openai":
        from .providers.openai import OpenAIExtractor
        return OpenAIExtractor(api_key=settings.openai_api_key, model=settings.openai_model)

    if settings.llm_provider == "ollama":
        from .providers.ollama import OllamaExtractor
        return OllamaExtractor(base_url=settings.ollama_base_url, model=settings.ollama_model)

    raise ValueError(
        f"Unknown LLM provider: '{settings.llm_provider}'. "
        "Valid options: 'gemini', 'openai', 'ollama'."
    )
