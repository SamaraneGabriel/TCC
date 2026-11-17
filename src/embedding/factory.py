from .base import BaseEmbedder
from ..config import Settings


def create_embedder(settings: Settings) -> BaseEmbedder:
    """Instantiate the correct embedding provider based on settings.embedding_provider."""
    if settings.embedding_provider == "gemini":
        from .providers.gemini import GeminiEmbedder
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
            dimension=settings.embedding_dimension,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )

    raise ValueError(
        f"Unknown embedding provider: '{settings.embedding_provider}'. "
        "Valid options: 'gemini'."
    )
