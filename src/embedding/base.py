import asyncio
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract embedding provider. Implement `embed_batch` to add a new provider."""

    model: str | None = None
    """Nome do modelo usado, para rastreamento de proveniência (ex: payload do Qdrant)."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text in the same order."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Delegates to `embed_batch`."""
        return self.embed_batch([text])[0]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Async variant — runs the sync `embed_batch` in a thread pool by default."""
        return await asyncio.to_thread(self.embed_batch, texts)

    async def aembed(self, text: str) -> list[float]:
        result = await self.aembed_batch([text])
        return result[0]
