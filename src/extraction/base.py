import asyncio
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Abstract LLM provider. Implement `call` to add a new provider."""

    model: str | None = None
    """Nome do modelo usado, para rastreamento de proveniência (ex: payload do Qdrant)."""

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Send prompt to LLM and return the raw JSON string response."""
        ...

    async def acall(self, prompt: str) -> str:
        """Async variant — runs the sync `call` in a thread pool by default."""
        return await asyncio.to_thread(self.call, prompt)
