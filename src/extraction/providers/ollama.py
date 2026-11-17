import httpx

from ..base import BaseExtractor


class OllamaExtractor(BaseExtractor):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self._base_url = base_url.rstrip("/")
        self.model = model

    def call(self, prompt: str) -> str:
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "format": "json", "stream": False},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]
