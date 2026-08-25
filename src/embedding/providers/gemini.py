from google import genai
from google.genai import types

from ...debug import dbg
from ..base import BaseEmbedder

_ETAPA = "embedding.gemini"


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimension: int = 768):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._config = types.EmbedContentConfig(output_dimensionality=dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        dbg(_ETAPA, f"vetorizando {len(texts)} texto(s) (modelo={self._model}, dim={self._dimension})")
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=self._config,
        )
        vetores = [e.values for e in response.embeddings]
        dbg(_ETAPA, f"{len(vetores)} vetor(es) recebido(s)")
        return vetores
