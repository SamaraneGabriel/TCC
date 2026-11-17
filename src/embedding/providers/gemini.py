from google import genai
from google.genai import types

from ...debug import dbg
from ...usage import registrar
from ..base import BaseEmbedder

_ETAPA = "embedding.gemini"


class GeminiEmbedder(BaseEmbedder):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        project: str = "",
        location: str = "us-central1",
    ):
        # Ver GeminiExtractor — mesma lógica de escolha Vertex AI vs Developer API.
        if project:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client(api_key=api_key)
        self.model = model
        self._dimension = dimension
        self._config = types.EmbedContentConfig(output_dimensionality=dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        dbg(_ETAPA, f"vetorizando {len(texts)} texto(s) (modelo={self.model}, dim={self._dimension})")
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=self._config,
        )
        meta = response.metadata
        registrar(
            self.model,
            # A resposta de embedding não traz token count; o SDK reporta só caracteres faturáveis.
            caracteres_faturaveis=getattr(meta, "billable_character_count", 0) or 0,
        )
        vetores = [e.values for e in response.embeddings]
        dbg(_ETAPA, f"{len(vetores)} vetor(es) recebido(s)")
        return vetores
