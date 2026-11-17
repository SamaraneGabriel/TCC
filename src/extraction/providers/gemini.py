from google import genai
from google.genai import types

from ...usage import registrar
from ..base import BaseExtractor


class GeminiExtractor(BaseExtractor):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-3.1-flash-lite",
        project: str = "",
        location: str = "us-central1",
    ):
        # Com `project` preenchido, usa Vertex AI (billing no projeto GCP, autenticação via
        # ADC) em vez da Developer API com api_key — ver src/config.py::gcp_project.
        if project:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client(api_key=api_key)
        self.model = model
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

    def call(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config,
        )
        uso = response.usage_metadata
        if uso:
            registrar(
                self.model,
                prompt_tokens=uso.prompt_token_count,
                resposta_tokens=uso.candidates_token_count,
                total_tokens=uso.total_token_count,
            )
        return response.text
