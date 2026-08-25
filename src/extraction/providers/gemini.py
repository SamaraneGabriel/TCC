from google import genai
from google.genai import types

from ..base import BaseExtractor


class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

    def call(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config,
        )
        return response.text
