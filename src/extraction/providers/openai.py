from src.extraction.providers.openai import OpenAI

from ..base import BaseExtractor


class OpenAIExtractor(BaseExtractor):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um extrator de dados estruturados. Retorne apenas JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
