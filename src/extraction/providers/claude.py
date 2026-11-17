"""Claude pela API da Anthropic, com a interface `BaseExtractor` dos outros provedores.

Juiz alternativo na calibração (2026-11-10): o mesmo prompt congelado do juiz, outro
modelo. Escolha do autor: um modelo sem raciocínio, o Claude Haiku 5.5 com
`thinking: disabled` (aceito com esforço `high` ou abaixo; o padrão é `medium`).

- **Sem temperatura.** A API não aceita mais `temperature`; o julgamento não é
  determinístico.
- **Sem modelo reserva em caso de recusa.** A recusa vira `ExtractionError` do par, que o
  juiz tenta de novo e, persistindo, registra como falha.
"""
import anthropic

from ...usage import registrar
from ..base import BaseExtractor
from ..parsing import ExtractionError


class ClaudeExtractor(BaseExtractor):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-5-5",
        raciocinio: bool = False,
        max_tokens: int = 8000,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        if raciocinio:
            self._thinking = {"type": "adaptive"}
        elif model == "claude-sonnet-5-5":
            # No Sonnet 5.5, "disabled" é rejeitado (400); o modo sem raciocínio é
            # "between_tools", aceito com esforço high ou abaixo (o padrão é high).
            self._thinking = {"type": "between_tools"}
        else:
            self._thinking = {"type": "disabled"}
        self._max_tokens = max_tokens

    def call(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            thinking=self._thinking,
            messages=[{"role": "user", "content": prompt}],
        )
        uso = response.usage
        if uso:
            registrar(
                self.model,
                prompt_tokens=uso.input_tokens,
                resposta_tokens=uso.output_tokens,
                total_tokens=(uso.input_tokens or 0) + (uso.output_tokens or 0),
            )
        if response.stop_reason == "refusal":
            categoria = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise ExtractionError(f"recusa do modelo (categoria: {categoria})")
        if response.stop_reason == "max_tokens":
            raise ExtractionError("resposta cortada por max_tokens")
        return "".join(b.text for b in response.content if b.type == "text")
