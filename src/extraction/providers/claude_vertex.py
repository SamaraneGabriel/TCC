"""Claude (Anthropic) via Vertex AI, com a mesma interface `BaseExtractor` dos outros provedores.

Usado como juiz alternativo na calibração (2026-11-10): o mesmo prompt congelado do juiz,
outro modelo. Diferenças em relação ao Gemini que importam para a comparação:

- **Sem temperatura.** Claude Opus 5.5 rejeita `temperature` (400); o julgamento não é
  determinístico. Rodar duas vezes pode dar notas diferentes.
- **Raciocínio sempre ligado.** A profundidade é controlada por `effort` (padrão do modelo:
  `medium`). Os tokens de raciocínio são cobrados como saída.
- **Sem modelo reserva em caso de recusa.** Um reserva trocaria o juiz no meio da rodada; a
  recusa vira erro do par (`ExtractionError`), que o juiz tenta de novo e, persistindo,
  registra como falha.

Autenticação: ADC do GCP (a mesma dos clientes Gemini via Vertex); nenhuma chave Anthropic.
"""
from anthropic import AnthropicVertex

from ...usage import registrar
from ..base import BaseExtractor
from ..parsing import ExtractionError


class ClaudeVertexExtractor(BaseExtractor):
    def __init__(
        self,
        project: str,
        model: str = "claude-opus-5-5",
        region: str = "global",
        effort: str = "medium",
        max_tokens: int = 16000,
    ):
        self._client = AnthropicVertex(project_id=project, region=region)
        self.model = model
        self.effort = effort
        self._max_tokens = max_tokens

    def call(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            output_config={"effort": self.effort},
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
