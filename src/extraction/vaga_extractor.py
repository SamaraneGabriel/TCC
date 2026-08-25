from pathlib import Path

from ..debug import dbg
from ..schemas.vaga import VagaEstruturada
from .base import BaseExtractor
from .parsing import ExtractionError, parse_llm_json

_PROMPT = (Path(__file__).parent / "prompts" / "vaga_prompt.txt").read_text(encoding="utf-8")

_ETAPA = "extraction.vaga"


class VagaExtractor:
    def __init__(self, llm: BaseExtractor):
        self._llm = llm

    def extract(self, text: str) -> VagaEstruturada:
        prompt = _PROMPT.replace("{texto}", text)
        dbg(_ETAPA, f"chamando LLM ({len(text)} chars de entrada)")
        raw = self._llm.call(prompt)
        dbg(_ETAPA, f"resposta recebida ({len(raw)} chars)")
        resultado = parse_llm_json(raw, VagaEstruturada)
        dbg(_ETAPA, f"extraído: {resultado.titulo} ({len(resultado.skills_obrigatorias)} skills obrigatórias)")
        return resultado

    async def aextract(self, text: str) -> VagaEstruturada:
        prompt = _PROMPT.replace("{texto}", text)
        dbg(_ETAPA, f"chamando LLM async ({len(text)} chars de entrada)")
        raw = await self._llm.acall(prompt)
        dbg(_ETAPA, f"resposta recebida ({len(raw)} chars)")
        resultado = parse_llm_json(raw, VagaEstruturada)
        dbg(_ETAPA, f"extraído: {resultado.titulo} ({len(resultado.skills_obrigatorias)} skills obrigatórias)")
        return resultado
