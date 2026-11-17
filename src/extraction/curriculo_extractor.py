from pathlib import Path

from ..debug import dbg
from ..schemas.curriculo import CurriculoEstruturado
from .base import BaseExtractor
from .parsing import ExtractionError, parse_llm_json

_PROMPT = (Path(__file__).parent / "prompts" / "curriculo_prompt.txt").read_text(encoding="utf-8")

_ETAPA = "extraction.curriculo"


def _sanitizar(text: str) -> str:
    # Aspas duplas no currículo são copiadas cruas para dentro das strings do JSON de resposta
    # e quebram o parse (visto em `“best practices"`, que fecha a string no meio).
    return text.replace('"', "'")


class CurriculoExtractor:
    def __init__(self, llm: BaseExtractor):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    def extract(self, text: str) -> CurriculoEstruturado:
        prompt = _PROMPT.replace("{texto}", _sanitizar(text))
        dbg(_ETAPA, f"chamando LLM ({len(text)} chars de entrada)")
        raw = self._llm.call(prompt)
        dbg(_ETAPA, f"resposta recebida ({len(raw)} chars)")
        resultado = parse_llm_json(raw, CurriculoEstruturado)
        dbg(_ETAPA, f"extraído: {resultado.nome or 'Nome não informado'} ({len(resultado.skills_tecnicas)} skills, {len(resultado.experiencias)} experiências)")
        return resultado

    async def aextract(self, text: str) -> CurriculoEstruturado:
        prompt = _PROMPT.replace("{texto}", _sanitizar(text))
        dbg(_ETAPA, f"chamando LLM async ({len(text)} chars de entrada)")
        raw = await self._llm.acall(prompt)
        dbg(_ETAPA, f"resposta recebida ({len(raw)} chars)")
        resultado = parse_llm_json(raw, CurriculoEstruturado)
        dbg(_ETAPA, f"extraído: {resultado.nome or 'Nome não informado'} ({len(resultado.skills_tecnicas)} skills, {len(resultado.experiencias)} experiências)")
        return resultado
