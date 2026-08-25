"""
LLM-as-judge: avalia, numa única chamada por vaga, a relevância de cada
candidato retornado pelo pipeline de matching + sugere sua própria ordenação.

Design (ver Fase 7 do plano de implementação):
- Candidatos são apresentados EMBARALHADOS e SEM NOME (rótulos "candidato_N"),
  pra evitar viés de posição/ancoragem e viés associado a nome/identidade.
- Uma chamada julga todos os candidatos de uma vaga de uma vez (lote), não uma
  chamada por par — importante dado o rate limit gratuito do Gemini.
- Captura tokens/latência de cada chamada (response.usage_metadata do SDK
  google-genai) pra permitir tracking de custo/uso do pipeline de avaliação.
"""
import random
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..debug import dbg
from ..extraction.parsing import parse_llm_json
from ..schemas.vaga import VagaEstruturada
from ..vectorstore.store import SearchResult

_ETAPA = "evaluation.judge"


class AvaliacaoCandidato(BaseModel):
    candidato: str
    grau: int
    justificativa: str


class RespostaJudge(BaseModel):
    avaliacoes: list[AvaliacaoCandidato]
    ordem_sugerida: list[str]


@dataclass
class Julgamento:
    graus: dict[str, int]  # id real do candidato -> grau (0/1/2)
    justificativas: dict[str, str]  # id real do candidato -> justificativa
    ordem_sugerida: list[str]  # ids reais, melhor pro pior, segundo a LLM
    usage: dict  # tokens/modelo/latencia da chamada


class GeminiJudge:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

    def julgar(self, vaga: VagaEstruturada, candidatos: list[SearchResult]) -> Julgamento:
        embaralhados = candidatos.copy()
        random.shuffle(embaralhados)
        candidatos_rotulados = [(f"candidato_{i + 1}", c) for i, c in enumerate(embaralhados)]
        rotulo_para_id = {rotulo: c.id for rotulo, c in candidatos_rotulados}

        prompt = _montar_prompt(vaga, candidatos_rotulados)
        dbg(_ETAPA, f"julgando {len(candidatos)} candidato(s) pra vaga {vaga.titulo!r} (embaralhados)")

        inicio = time.monotonic()
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=self._config
        )
        latencia_s = time.monotonic() - inicio
        dbg(_ETAPA, f"resposta recebida em {latencia_s:.2f}s ({len(response.text)} chars)")

        resposta = parse_llm_json(response.text, RespostaJudge)

        graus: dict[str, int] = {}
        justificativas: dict[str, str] = {}
        for av in resposta.avaliacoes:
            id_real = rotulo_para_id.get(av.candidato)
            if id_real is None:
                dbg(_ETAPA, f"rótulo desconhecido na resposta, ignorado: {av.candidato!r}")
                continue
            graus[id_real] = av.grau
            justificativas[id_real] = av.justificativa

        ordem_sugerida = [rotulo_para_id[r] for r in resposta.ordem_sugerida if r in rotulo_para_id]

        usage_meta = response.usage_metadata
        usage = {
            "modelo": self._model,
            "prompt_tokens": usage_meta.prompt_token_count if usage_meta else None,
            "resposta_tokens": usage_meta.candidates_token_count if usage_meta else None,
            "total_tokens": usage_meta.total_token_count if usage_meta else None,
            "latencia_s": round(latencia_s, 2),
        }
        dbg(_ETAPA, f"julgamento: {len(graus)} grau(s), tokens={usage['total_tokens']}")

        return Julgamento(
            graus=graus,
            justificativas=justificativas,
            ordem_sugerida=ordem_sugerida,
            usage=usage,
        )


def _montar_prompt(vaga: VagaEstruturada, candidatos_rotulados: list[tuple[str, SearchResult]]) -> str:
    skills_obrig = ", ".join(vaga.skills_obrigatorias) or "não especificadas"
    skills_desej = ", ".join(vaga.skills_desejadas) or "não especificadas"
    responsabilidades = "; ".join(vaga.responsabilidades) or "não especificadas"

    blocos_candidatos = []
    for rotulo, c in candidatos_rotulados:
        blocos_candidatos.append(
            f"[{rotulo}]\n"
            f"Área: {c.area_principal or 'não informada'}\n"
            f"Senioridade: {c.senioridade_estimada or 'não informada'}\n"
            f"Localidade: {c.localidade or 'não informada'}\n"
            f"Modelo de trabalho: {c.modelo_trabalho_preferido or 'não informado'}\n"
            f"Skills técnicas: {', '.join(c.skills_tecnicas) or 'não informadas'}"
        )

    return f"""Você é um avaliador especialista em recrutamento técnico. Avalie, de forma objetiva e imparcial, quão adequado cada candidato é para a vaga abaixo. Não invente informação que não esteja nos dados apresentados.

VAGA
Título: {vaga.titulo}
Área: {vaga.area or 'não informada'}
Senioridade: {vaga.senioridade or 'não informada'}
Localidade: {vaga.localidade or 'não informada'}
Modelo de trabalho: {vaga.modelo_trabalho or 'não informado'}
Skills obrigatórias: {skills_obrig}
Skills desejadas: {skills_desej}
Responsabilidades: {responsabilidades}

CANDIDATOS (ordem embaralhada, não reflete nenhum ranking prévio)
{chr(10).join(blocos_candidatos)}

Para cada candidato, atribua um grau de relevância pra essa vaga:
0 = não relevante (área/skills incompatíveis com a vaga)
1 = parcialmente relevante (alguma sobreposição, mas não é um bom match — ex: área certa mas senioridade muito diferente, ou skills parcialmente sobrepostas)
2 = relevante (bom match — área, senioridade e skills compatíveis)

Retorne APENAS um JSON no formato:
{{
  "avaliacoes": [
    {{"candidato": "candidato_1", "grau": 0, "justificativa": "..."}}
  ],
  "ordem_sugerida": ["candidato_3", "candidato_1"]
}}

"ordem_sugerida" deve listar TODOS os {len(candidatos_rotulados)} candidatos (rótulos [candidato_N]), do melhor pro pior encaixe pra essa vaga na sua opinião — não precisa seguir a ordem em que foram apresentados acima."""
