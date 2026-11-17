"""Juiz LLM item a item contra a rubrica da vaga.

Julga **um par vaga-currículo por chamada**: lê a rubrica da vaga e o texto
original do currículo, marca o checklist (atende / parcial / não atende) para
cada item e dá uma nota de 0 a 3. É a mesma tarefa que o anotador humano faz no
`app_anotacao.py`, com o mesmo texto de critérios.

Decisões de projeto (tomadas com o usuário em 2026-11-10):

- **Prompt congelado.** `prompts/juiz_prompt.txt` é lido como está. O trecho de
  critérios dele é exatamente o que a interface de anotação mostra ao humano
  (`app_pages/_comum.py::criterios_de_julgamento`): se humano e juiz seguissem
  textos diferentes, a concordância mediria a diferença entre os textos, não o
  juiz. O hash do arquivo vai em cada julgamento, para provar que não mudou.
- **Rubrica revisada nas vagas de calibração.** O juiz recebe a rubrica que
  estiver em `rubricas_vagas.csv` — nas 3 vagas de calibração, a revisada por
  humano, a mesma que o anotador usa.
- **A nota é do juiz.** Ele marca o checklist e escolhe a nota, como o humano.
  A nota que a regra do checklist sugeriria (`anotacao.nota_sugerida`) é gravada
  ao lado só para auditoria; não substitui a do juiz.
- **JSON + Pydantic, com nova tentativa.** Resposta inválida, item da rubrica
  faltando, repetido ou desconhecido dispara nova chamada, até
  `MAX_TENTATIVAS` no total.

O juiz não recebe estrato, posição no ranking nem score: `montar_prompt` só
conhece vaga, rubrica e texto do currículo.

Nada aqui abre o Qdrant nem lê o currículo do disco: quem chama resolve o texto
(`anotacao.texto_do_curriculo`) e passa pronto.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from ...debug import dbg
from ...extraction.base import BaseExtractor
from ...extraction.parsing import ExtractionError, parse_llm_json
from .anotacao import nota_sugerida
from .rubrica import ItemQualificacao, RubricaVaga

_ETAPA = "evaluation.plano.juiz"

PROMPT_JUIZ = Path(__file__).parent / "prompts" / "juiz_prompt.txt"
MAX_TENTATIVAS = 3

Atendimento = Literal["atende", "parcial", "nao_atende"]


class JuizError(Exception):
    """O juiz não devolveu um julgamento válido em `MAX_TENTATIVAS` chamadas."""


# ----------------------------------------------------------------------
# Resposta do LLM


# Variações de escrita que o modelo produz de forma determinística (temperatura 0
# repete o erro em toda nova tentativa). Só entram grafias sem ambiguidade.
# Visto na calibração de 2026-11-10: 12 de 202 pares com "nao_atendimento".
_GRAFIAS_ATENDIMENTO = {"nao_atendimento": "nao_atende"}


class ItemJulgado(BaseModel):
    item_id: str
    evidencia: str = ""
    atendimento: Atendimento

    @field_validator("atendimento", mode="before")
    @classmethod
    def _normalizar_grafia(cls, v):
        if isinstance(v, str):
            return _GRAFIAS_ATENDIMENTO.get(v.strip(), v.strip())
        return v


class _RespostaJuiz(BaseModel):
    itens: list[ItemJulgado]
    justificativa: str = ""
    nota: int = Field(ge=0, le=3)

    @field_validator("nota", mode="before")
    @classmethod
    def _nota_inteira(cls, v):
        # O schema pede inteiro, mas "2" em string é o mesmo julgamento.
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return v


class JulgamentoPar(BaseModel):
    """Um julgamento completo de um par vaga-currículo, pronto para gravar."""

    vaga_id: str
    curriculo_id: str
    nota: int
    justificativa: str
    itens: list[ItemJulgado]
    nota_sugerida_checklist: int | None
    modelo: str | None
    prompt_sha256: str
    tentativas: int
    data: str

    def atendimentos(self) -> dict[str, str]:
        return {i.item_id: i.atendimento for i in self.itens}


# ----------------------------------------------------------------------
# Prompt


def hash_prompt(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _formatar_itens(itens: list[ItemQualificacao]) -> str:
    if not itens:
        return "(nenhum)"
    return "\n".join(f"- {i.item_id} [{i.categoria}] {i.descricao}" for i in itens)


def conferir_itens(resposta: _RespostaJuiz, rubrica: RubricaVaga) -> list[str]:
    """Problemas de correspondência entre a resposta e a rubrica. Vazio = ok."""
    esperados = [i.item_id for i in rubrica.itens]
    recebidos = [i.item_id for i in resposta.itens]
    problemas = []
    faltando = [i for i in esperados if i not in recebidos]
    if faltando:
        problemas.append(f"itens faltando: {faltando}")
    extras = sorted({i for i in recebidos if i not in esperados})
    if extras:
        problemas.append(f"itens fora da rubrica: {extras}")
    repetidos = sorted({i for i in recebidos if recebidos.count(i) > 1})
    if repetidos:
        problemas.append(f"itens repetidos: {repetidos}")
    return problemas


class JuizRubrica:
    """Juiz LLM item a item. Recebe o `BaseExtractor` do projeto (Gemini via
    Vertex, com o modelo do `.env`) — este módulo não escolhe provider."""

    def __init__(self, llm: BaseExtractor, prompt_path: Path | str = PROMPT_JUIZ) -> None:
        self._llm = llm
        self._template = Path(prompt_path).read_text(encoding="utf-8")
        self.prompt_sha256 = hash_prompt(self._template)

    @property
    def modelo(self) -> str | None:
        return getattr(self._llm, "model", None)

    def montar_prompt(self, vaga: dict, rubrica: RubricaVaga, texto_curriculo: str) -> str:
        """`vaga` é o registro de `vagas_estudo.jsonl` (usa cargo, área e senioridade)."""
        # O currículo entra por último: se ele contiver "{cargo}" ou similar, não é
        # substituído, porque as trocas anteriores já aconteceram.
        return (
            self._template.replace("{cargo}", vaga.get("cargo") or "não informado")
            .replace("{area}", vaga.get("area_tecnica") or "não informada")
            .replace("{senioridade}", vaga.get("senioridade") or "não informada")
            .replace("{itens_obrigatorios}", _formatar_itens(rubrica.obrigatorios))
            .replace("{itens_desejaveis}", _formatar_itens(rubrica.desejaveis))
            .replace("{curriculo}", texto_curriculo)
        )

    def _interpretar(self, bruto: str, rubrica: RubricaVaga) -> _RespostaJuiz:
        resposta = parse_llm_json(bruto, _RespostaJuiz)
        problemas = conferir_itens(resposta, rubrica)
        if problemas:
            raise ExtractionError("; ".join(problemas))
        return resposta

    def _montar_julgamento(
        self, vaga_id: str, curriculo_id: str, rubrica: RubricaVaga,
        resposta: _RespostaJuiz, tentativas: int,
    ) -> JulgamentoPar:
        # Itens na ordem da rubrica, independentemente da ordem da resposta.
        por_id = {i.item_id: i for i in resposta.itens}
        itens = [por_id[i.item_id] for i in rubrica.itens]
        return JulgamentoPar(
            vaga_id=vaga_id,
            curriculo_id=curriculo_id,
            nota=resposta.nota,
            justificativa=resposta.justificativa,
            itens=itens,
            nota_sugerida_checklist=nota_sugerida(rubrica, {i.item_id: i.atendimento for i in itens}),
            modelo=self.modelo,
            prompt_sha256=self.prompt_sha256,
            tentativas=tentativas,
            data=datetime.now().isoformat(timespec="seconds"),
        )

    def julgar(
        self, vaga: dict, rubrica: RubricaVaga, curriculo_id: str, texto_curriculo: str
    ) -> JulgamentoPar:
        prompt = self.montar_prompt(vaga, rubrica, texto_curriculo)
        erros = []
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                resposta = self._interpretar(self._llm.call(prompt), rubrica)
                return self._montar_julgamento(vaga["vaga_id"], curriculo_id, rubrica, resposta, tentativa)
            except (ExtractionError, ValidationError) as e:
                dbg(_ETAPA, f"{vaga['vaga_id']}/{curriculo_id} tentativa {tentativa}: {e}")
                erros.append(str(e)[:300])
        raise JuizError(f"{vaga['vaga_id']}/{curriculo_id}: {erros}")

    async def ajulgar(
        self, vaga: dict, rubrica: RubricaVaga, curriculo_id: str, texto_curriculo: str
    ) -> JulgamentoPar:
        prompt = self.montar_prompt(vaga, rubrica, texto_curriculo)
        erros = []
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                resposta = self._interpretar(await self._llm.acall(prompt), rubrica)
                return self._montar_julgamento(vaga["vaga_id"], curriculo_id, rubrica, resposta, tentativa)
            except (ExtractionError, ValidationError) as e:
                dbg(_ETAPA, f"{vaga['vaga_id']}/{curriculo_id} tentativa {tentativa}: {e}")
                erros.append(str(e)[:300])
        raise JuizError(f"{vaga['vaga_id']}/{curriculo_id}: {erros}")


# ----------------------------------------------------------------------
# Persistência: um julgamento por linha (JSONL), gravado a cada par.
#
# Fica num arquivo próprio, separado de `anotacoes_calibracao.csv` e
# `checklist_rubrica.csv`: esses são lidos pela interface de anotação, e a
# §4.3 proíbe o anotador humano de ver a nota do LLM.


def carregar_julgamentos(caminho: Path | str) -> dict[tuple[str, str], JulgamentoPar]:
    """(vaga_id, curriculo_id) -> julgamento. Linha corrompida (gravação
    interrompida no meio) é ignorada: o par volta a ser julgado."""
    caminho = Path(caminho)
    julgados: dict[tuple[str, str], JulgamentoPar] = {}
    if not caminho.exists():
        return julgados
    with caminho.open(encoding="utf-8") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            try:
                j = JulgamentoPar.model_validate(json.loads(linha))
            except (json.JSONDecodeError, ValidationError):
                continue
            julgados[(j.vaga_id, j.curriculo_id)] = j
    return julgados


def gravar_julgamento(caminho: Path | str, julgamento: JulgamentoPar) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as fh:
        fh.write(julgamento.model_dump_json() + "\n")
        fh.flush()
