"""Rubrica de qualificação por vaga (seção 3 do plano) — schema, geração via LLM
e persistência no formato da seção 7.3.

É a peça central da v2 da spec: todo julgamento de relevância, humano ou LLM, é
feito contra a rubrica da vaga, não por impressão livre. Consequências de
projeto que valem registrar:

- **`item_id` é atribuído aqui, não pelo LLM.** Ele é a chave que liga a rubrica
  ao checklist de atendimento (§7.4) e precisa ser estável entre execuções e
  legível num CSV. O LLM só devolve descrição/tipo/categoria.
- **A rubrica é gerada uma vez e reutilizada** (§3.2 passo 4). Por isso
  `carregar_rubricas` existe e o script de geração pula vaga já gerada: regerar
  invalidaria anotações feitas contra a versão anterior.
- **Nada aqui lê vaga de disco nem conhece Qdrant.** Recebe `VagaPool` pronta,
  como os outros módulos deste pacote.
- **O vocabulário O*NET entra no prompt como apoio, nunca como fonte** (§3.1):
  sem vocabulário disponível a rubrica ainda é gerada, só sem essa ajuda.
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from ...extraction.base import BaseExtractor
from ...extraction.parsing import parse_llm_json
from .onet import VocabularioONet
from .selecao import VagaPool

TIPOS = ("obrigatoria", "desejavel")
CATEGORIAS = (
    "conhecimento",
    "ferramenta_tecnologia",
    "habilidade_tecnica",
    "habilidade_basica",
    "experiencia_relacionada",
    "formacao",
    "senioridade",
    "outro",
)
ORIGENS = ("llm", "humano_revisado")

CAMPOS_RUBRICA = ["vaga_id", "item_id", "descricao", "tipo", "categoria", "origem", "data_geracao"]

# Faixa da regra 6 do prompt. Fora dela não é erro fatal (uma vaga pobre pode
# render 2 itens), vira aviso em `problemas()`.
MIN_ITENS, MAX_ITENS = 3, 25

# Quantas tecnologias do O*NET entram no prompt. O perfil completo de Software
# Developers tem 430 termos; mesmo o corte "hot" passa de 150. 120 mantém o
# vocabulário útil sem afogar o texto da vaga, que é a fonte real do requisito.
LIMITE_VOCABULARIO = 120
LIMITE_CATEGORIAS_ONET = 40

_PROMPT = (Path(__file__).parent / "prompts" / "rubrica_prompt.txt").read_text(encoding="utf-8")

Tipo = Literal["obrigatoria", "desejavel"]
Categoria = Literal[
    "conhecimento",
    "ferramenta_tecnologia",
    "habilidade_tecnica",
    "habilidade_basica",
    "experiencia_relacionada",
    "formacao",
    "senioridade",
    "outro",
]
Origem = Literal["llm", "humano_revisado"]


class _ItemBruto(BaseModel):
    """O que o LLM devolve por item — sem `item_id`, que é do código."""

    descricao: str
    tipo: Tipo
    categoria: Categoria

    @field_validator("descricao")
    @classmethod
    def _nao_vazia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("descrição vazia")
        return v


class _RespostaRubrica(BaseModel):
    itens: list[_ItemBruto]


class ItemQualificacao(BaseModel):
    """Um item da rubrica, já com identidade estável (§7.3)."""

    item_id: str
    descricao: str
    tipo: Tipo
    categoria: Categoria

    @property
    def obrigatorio(self) -> bool:
        return self.tipo == "obrigatoria"


class RubricaVaga(BaseModel):
    """A rubrica completa de uma vaga.

    `origem` distingue o rascunho do LLM (§3.2 passo 1) da versão revisada por
    humano, obrigatória nas vagas de calibração (§3.2 passo 2) — é o campo que
    o script de conferência usa para saber se a revisão já aconteceu.
    """

    vaga_id: str
    itens: list[ItemQualificacao]
    origem: Origem = "llm"
    data_geracao: str
    modelo: Optional[str] = None
    ocupacoes_onet: list[str] = []

    # ------------------------------------------------------------------

    @property
    def obrigatorios(self) -> list[ItemQualificacao]:
        return [i for i in self.itens if i.obrigatorio]

    @property
    def desejaveis(self) -> list[ItemQualificacao]:
        return [i for i in self.itens if not i.obrigatorio]

    def item(self, item_id: str) -> ItemQualificacao | None:
        return next((i for i in self.itens if i.item_id == item_id), None)

    def problemas(self) -> list[str]:
        """O que impede usar esta rubrica numa anotação. Lista vazia = pode usar."""
        p: list[str] = []
        if not self.itens:
            p.append(f"{self.vaga_id}: rubrica sem itens")
        if not self.obrigatorios:
            # Sem obrigatórios, "atende todos os obrigatórios" é vacuamente
            # verdadeiro e as notas 2 e 3 da §6 deixam de discriminar.
            p.append(f"{self.vaga_id}: nenhum item obrigatório — a escala da §6 fica indefinida")
        ids = [i.item_id for i in self.itens]
        if len(ids) != len(set(ids)):
            p.append(f"{self.vaga_id}: item_id repetido")
        descricoes = [i.descricao.strip().lower() for i in self.itens]
        if len(descricoes) != len(set(descricoes)):
            p.append(f"{self.vaga_id}: descrição repetida entre itens")
        return p

    def avisos(self) -> list[str]:
        """Não impedem o uso, mas merecem olhada na revisão humana."""
        a: list[str] = []
        if len(self.itens) < MIN_ITENS:
            a.append(f"{self.vaga_id}: só {len(self.itens)} itens (mínimo sugerido {MIN_ITENS})")
        if len(self.itens) > MAX_ITENS:
            a.append(f"{self.vaga_id}: {len(self.itens)} itens (máximo sugerido {MAX_ITENS})")
        if len(self.itens) and not self.desejaveis:
            a.append(f"{self.vaga_id}: nenhum item desejável — a nota 3 da §6 fica inalcançável")
        basicas_obrigatorias = [
            i.item_id for i in self.obrigatorios if i.categoria == "habilidade_basica"
        ]
        if basicas_obrigatorias:
            # §3.1: habilidade_basica é desejável por padrão, salvo exigência explícita.
            a.append(
                f"{self.vaga_id}: habilidade_basica marcada como obrigatória "
                f"({', '.join(basicas_obrigatorias)}) — confirmar que a vaga exige explicitamente"
            )
        return a

    def resumo_categorias(self) -> dict[str, int]:
        return {c: sum(1 for i in self.itens if i.categoria == c) for c in CATEGORIAS if any(
            i.categoria == c for i in self.itens
        )}


def gerar_item_id(vaga_id: str, indice: int) -> str:
    """`item_id` legível e estável: `<vaga_id>#03`. Determinístico dada a ordem
    dos itens, que é a ordem em que o LLM os devolveu e que a persistência
    preserva."""
    return f"{vaga_id}#{indice:02d}"


class GeradorRubrica:
    """Gera o rascunho da rubrica de uma vaga (§3.2 passo 1).

    Recebe o mesmo `BaseExtractor` que o resto do projeto usa (Gemini via Vertex,
    OpenAI ou Ollama, conforme o factory) — este módulo não escolhe provider.
    """

    def __init__(
        self,
        llm: BaseExtractor,
        vocabulario: VocabularioONet | None = None,
        limite_vocabulario: int = LIMITE_VOCABULARIO,
    ) -> None:
        self._llm = llm
        self._vocab = vocabulario
        self._limite = limite_vocabulario

    # ------------------------------------------------------------------

    def montar_prompt(self, vaga: VagaPool) -> str:
        area = vaga.area_tecnica or "não informada"
        if self._vocab:
            socs = self._vocab.ocupacoes(area)
            ocupacoes = "; ".join(f"{soc} {self._vocab.titulo(soc)}" for soc in socs)
            tecnologias = ", ".join(self._vocab.termos(area, limite=self._limite))
            categorias = ", ".join(self._vocab.categorias(area)[:LIMITE_CATEGORIAS_ONET])
        else:
            ocupacoes = tecnologias = categorias = (
                "(vocabulário O*NET indisponível nesta execução — classifique pelo texto da vaga)"
            )

        return (
            _PROMPT.replace("{ocupacoes}", ocupacoes)
            .replace("{tecnologias}", tecnologias)
            .replace("{categorias_onet}", categorias)
            .replace("{cargo}", vaga.cargo)
            .replace("{area}", area)
            .replace("{senioridade}", vaga.senioridade or "não informada")
            .replace("{texto}", vaga.texto)
        )

    def _montar_rubrica(self, vaga: VagaPool, resposta: _RespostaRubrica) -> RubricaVaga:
        itens = [
            ItemQualificacao(
                item_id=gerar_item_id(vaga.vaga_id, n),
                descricao=bruto.descricao,
                tipo=bruto.tipo,
                categoria=bruto.categoria,
            )
            for n, bruto in enumerate(resposta.itens, start=1)
        ]
        ocupacoes = list(self._vocab.ocupacoes(vaga.area_tecnica or "")) if self._vocab else []
        return RubricaVaga(
            vaga_id=vaga.vaga_id,
            itens=itens,
            origem="llm",
            data_geracao=datetime.now().isoformat(timespec="seconds"),
            modelo=getattr(self._llm, "model", None),
            ocupacoes_onet=ocupacoes,
        )

    def gerar(self, vaga: VagaPool) -> RubricaVaga:
        resposta = parse_llm_json(self._llm.call(self.montar_prompt(vaga)), _RespostaRubrica)
        return self._montar_rubrica(vaga, resposta)

    async def agerar(self, vaga: VagaPool) -> RubricaVaga:
        bruto = await self._llm.acall(self.montar_prompt(vaga))
        return self._montar_rubrica(vaga, parse_llm_json(bruto, _RespostaRubrica))


# ----------------------------------------------------------------------
# Persistência (§7.3): uma linha por item de qualificação


def linhas_csv(rubricas: list[RubricaVaga]) -> list[dict]:
    return [
        {
            "vaga_id": r.vaga_id,
            "item_id": i.item_id,
            "descricao": i.descricao,
            "tipo": i.tipo,
            "categoria": i.categoria,
            "origem": r.origem,
            "data_geracao": r.data_geracao,
        }
        for r in rubricas
        for i in r.itens
    ]


def remover_itens(
    rubricas: dict[str, RubricaVaga], item_ids: set[str]
) -> dict[str, RubricaVaga]:
    """Devolve cópia das rubricas sem os itens em `item_ids` (desvio D9).

    Os `item_id` restantes **não são renumerados**: buraco na sequência é
    inofensivo, e renumerar faria um id passar a apontar para outro item sem que
    nada acusasse. `origem` e `data_geracao` ficam como estavam — a remoção é
    aplicação de uma regra registrada, não revisão humana da rubrica inteira.

    Falha se algum id não existir (decisão sobre item que não está na rubrica é
    erro de digitação ou de arquivo) ou se a remoção deixar uma rubrica inválida.
    """
    existentes = {i.item_id for r in rubricas.values() for i in r.itens}
    desconhecidos = sorted(item_ids - existentes)
    if desconhecidos:
        raise ValueError(f"item_id inexistente nas rubricas: {', '.join(desconhecidos)}")

    novas = {
        vaga_id: r.model_copy(update={"itens": [i for i in r.itens if i.item_id not in item_ids]})
        for vaga_id, r in rubricas.items()
    }
    problemas = [p for r in novas.values() for p in r.problemas()]
    if problemas:
        raise ValueError("remoção deixaria rubrica inválida: " + "; ".join(problemas))
    return novas


def salvar_rubricas(rubricas: list[RubricaVaga], caminho: Path | str) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_RUBRICA)
        w.writeheader()
        w.writerows(linhas_csv(rubricas))
    return caminho


def carregar_rubricas(caminho: Path | str) -> dict[str, RubricaVaga]:
    """Lê o CSV de volta, indexado por `vaga_id`.

    A ordem das linhas define a ordem dos itens — é ela que `item_id` reflete, e
    é por isso que o CSV não deve ser reordenado à mão. Arquivo inexistente
    devolve dict vazio: é o caso normal da primeira execução.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return {}

    por_vaga: dict[str, list[ItemQualificacao]] = {}
    meta: dict[str, tuple[str, str]] = {}
    with caminho.open(encoding="utf-8", newline="") as fh:
        for linha in csv.DictReader(fh):
            vaga_id = linha["vaga_id"]
            por_vaga.setdefault(vaga_id, []).append(
                ItemQualificacao(
                    item_id=linha["item_id"],
                    descricao=linha["descricao"],
                    tipo=linha["tipo"],
                    categoria=linha["categoria"],
                )
            )
            meta[vaga_id] = (linha["origem"], linha["data_geracao"])

    return {
        vaga_id: RubricaVaga(
            vaga_id=vaga_id,
            itens=itens,
            origem=meta[vaga_id][0],
            data_geracao=meta[vaga_id][1],
        )
        for vaga_id, itens in por_vaga.items()
    }


def conferir_rubricas(
    rubricas: dict[str, RubricaVaga], vagas: list[VagaPool], ids_calibracao: set[str]
) -> dict:
    """Checagem de prontidão antes da amostragem/anotação.

    Bloqueia se falta rubrica para alguma vaga do estudo, se alguma rubrica é
    inválida (§6 indefinida), ou se uma vaga de calibração ainda está com a
    rubrica do LLM sem revisão humana — que a §3.2 passo 2 torna obrigatória.
    """
    bloqueios: list[str] = []
    avisos: list[str] = []

    faltando = [v.vaga_id for v in vagas if v.vaga_id not in rubricas]
    if faltando:
        bloqueios.append(f"{len(faltando)} vaga(s) sem rubrica: {', '.join(faltando[:5])}"
                         + (" ..." if len(faltando) > 5 else ""))

    for vaga_id in ids_calibracao:
        r = rubricas.get(vaga_id)
        if r and r.origem != "humano_revisado":
            bloqueios.append(
                f"{vaga_id} é vaga de calibração e a rubrica ainda está como '{r.origem}' — "
                "a §3.2 passo 2 exige revisão humana antes do uso"
            )

    for r in rubricas.values():
        bloqueios.extend(r.problemas())
        avisos.extend(r.avisos())

    itens = [len(r.itens) for r in rubricas.values()]
    return {
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "avisos": avisos,
        "rubricas": len(rubricas),
        "revisadas": sum(1 for r in rubricas.values() if r.origem == "humano_revisado"),
        "itens_total": sum(itens),
        "itens_por_vaga_min": min(itens) if itens else 0,
        "itens_por_vaga_max": max(itens) if itens else 0,
        "itens_por_vaga_media": round(sum(itens) / len(itens), 1) if itens else 0.0,
    }
