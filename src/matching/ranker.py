"""
Ranker — score composto simples (decisão #5 de context/09_DECISOES_PENDENTES.md:
"testar score composto simples primeiro").

O valor de `SearchResult.score` (fusão RRF/DBSF) não é comparável em escala
absoluta — para RRF em especial, a diferença bruta entre dois candidatos é
quase sempre minúscula (ex.: 1/61 vs 1/62), então normalizar por min-max sobre
esse valor bruto amplifica ruído de desempate do Qdrant num salto artificial de
0 a 1. O que a fusão garante de fato é a ORDEM dos candidatos — por isso usamos
a posição no resultado (já ordenado por `CurriculoStore.search`) como sinal
vetorial normalizado, não o valor de `score` em si.

Decaimento independente do tamanho da lista (corrigido em 2026-09-22, desvio D7
do plano de avaliação — ver `context/11_DESVIOS_AVALIACAO.md`)
-------------------------------------------------------------------------------
A normalização anterior era `1 - posicao/(n-1)`, com `n` = quantos candidatos
foram passados. Isso fazia o score de um candidato depender de **quantos outros
foram pedidos na busca**: com n=100 a distância entre posições vizinhas é 1/99,
com n=25 é 1/24, e o peso relativo de skills/senioridade/localidade muda junto.
Medido em 3 vagas reais, o top-25 obtido buscando 100 candidatos coincidia em
apenas 2 a 7 das 25 posições com o obtido buscando 25 — o mesmo sistema, a mesma
vaga e o mesmo banco davam rankings diferentes conforme a profundidade da busca.

Agora o sinal usa o decaimento recíproco do RRF, `(K0+1)/(K0+1+posicao)`, que
depende só da posição absoluta: vale 1,0 na primeira posição, 0,718 na 25ª e 0,381 na 100ª, de
forma idêntica esteja a lista com 25 ou 10.000 itens. Buscar mais fundo passa a
só acrescentar candidatos ao fim da lista, nunca a reordenar o topo — que é o
comportamento que a seção 5.2 do plano de avaliação pressupõe ao mandar chamar o
sistema com `limit=TOP_POOL` em vez do K de produção.

K0=60 é a constante usual do RRF (Cormack et al., 2009), a mesma família de
decaimento que o `CurriculoStore.search` já usa para fundir os dois vetores.

Pesos abaixo são um ponto de partida, não uma calibração — ajustar com os
resultados da avaliação (Precision@K/MRR) quando essa fase existir.
"""
from dataclasses import dataclass

from ..debug import dbg
from ..schemas.vaga import VagaEstruturada
from ..vectorstore.store import SearchResult

_ETAPA = "matching.ranker"

# Constante do decaimento recíproco (RRF). Ver a nota no topo do módulo.
K0_RRF = 60

PESO_VETORIAL = 0.6
PESO_SKILLS = 0.3
PESO_SENIORIDADE = 0.05
PESO_LOCALIDADE = 0.05


@dataclass
class ResultadoRankeado:
    rank: int
    score_final: float
    score_vetorial_normalizado: float
    skills_match: list[str]
    skills_total_vaga: int
    senioridade_match: bool
    localidade_match: bool
    candidato: SearchResult


def score_posicional(posicao: int) -> float:
    """Sinal vetorial a partir da posição (0-based) no resultado da busca.

    `(K0+1)/(K0+1+posicao)`: 1,0 na primeira posição, decaimento suave e
    **independente de quantos candidatos a lista tem** — ver a nota no topo do
    módulo sobre por que isso importa."""
    return (K0_RRF + 1) / (K0_RRF + 1 + posicao)


def rank(vaga: VagaEstruturada, candidatos: list[SearchResult]) -> list[ResultadoRankeado]:
    """`candidatos` deve vir na ordem retornada por `CurriculoStore.search`
    (melhor fusão vetorial primeiro) — a posição na lista é o sinal usado,
    não `SearchResult.score`.

    Pedir mais candidatos não altera a ordem relativa dos primeiros: o sinal
    posicional não depende do tamanho da lista."""
    if not candidatos:
        dbg(_ETAPA, "nenhum candidato para ranquear")
        return []

    dbg(_ETAPA, f"ranqueando {len(candidatos)} candidato(s)")
    skills_vaga = {s.lower() for s in (vaga.skills_obrigatorias + vaga.skills_desejadas)}

    resultados = []
    for posicao, c in enumerate(candidatos):
        score_norm = score_posicional(posicao)

        skills_candidato = {s.lower() for s in c.skills_tecnicas}
        match = sorted(skills_candidato & skills_vaga)
        skills_ratio = len(match) / len(skills_vaga) if skills_vaga else 0.0

        senioridade_match = bool(vaga.senioridade) and c.senioridade_estimada == vaga.senioridade
        localidade_match = bool(vaga.localidade) and c.localidade == vaga.localidade

        score_final = (
            PESO_VETORIAL * score_norm
            + PESO_SKILLS * skills_ratio
            + PESO_SENIORIDADE * senioridade_match
            + PESO_LOCALIDADE * localidade_match
        )

        resultados.append(
            ResultadoRankeado(
                rank=0,
                score_final=round(score_final, 4),
                score_vetorial_normalizado=round(score_norm, 4),
                skills_match=match,
                skills_total_vaga=len(skills_vaga),
                senioridade_match=senioridade_match,
                localidade_match=localidade_match,
                candidato=c,
            )
        )

    resultados.sort(key=lambda r: r.score_final, reverse=True)
    for i, r in enumerate(resultados, start=1):
        r.rank = i

    top = resultados[0]
    dbg(_ETAPA, f"#1: {top.candidato.nome} (score_final={top.score_final}, skills={len(top.skills_match)}/{top.skills_total_vaga})")
    return resultados
