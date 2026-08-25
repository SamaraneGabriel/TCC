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

Pesos abaixo são um ponto de partida, não uma calibração — ajustar com os
resultados da avaliação (Precision@K/MRR) quando essa fase existir.
"""
from dataclasses import dataclass

from ..debug import dbg
from ..schemas.vaga import VagaEstruturada
from ..vectorstore.store import SearchResult

_ETAPA = "matching.ranker"

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


def rank(vaga: VagaEstruturada, candidatos: list[SearchResult]) -> list[ResultadoRankeado]:
    """`candidatos` deve vir na ordem retornada por `CurriculoStore.search`
    (melhor fusão vetorial primeiro) — a posição na lista é o sinal usado,
    não `SearchResult.score`."""
    if not candidatos:
        dbg(_ETAPA, "nenhum candidato para ranquear")
        return []

    dbg(_ETAPA, f"ranqueando {len(candidatos)} candidato(s)")
    n = len(candidatos)
    skills_vaga = {s.lower() for s in (vaga.skills_obrigatorias + vaga.skills_desejadas)}

    resultados = []
    for posicao, c in enumerate(candidatos):
        score_norm = 1 - (posicao / (n - 1)) if n > 1 else 1.0

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
