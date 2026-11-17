"""
Tests for the matching module (retriever + ranker), using Qdrant in-memory
(no Docker) and a deterministic FakeEmbedder (no network).
"""
import pytest
from qdrant_client import QdrantClient

from src.embedding.base import BaseEmbedder
from src.matching.ranker import rank, score_posicional
from src.matching.retriever import buscar_candidatos
from src.schemas.curriculo import CurriculoEstruturado
from src.schemas.vaga import VagaEstruturada
from src.vectorstore.store import CurriculoStore, SearchResult

VECTOR_SIZE = 8


def make_store() -> CurriculoStore:
    store = CurriculoStore.__new__(CurriculoStore)
    store.client = QdrantClient(":memory:")
    store.create_collection(vector_size=VECTOR_SIZE)
    return store


def make_curriculo(nome: str, senioridade: str, skills: list[str], localidade: str = "Belo Horizonte, MG") -> CurriculoEstruturado:
    return CurriculoEstruturado(
        nome=nome,
        localidade=localidade,
        senioridade_estimada=senioridade,
        skills_tecnicas=skills,
        modelo_trabalho_preferido="remoto",
    )


def vec(seed: float, size: int = VECTOR_SIZE) -> list[float]:
    raw = [(seed + i * 0.1) for i in range(size)]
    norm = sum(x**2 for x in raw) ** 0.5
    return [x / norm for x in raw]


class FakeEmbedder(BaseEmbedder):
    """Embedder determinístico: cada texto mapeia para um vetor fixo via seed por tamanho."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [vec(len(t) % 5 * 0.1) for t in texts]


VAGA = VagaEstruturada(
    titulo="Dev Backend Python",
    senioridade="pleno",
    localidade="Belo Horizonte, MG",
    skills_obrigatorias=["Python", "FastAPI"],
    skills_desejadas=["Docker"],
)


# ── retriever ────────────────────────────────────────────────────────────────


def test_buscar_candidatos_returns_results():
    store = make_store()
    store.upsert(make_curriculo("Dev A", "pleno", ["Python", "FastAPI"]), vec(0.1), vec(0.1))
    store.upsert(make_curriculo("Dev B", "junior", ["Java"]), vec(0.5), vec(0.5))

    resultados = buscar_candidatos(VAGA, FakeEmbedder(), store, top_k=5)
    assert len(resultados) == 2


def test_buscar_candidatos_applies_filters():
    store = make_store()
    store.upsert(make_curriculo("Pleno BH", "pleno", ["Python"]), vec(0.1), vec(0.1))
    store.upsert(make_curriculo("Senior BH", "senior", ["Python"]), vec(0.1), vec(0.1))

    resultados = buscar_candidatos(VAGA, FakeEmbedder(), store, top_k=5, senioridade="pleno")
    assert len(resultados) == 1
    assert resultados[0].senioridade_estimada == "pleno"


# ── ranker ───────────────────────────────────────────────────────────────────
#
# rank() usa a POSIÇÃO na lista de candidatos como sinal vetorial normalizado
# (ver comentário em src/matching/ranker.py sobre por que o valor bruto de
# SearchResult.score — RRF — não é usado diretamente). Por isso os testes
# constroem SearchResult diretamente, na ordem de fusão desejada, em vez de
# depender de scores de fusão reais do Qdrant (cuja diferença bruta entre
# candidatos próximos é minúscula e sujeita a desempate arbitrário).


def make_result(
    nome: str,
    senioridade: str = "pleno",
    localidade: str = "Belo Horizonte, MG",
    skills: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        id=nome,
        score=0.0,  # nao usado por rank() — ver comentario acima
        nome=nome,
        email=None,
        localidade=localidade,
        senioridade_estimada=senioridade,
        anos_experiencia_estimados=None,
        area_principal=None,
        skills_tecnicas=skills or [],
        modelo_trabalho_preferido="remoto",
    )


def test_rank_empty_returns_empty():
    assert rank(VAGA, []) == []


def test_rank_assigns_sequential_ranks():
    candidatos = [make_result("Dev A"), make_result("Dev B")]
    ranqueados = rank(VAGA, candidatos)
    assert [r.rank for r in ranqueados] == [1, 2]


def test_rank_computes_skills_overlap_correctly():
    candidatos = [
        make_result("Match Skills", skills=["Python", "FastAPI", "Docker"]),
        make_result("No Match", skills=["Java", "Spring"]),
    ]
    ranqueados = {r.candidato.nome: r for r in rank(VAGA, candidatos)}
    assert ranqueados["Match Skills"].skills_match == ["docker", "fastapi", "python"]
    assert ranqueados["No Match"].skills_match == []
    assert ranqueados["Match Skills"].skills_total_vaga == 3  # obrigatorias + desejadas


def test_rank_skills_overlap_can_overcome_modest_vector_rank_gap():
    # posicoes 0..3 -> score_norm 1.0, 0.984, 0.968, 0.953 (decaimento RRF, ver rank()).
    # o 2o colocado no vetor, com skills completas, deve superar o 1o colocado sem skills.
    candidatos = [
        make_result("Vetor top, sem skills", skills=[]),
        make_result("Vetor 2o, skills completas", skills=["Python", "FastAPI", "Docker"]),
        make_result("Vetor 3o, sem skills", skills=[]),
        make_result("Vetor ultimo, sem skills", skills=[]),
    ]
    ranqueados = rank(VAGA, candidatos)
    assert ranqueados[0].candidato.nome == "Vetor 2o, skills completas"


def test_rank_senioridade_and_localidade_match_flags():
    candidatos = [
        make_result("Match Tudo", senioridade="pleno", localidade="Belo Horizonte, MG"),
        make_result("Outro Local", senioridade="pleno", localidade="São Paulo, SP"),
    ]
    ranqueados = {r.candidato.nome: r for r in rank(VAGA, candidatos)}
    assert ranqueados["Match Tudo"].senioridade_match is True
    assert ranqueados["Match Tudo"].localidade_match is True
    assert ranqueados["Outro Local"].localidade_match is False


def test_score_posicional_nao_depende_do_tamanho_da_lista():
    """Desvio D7: o sinal vetorial vem da posição absoluta, não da fração
    `posicao/(n-1)`. Sem isso, pedir mais candidatos na busca reordena o topo."""
    assert score_posicional(0) == 1.0
    assert score_posicional(24) == pytest.approx(0.7176, abs=1e-4)   # 61/85
    assert score_posicional(99) == pytest.approx(0.3813, abs=1e-4)   # 61/160
    assert score_posicional(10) > score_posicional(11)  # monotônico


def test_rank_de_prefixo_preserva_a_ordem_relativa():
    """A propriedade que a §5.2 do plano de avaliação pressupõe ao mandar chamar
    o sistema com `limit=TOP_POOL` em vez do K de produção: ranquear os mesmos
    candidatos dentro de uma lista maior não muda a ordem entre eles.

    Antes da correção D7 isto falhava — medido em 3 vagas reais, o top-25 de uma
    busca com limit=100 coincidia em 2 a 7 das 25 posições com o de limit=25."""
    candidatos = [
        make_result("A", skills=["Python"]),
        make_result("B", skills=[]),
        make_result("C", skills=["Python", "FastAPI", "Docker"]),
        make_result("D", skills=["Docker"]),
        make_result("E", skills=["FastAPI"]),
        make_result("F", skills=[]),
    ]
    prefixo = {c.nome for c in candidatos[:3]}
    dentro_do_todo = [r.candidato.nome for r in rank(VAGA, candidatos) if r.candidato.nome in prefixo]
    sozinhos = [r.candidato.nome for r in rank(VAGA, candidatos[:3])]
    assert dentro_do_todo == sozinhos


def test_rank_handles_single_candidate_without_zero_division():
    ranqueados = rank(VAGA, [make_result("Único")])
    assert len(ranqueados) == 1
    assert ranqueados[0].score_vetorial_normalizado == 1.0


def test_rank_score_final_matches_formula():
    r = rank(VAGA, [make_result("Único", senioridade="pleno", localidade="Belo Horizonte, MG", skills=["Python", "FastAPI"])])[0]
    # unico candidato -> score_norm=1.0; skills_vaga={python,fastapi,docker}, match={python,fastapi} -> ratio 2/3
    esperado = round(0.6 * 1.0 + 0.3 * (2 / 3) + 0.05 * 1 + 0.05 * 1, 4)
    assert r.score_final == esperado
