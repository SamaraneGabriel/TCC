"""
Testes do LLM-as-judge com o cliente Gemini mockado (sem chamar rede).
"""
import json
from unittest.mock import patch

from src.evaluation.judge import GeminiJudge
from src.schemas.vaga import VagaEstruturada
from src.vectorstore.store import SearchResult


class FakeUsage:
    def __init__(self, prompt=100, candidates=50, total=150):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, text: str, usage: FakeUsage | None = None):
        self.text = text
        self.usage_metadata = usage if usage is not None else FakeUsage()


def make_candidato(id_: str, nome: str, skills: list[str] | None = None) -> SearchResult:
    return SearchResult(
        id=id_,
        score=0.0,
        nome=nome,
        email=None,
        localidade="Belo Horizonte, MG",
        senioridade_estimada="pleno",
        anos_experiencia_estimados=None,
        area_principal="Engenharia de Software",
        skills_tecnicas=skills or [],
        modelo_trabalho_preferido="remoto",
    )


VAGA = VagaEstruturada(titulo="Dev Backend Python", senioridade="pleno", skills_obrigatorias=["Python"])


def test_julgar_maps_labels_back_to_real_ids():
    candidatos = [make_candidato("id-1", "Ana"), make_candidato("id-2", "Bruno")]
    resposta_fake = {
        "avaliacoes": [
            {"candidato": "candidato_1", "grau": 2, "justificativa": "bom match"},
            {"candidato": "candidato_2", "grau": 0, "justificativa": "sem match"},
        ],
        "ordem_sugerida": ["candidato_1", "candidato_2"],
    }

    judge = GeminiJudge(api_key="fake-key")
    with patch("random.shuffle", lambda x: None):  # mantem ordem original -> previsibilidade do teste
        with patch.object(
            judge._client.models, "generate_content", return_value=FakeResponse(json.dumps(resposta_fake))
        ):
            julgamento = judge.julgar(VAGA, candidatos)

    assert julgamento.graus == {"id-1": 2, "id-2": 0}
    assert julgamento.justificativas["id-1"] == "bom match"
    assert julgamento.ordem_sugerida == ["id-1", "id-2"]


def test_julgar_captures_usage_metadata():
    candidatos = [make_candidato("id-1", "Ana")]
    resposta_fake = {
        "avaliacoes": [{"candidato": "candidato_1", "grau": 1, "justificativa": "x"}],
        "ordem_sugerida": ["candidato_1"],
    }

    judge = GeminiJudge(api_key="fake-key", model="modelo-teste")
    with patch.object(
        judge._client.models,
        "generate_content",
        return_value=FakeResponse(json.dumps(resposta_fake), FakeUsage(10, 5, 15)),
    ):
        julgamento = judge.julgar(VAGA, candidatos)

    assert julgamento.usage["modelo"] == "modelo-teste"
    assert julgamento.usage["prompt_tokens"] == 10
    assert julgamento.usage["resposta_tokens"] == 5
    assert julgamento.usage["total_tokens"] == 15
    assert julgamento.usage["latencia_s"] >= 0


def test_julgar_ignores_unknown_labels():
    candidatos = [make_candidato("id-1", "Ana")]
    resposta_fake = {
        "avaliacoes": [
            {"candidato": "candidato_1", "grau": 2, "justificativa": "ok"},
            {"candidato": "candidato_99", "grau": 0, "justificativa": "rotulo que nao existe"},
        ],
        "ordem_sugerida": ["candidato_1"],
    }

    judge = GeminiJudge(api_key="fake-key")
    with patch.object(
        judge._client.models, "generate_content", return_value=FakeResponse(json.dumps(resposta_fake))
    ):
        julgamento = judge.julgar(VAGA, candidatos)

    assert julgamento.graus == {"id-1": 2}
