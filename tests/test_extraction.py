"""
Unit and integration tests for the CV extraction layer.

Run all unit tests (no API key needed):
    pytest tests/test_extraction.py

Run including integration test (requires GEMINI_API_KEY in .env):
    pytest tests/test_extraction.py -m integration
"""
import json

import pytest

from src.extraction.base import BaseExtractor
from src.extraction.curriculo_extractor import CurriculoExtractor, ExtractionError
from src.schemas.curriculo import CurriculoEstruturado

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CV_TEXT = """
Maria Oliveira
Email: maria@email.com | Tel: (31) 99999-8888
Belo Horizonte, MG

RESUMO PROFISSIONAL
Desenvolvedora Backend com 4 anos de experiência em APIs e microsserviços.

HABILIDADES TÉCNICAS
Python, FastAPI, PostgreSQL, Docker, Redis, AWS

HABILIDADES COMPORTAMENTAIS
Comunicação, trabalho em equipe

EXPERIÊNCIA PROFISSIONAL
Desenvolvedora Backend | TechCorp | Jan 2022 – Jan 2024 (24 meses)
- Desenvolvimento de APIs REST com FastAPI e PostgreSQL
- Deploy em AWS ECS com Docker

FORMAÇÃO
Bacharelado em Ciência da Computação | UFMG | Concluído em 2020

CERTIFICAÇÕES
AWS Cloud Practitioner | Amazon | 2023

IDIOMAS
Português (nativo), Inglês (avançado)

Disponibilidade: imediata | Modelo de trabalho: remoto
"""

SAMPLE_JSON = {
    "nome": "Maria Oliveira",
    "email": "maria@email.com",
    "telefone": "(31) 99999-8888",
    "localidade": "Belo Horizonte, MG",
    "resumo_profissional": "Desenvolvedora Backend com 4 anos de experiência em APIs e microsserviços.",
    "senioridade_estimada": "pleno",
    "anos_experiencia_estimados": 4.0,
    "area_principal": "Engenharia de Software",
    "skills_tecnicas": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "AWS"],
    "skills_comportamentais": ["comunicação", "trabalho em equipe"],
    "idiomas": ["Português (nativo)", "Inglês (avançado)"],
    "experiencias": [
        {
            "cargo": "Desenvolvedora Backend",
            "empresa": "TechCorp",
            "duracao_meses": 24,
            "descricao": "Desenvolvimento de APIs REST com FastAPI e PostgreSQL.",
            "skills_utilizadas": ["Python", "FastAPI", "PostgreSQL"],
        }
    ],
    "formacoes": [
        {
            "grau": "graduação",
            "curso": "Ciência da Computação",
            "instituicao": "UFMG",
            "ano_conclusao": 2020,
            "em_andamento": False,
        }
    ],
    "certificacoes": [{"nome": "AWS Cloud Practitioner", "instituicao": "Amazon", "ano": 2023}],
    "pretensao_salarial": None,
    "disponibilidade": "imediata",
    "modelo_trabalho_preferido": "remoto",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockExtractor(BaseExtractor):
    """LLM stub that returns a fixed JSON string."""

    def __init__(self, response: dict | str):
        self._response = response if isinstance(response, str) else json.dumps(response)

    def call(self, prompt: str) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Unit tests — no network calls
# ---------------------------------------------------------------------------


def test_extract_returns_correct_type():
    result = CurriculoExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_CV_TEXT)
    assert isinstance(result, CurriculoEstruturado)


def test_extract_basic_fields():
    result = CurriculoExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_CV_TEXT)
    assert result.nome == "Maria Oliveira"
    assert result.email == "maria@email.com"
    assert result.localidade == "Belo Horizonte, MG"
    assert result.senioridade_estimada == "pleno"
    assert result.modelo_trabalho_preferido == "remoto"


def test_extract_list_fields():
    result = CurriculoExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_CV_TEXT)
    assert "Python" in result.skills_tecnicas
    assert "AWS" in result.skills_tecnicas
    assert "Português (nativo)" in result.idiomas


def test_extract_nested_experiencia():
    result = CurriculoExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_CV_TEXT)
    assert len(result.experiencias) == 1
    exp = result.experiencias[0]
    assert exp.cargo == "Desenvolvedora Backend"
    assert exp.empresa == "TechCorp"
    assert exp.duracao_meses == 24
    assert "FastAPI" in exp.skills_utilizadas


def test_extract_nested_formacao():
    result = CurriculoExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_CV_TEXT)
    assert len(result.formacoes) == 1
    f = result.formacoes[0]
    assert f.grau == "graduação"
    assert f.instituicao == "UFMG"
    assert f.ano_conclusao == 2020
    assert f.em_andamento is False


def test_extract_handles_markdown_fences():
    wrapped = f"```json\n{json.dumps(SAMPLE_JSON)}\n```"
    result = CurriculoExtractor(MockExtractor(wrapped)).extract(SAMPLE_CV_TEXT)
    assert result.nome == "Maria Oliveira"


def test_extract_handles_plain_markdown_fence():
    wrapped = f"```\n{json.dumps(SAMPLE_JSON)}\n```"
    result = CurriculoExtractor(MockExtractor(wrapped)).extract(SAMPLE_CV_TEXT)
    assert result.nome == "Maria Oliveira"


def test_extract_raises_on_invalid_json():
    with pytest.raises(ExtractionError, match="invalid JSON"):
        CurriculoExtractor(MockExtractor("not json at all")).extract(SAMPLE_CV_TEXT)


def test_extract_allows_missing_nome():
    """Currículos anonimizados/redigidos não têm nome explícito no texto — o schema
    aceita isso (nome vira None) em vez de forçar o LLM a inventar um nome pra passar
    na validação."""
    without_nome = {k: v for k, v in SAMPLE_JSON.items() if k != "nome"}
    result = CurriculoExtractor(MockExtractor(without_nome)).extract(SAMPLE_CV_TEXT)
    assert result.nome is None


def test_extract_minimal_cv_all_optionals_none():
    minimal = {"nome": "João Silva"}
    result = CurriculoExtractor(MockExtractor(minimal)).extract("João Silva")
    assert result.nome == "João Silva"
    assert result.email is None
    assert result.skills_tecnicas == []
    assert result.experiencias == []
    assert result.formacoes == []


def test_factory_raises_on_unknown_provider():
    from src.config import Settings
    from src.extraction.factory import create_extractor

    s = Settings(llm_provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_extractor(s)


# ---------------------------------------------------------------------------
# Integration test — calls real Gemini API
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gemini_extracts_sample_cv():
    """Via factory, para exercitar o mesmo caminho da aplicação (Vertex AI quando
    `GCP_PROJECT` está preenchido) — ver a nota em test_vaga_extraction.py."""
    from src.config import settings
    from src.extraction.factory import create_extractor

    if not settings.gcp_project and (
        not settings.gemini_api_key or settings.gemini_api_key == "No Key Provided"
    ):
        pytest.skip("nem GCP_PROJECT nem GEMINI_API_KEY configurados no .env")

    llm = create_extractor(settings)
    result = CurriculoExtractor(llm).extract(SAMPLE_CV_TEXT)

    assert isinstance(result, CurriculoEstruturado)
    assert result.nome  # must have extracted a name
    assert len(result.skills_tecnicas) > 0
    assert len(result.experiencias) > 0
    assert result.modelo_trabalho_preferido in (None, "remoto", "híbrido", "presencial")
