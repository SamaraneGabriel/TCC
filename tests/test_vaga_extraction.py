"""
Unit and integration tests for the vaga extraction layer.

Run all unit tests (no API key needed):
    pytest tests/test_vaga_extraction.py

Run including integration test (requires GEMINI_API_KEY in .env):
    pytest tests/test_vaga_extraction.py -m integration
"""
import json

import pytest

from src.extraction.base import BaseExtractor
from src.extraction.vaga_extractor import ExtractionError, VagaExtractor
from src.schemas.vaga import VagaEstruturada

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_VAGA_TEXT = """
Desenvolvedor(a) Backend Python - Pleno
FinTech ABC | São Paulo, SP | Híbrido

SOBRE A VAGA
Buscamos dev backend para atuar no time de pagamentos.

RESPONSABILIDADES
- Desenvolver e manter APIs REST
- Participar de code reviews
- Contribuir com arquitetura de microsserviços

REQUISITOS OBRIGATÓRIOS
Python, FastAPI, PostgreSQL, Docker

DIFERENCIAIS
Redis, Kafka, AWS, Kubernetes

Experiência mínima: 3 anos
Formação: graduação

Regime: CLT
"""

SAMPLE_JSON = {
    "titulo": "Desenvolvedor(a) Backend Python - Pleno",
    "empresa": "FinTech ABC",
    "localidade": "São Paulo, SP",
    "modelo_trabalho": "híbrido",
    "area": "Engenharia de Software",
    "senioridade": "pleno",
    "descricao": "Buscamos dev backend para atuar no time de pagamentos.",
    "responsabilidades": [
        "Desenvolver e manter APIs REST",
        "Participar de code reviews",
        "Contribuir com arquitetura de microsserviços",
    ],
    "skills_obrigatorias": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "skills_desejadas": ["Redis", "Kafka", "AWS", "Kubernetes"],
    "requisitos": [],
    "experiencia_minima_anos": 3,
    "formacao_minima": "graduação",
    "faixa_salarial": None,
    "beneficios": [],
    "regime_contratacao": "CLT",
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
    result = VagaExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_VAGA_TEXT)
    assert isinstance(result, VagaEstruturada)


def test_extract_basic_fields():
    result = VagaExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_VAGA_TEXT)
    assert result.titulo == "Desenvolvedor(a) Backend Python - Pleno"
    assert result.empresa == "FinTech ABC"
    assert result.localidade == "São Paulo, SP"
    assert result.senioridade == "pleno"
    assert result.modelo_trabalho == "híbrido"


def test_extract_list_fields():
    result = VagaExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_VAGA_TEXT)
    assert "Python" in result.skills_obrigatorias
    assert "Docker" in result.skills_obrigatorias
    assert "Redis" in result.skills_desejadas
    assert len(result.responsabilidades) == 3


def test_extract_numeric_and_enum_fields():
    result = VagaExtractor(MockExtractor(SAMPLE_JSON)).extract(SAMPLE_VAGA_TEXT)
    assert result.experiencia_minima_anos == 3
    assert result.formacao_minima == "graduação"
    assert result.regime_contratacao == "CLT"


def test_extract_handles_markdown_fences():
    wrapped = f"```json\n{json.dumps(SAMPLE_JSON)}\n```"
    result = VagaExtractor(MockExtractor(wrapped)).extract(SAMPLE_VAGA_TEXT)
    assert result.titulo == "Desenvolvedor(a) Backend Python - Pleno"


def test_extract_handles_plain_markdown_fence():
    wrapped = f"```\n{json.dumps(SAMPLE_JSON)}\n```"
    result = VagaExtractor(MockExtractor(wrapped)).extract(SAMPLE_VAGA_TEXT)
    assert result.titulo == "Desenvolvedor(a) Backend Python - Pleno"


def test_extract_raises_on_invalid_json():
    with pytest.raises(ExtractionError, match="invalid JSON"):
        VagaExtractor(MockExtractor("not json at all")).extract(SAMPLE_VAGA_TEXT)


def test_extract_raises_on_wrong_field_type():
    """Não há mais campo obrigatório em VagaEstruturada (ver `titulo`, desvio D8),
    mas tipo errado continua tendo que falhar — é o que impede o LLM devolver
    `skills_obrigatorias` como string e isso passar batido."""
    tipo_errado = {**SAMPLE_JSON, "skills_obrigatorias": "Python, Django"}
    with pytest.raises(ExtractionError, match="validation failed"):
        VagaExtractor(MockExtractor(tipo_errado)).extract(SAMPLE_VAGA_TEXT)


def test_extract_sem_titulo_e_aceito():
    """Anúncio real que é só uma lista de requisitos, com o cargo em campo separado
    fora do corpo do texto (`vaga_3626317669` do estudo). Antes do desvio D8 isso
    derrubava a extração de uma vaga válida."""
    sem_titulo = {k: v for k, v in SAMPLE_JSON.items() if k != "titulo"}
    result = VagaExtractor(MockExtractor(sem_titulo)).extract(SAMPLE_VAGA_TEXT)
    assert result.titulo is None
    assert result.skills_obrigatorias  # o resto do anúncio continua extraído


def test_extract_minimal_vaga_all_optionals_none():
    minimal = {"titulo": "Vaga genérica"}
    result = VagaExtractor(MockExtractor(minimal)).extract("Vaga genérica")
    assert result.titulo == "Vaga genérica"
    assert result.empresa is None
    assert result.skills_obrigatorias == []
    assert result.requisitos == []


# ---------------------------------------------------------------------------
# Integration test — calls real Gemini API
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gemini_extracts_sample_vaga():
    """Passa pelo factory de propósito: é ele que decide entre Vertex AI e Developer
    API a partir de `GCP_PROJECT`. Instanciar `GeminiExtractor` direto com a api_key,
    como este teste fazia antes, testava um caminho que a aplicação não usa — e batia
    503/429 no tier gratuito enquanto o Vertex respondia normalmente."""
    from src.config import settings
    from src.extraction.factory import create_extractor

    if not settings.gcp_project and (
        not settings.gemini_api_key or settings.gemini_api_key == "No Key Provided"
    ):
        pytest.skip("nem GCP_PROJECT nem GEMINI_API_KEY configurados no .env")

    llm = create_extractor(settings)
    result = VagaExtractor(llm).extract(SAMPLE_VAGA_TEXT)

    assert isinstance(result, VagaEstruturada)
    assert result.titulo
    assert len(result.skills_obrigatorias) > 0
    assert result.modelo_trabalho in (None, "remoto", "híbrido", "presencial")
