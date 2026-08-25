"""
Unit and integration tests for the embedding layer.

Run all unit tests (no API key needed):
    pytest tests/test_embedding.py

Run including integration test (requires GEMINI_API_KEY in .env):
    pytest tests/test_embedding.py -m integration
"""
import asyncio

import pytest

from src.embedding.base import BaseEmbedder
from src.embedding.templates import (
    curriculo_to_perfil_text,
    curriculo_to_skills_text,
    vaga_to_perfil_text,
    vaga_to_skills_text,
)
from src.schemas.curriculo import CurriculoEstruturado, Experiencia, Formacao
from src.schemas.vaga import VagaEstruturada

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CURRICULO_COMPLETO = CurriculoEstruturado(
    nome="Maria Oliveira",
    email="maria@email.com",
    localidade="Belo Horizonte, MG",
    senioridade_estimada="pleno",
    anos_experiencia_estimados=4.5,
    area_principal="Engenharia de Software",
    skills_tecnicas=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    skills_comportamentais=["comunicação", "trabalho em equipe"],
    modelo_trabalho_preferido="remoto",
    experiencias=[
        Experiencia(
            cargo="Desenvolvedora Backend",
            empresa="TechCorp",
            duracao_meses=24,
            descricao="Desenvolvimento de APIs REST com FastAPI e PostgreSQL",
            skills_utilizadas=["Python", "FastAPI", "PostgreSQL"],
        )
    ],
    formacoes=[
        Formacao(grau="graduação", curso="Ciência da Computação", instituicao="UFMG", ano_conclusao=2020)
    ],
)

CURRICULO_MINIMO = CurriculoEstruturado(nome="João Silva")

VAGA_COMPLETA = VagaEstruturada(
    titulo="Desenvolvedor(a) Backend Python - Pleno",
    empresa="FinTech ABC",
    localidade="São Paulo, SP",
    modelo_trabalho="híbrido",
    area="Engenharia de Software",
    senioridade="pleno",
    responsabilidades=["Desenvolver e manter APIs REST", "Participar de code reviews"],
    skills_obrigatorias=["Python", "FastAPI", "PostgreSQL", "Docker"],
    skills_desejadas=["Redis", "Kafka", "AWS"],
    experiencia_minima_anos=3,
    formacao_minima="graduação",
)

VAGA_MINIMA = VagaEstruturada(titulo="Vaga genérica")


class FakeEmbedder(BaseEmbedder):
    """Embedder determinístico — usado para testar wiring sem chamar rede."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(len(t.split()))] for t in texts]


# ---------------------------------------------------------------------------
# Templates — funções puras, sem rede
# ---------------------------------------------------------------------------


def test_skills_text_lists_technical_skills():
    text = curriculo_to_skills_text(CURRICULO_COMPLETO)
    assert "Python" in text
    assert "FastAPI" in text
    assert "AWS" in text


def test_skills_text_handles_empty_skills():
    text = curriculo_to_skills_text(CURRICULO_MINIMO)
    assert text == "sem skills técnicas informadas"


def test_perfil_text_includes_area_and_senioridade():
    text = curriculo_to_perfil_text(CURRICULO_COMPLETO)
    assert "Engenharia de Software" in text
    assert "pleno" in text
    assert "4.5 anos" in text


def test_perfil_text_includes_formacao_experiencia_e_preferencia():
    text = curriculo_to_perfil_text(CURRICULO_COMPLETO)
    assert "UFMG" in text
    assert "TechCorp" in text
    assert "remoto" in text


def test_perfil_text_handles_all_optionals_none():
    text = curriculo_to_perfil_text(CURRICULO_MINIMO)
    assert "área não especificada" in text
    assert "experiência não informada" in text
    assert "não informado" in text


def test_vaga_skills_text_lists_obrigatorias_e_desejadas():
    text = vaga_to_skills_text(VAGA_COMPLETA)
    assert "Python" in text
    assert "FastAPI" in text
    assert "Redis" in text


def test_vaga_skills_text_handles_empty_skills():
    text = vaga_to_skills_text(VAGA_MINIMA)
    assert text == "sem skills técnicas informadas"


def test_vaga_perfil_text_includes_titulo_area_e_senioridade():
    text = vaga_to_perfil_text(VAGA_COMPLETA)
    assert "Desenvolvedor(a) Backend Python - Pleno" in text
    assert "Engenharia de Software" in text
    assert "pleno" in text
    assert "3.0 anos" in text  # experiencia_minima_anos e float, Pydantic converte 3 -> 3.0


def test_vaga_perfil_text_includes_requisitos_e_modelo_trabalho():
    text = vaga_to_perfil_text(VAGA_COMPLETA)
    assert "graduação" in text
    assert "code reviews" in text
    assert "híbrido" in text


def test_vaga_perfil_text_handles_all_optionals_none():
    text = vaga_to_perfil_text(VAGA_MINIMA)
    assert "área não especificada" in text
    assert "experiência não informada" in text
    assert "não informado" in text


# ---------------------------------------------------------------------------
# BaseEmbedder — wiring
# ---------------------------------------------------------------------------


def test_embed_delegates_to_embed_batch():
    embedder = FakeEmbedder()
    assert embedder.embed("abc") == embedder.embed_batch(["abc"])[0]


def test_embed_batch_preserves_order():
    embedder = FakeEmbedder()
    vectors = embedder.embed_batch(["a", "bb", "ccc"])
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_aembed_batch_matches_sync():
    embedder = FakeEmbedder()
    result = asyncio.run(embedder.aembed_batch(["abc", "de"]))
    assert result == embedder.embed_batch(["abc", "de"])


def test_aembed_matches_sync():
    embedder = FakeEmbedder()
    result = asyncio.run(embedder.aembed("abc"))
    assert result == embedder.embed("abc")


def test_factory_raises_on_unknown_provider():
    from src.config import Settings
    from src.embedding.factory import create_embedder

    s = Settings(embedding_provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedder(s)


# ---------------------------------------------------------------------------
# Integration test — calls real Gemini API
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gemini_embeds_curriculo():
    from src.config import settings
    from src.embedding.providers.gemini import GeminiEmbedder

    if not settings.gemini_api_key or settings.gemini_api_key == "No Key Provided":
        pytest.skip("GEMINI_API_KEY not set in .env")

    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
    )
    skills_vec = embedder.embed(curriculo_to_skills_text(CURRICULO_COMPLETO))
    perfil_vec = embedder.embed(curriculo_to_perfil_text(CURRICULO_COMPLETO))

    assert len(skills_vec) == settings.embedding_dimension
    assert len(perfil_vec) == settings.embedding_dimension
    assert skills_vec != perfil_vec


@pytest.mark.integration
def test_gemini_embeds_vaga():
    from src.config import settings
    from src.embedding.providers.gemini import GeminiEmbedder

    if not settings.gemini_api_key or settings.gemini_api_key == "No Key Provided":
        pytest.skip("GEMINI_API_KEY not set in .env")

    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
    )
    skills_vec = embedder.embed(vaga_to_skills_text(VAGA_COMPLETA))
    perfil_vec = embedder.embed(vaga_to_perfil_text(VAGA_COMPLETA))

    assert len(skills_vec) == settings.embedding_dimension
    assert len(perfil_vec) == settings.embedding_dimension
    assert skills_vec != perfil_vec
