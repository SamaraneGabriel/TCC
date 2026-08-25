"""Pipeline de ingestão: extrai, vetoriza (skills+perfil) e indexa um currículo.

Usado por scripts/ingest_curriculos.py (CLI) e pela aba de ingestão do app_streamlit.py —
mesma lógica, orquestração diferente (retry/backoff em lote vs. progresso na UI).
"""
from .debug import dbg
from .embedding.base import BaseEmbedder
from .embedding.templates import curriculo_to_perfil_text, curriculo_to_skills_text
from .extraction.curriculo_extractor import CurriculoExtractor
from .schemas.curriculo import CurriculoEstruturado
from .vectorstore.store import CurriculoStore

_ETAPA = "ingestion"


def ingerir_curriculo(
    texto: str,
    extractor: CurriculoExtractor,
    embedder: BaseEmbedder,
    store: CurriculoStore,
) -> CurriculoEstruturado:
    """Extrai o currículo do texto bruto, gera os vetores skills+perfil e indexa no store."""
    dbg(_ETAPA, f"iniciando pipeline ({len(texto)} chars de entrada)")

    curriculo = extractor.extract(texto)

    dbg(_ETAPA, "gerando vetor de skills")
    skills_vec = embedder.embed(curriculo_to_skills_text(curriculo))
    dbg(_ETAPA, "gerando vetor de perfil")
    perfil_vec = embedder.embed(curriculo_to_perfil_text(curriculo))

    point_id = store.upsert(curriculo, skills_vec, perfil_vec)
    dbg(_ETAPA, f"indexado: {curriculo.nome} (id={point_id})")

    return curriculo
