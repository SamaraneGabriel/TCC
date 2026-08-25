from ..debug import dbg
from ..embedding.base import BaseEmbedder
from ..embedding.templates import vaga_to_perfil_text, vaga_to_skills_text
from ..schemas.vaga import VagaEstruturada
from ..vectorstore.store import CurriculoStore, SearchResult

_ETAPA = "matching.retriever"


def buscar_candidatos(
    vaga: VagaEstruturada,
    embedder: BaseEmbedder,
    store: CurriculoStore,
    top_k: int = 10,
    senioridade: str | None = None,
    localidade: str | None = None,
    modelo_trabalho: str | None = None,
) -> list[SearchResult]:
    """Busca currículos candidatos para uma vaga via fusão dual-vetor (skills+perfil)."""
    dbg(_ETAPA, f"vetorizando vaga: {vaga.titulo!r}")
    skills_vec = embedder.embed(vaga_to_skills_text(vaga))
    perfil_vec = embedder.embed(vaga_to_perfil_text(vaga))

    filtros = {k: v for k, v in {
        "senioridade": senioridade, "localidade": localidade, "modelo_trabalho": modelo_trabalho,
    }.items() if v}
    dbg(_ETAPA, f"buscando top_k={top_k}, filtros={filtros or 'nenhum'}")

    resultados = store.search(
        skills_vec,
        perfil_vec,
        limit=top_k,
        senioridade=senioridade,
        localidade=localidade,
        modelo_trabalho=modelo_trabalho,
    )
    dbg(_ETAPA, f"{len(resultados)} candidato(s) retornado(s)")
    return resultados
