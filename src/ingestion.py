"""Pipeline de ingestão: extrai, vetoriza (skills+perfil) e indexa um currículo.

Usado por scripts/ingest_curriculos.py (CLI) e pela aba de ingestão do app_streamlit.py —
mesma lógica, orquestração diferente (retry/backoff em lote vs. progresso na UI).

O ID do ponto é derivado do conteúdo do currículo (`src/dedup.id_deterministico`),
não sorteado: reingerir o mesmo texto sobrescreve o mesmo ponto em vez de criar
uma duplicata. `arquivo_fonte`/`conteudo_hash` ficam no payload para rastrear o
ponto de volta ao texto que o originou.
"""
from .debug import dbg
from .dedup import hash_conteudo, id_deterministico
from .embedding.base import BaseEmbedder
from .embedding.templates import curriculo_to_perfil_text, curriculo_to_skills_text
from .extraction.curriculo_extractor import CurriculoExtractor
from .schemas.curriculo import CurriculoEstruturado
from .vectorstore.store import CurriculoStore

_ETAPA = "ingestion"


def _origem(texto: str, arquivo_fonte: str | None) -> tuple[str, dict]:
    conteudo_hash = hash_conteudo(texto)
    payload = {
        "arquivo_fonte": arquivo_fonte,
        "arquivos_fonte": [arquivo_fonte] if arquivo_fonte else [],
        "conteudo_hash": conteudo_hash,
    }
    return id_deterministico(conteudo_hash), payload


def ingerir_curriculo(
    texto: str,
    extractor: CurriculoExtractor,
    embedder: BaseEmbedder,
    store: CurriculoStore,
    arquivo_fonte: str | None = None,
) -> CurriculoEstruturado:
    """Extrai o currículo do texto bruto, gera os vetores skills+perfil e indexa no store."""
    dbg(_ETAPA, f"iniciando pipeline ({len(texto)} chars de entrada)")
    point_id, origem = _origem(texto, arquivo_fonte)

    curriculo = extractor.extract(texto)

    dbg(_ETAPA, "gerando vetores skills+perfil (lote)")
    skills_vec, perfil_vec = embedder.embed_batch(
        [curriculo_to_skills_text(curriculo), curriculo_to_perfil_text(curriculo)]
    )

    point_id = store.upsert(
        curriculo,
        skills_vec,
        perfil_vec,
        curriculo_id=point_id,
        extraction_model=extractor.model,
        embedding_model=embedder.model,
        payload_extra=origem,
    )
    dbg(_ETAPA, f"indexado: {curriculo.nome or 'Nome não informado'} (id={point_id})")

    return curriculo


async def aingerir_curriculo(
    texto: str,
    extractor: CurriculoExtractor,
    embedder: BaseEmbedder,
    store: CurriculoStore,
    arquivo_fonte: str | None = None,
) -> CurriculoEstruturado:
    """Variante async de `ingerir_curriculo` — mesma lógica, usa `aextract`/`aembed_batch`
    (rodam a chamada síncrona em thread pool por padrão) para permitir concorrência entre
    vários currículos sem bloquear o event loop."""
    dbg(_ETAPA, f"iniciando pipeline async ({len(texto)} chars de entrada)")
    point_id, origem = _origem(texto, arquivo_fonte)

    curriculo = await extractor.aextract(texto)

    dbg(_ETAPA, "gerando vetores skills+perfil (lote, async)")
    skills_vec, perfil_vec = await embedder.aembed_batch(
        [curriculo_to_skills_text(curriculo), curriculo_to_perfil_text(curriculo)]
    )

    point_id = store.upsert(
        curriculo,
        skills_vec,
        perfil_vec,
        curriculo_id=point_id,
        extraction_model=extractor.model,
        embedding_model=embedder.model,
        payload_extra=origem,
    )
    dbg(_ETAPA, f"indexado: {curriculo.nome or 'Nome não informado'} (id={point_id})")

    return curriculo
