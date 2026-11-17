import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HasIdCondition,
    MatchValue,
    PointIdsList,
    PointStruct,
    Prefetch,
    VectorParams,
)

from src.debug import dbg
from src.schemas.curriculo import CurriculoEstruturado

COLLECTION_NAME = "curriculos"
VEC_SKILLS = "skills"   # embedding das skills tecnicas (texto de palavras-chave)
VEC_PERFIL = "perfil"   # embedding do perfil narrativo (resumo + experiencia + senioridade)

_ETAPA = "vectorstore"


@dataclass
class SearchResult:
    id: str
    score: float  # fusion score (RRF/DBSF) — relativo, maior = mais relevante
    nome: str
    email: str | None
    localidade: str | None
    senioridade_estimada: str | None
    anos_experiencia_estimados: float | None
    area_principal: str | None
    skills_tecnicas: list[str]
    modelo_trabalho_preferido: str | None
    extraction_model: str | None = None
    embedding_model: str | None = None


class CurriculoStore:
    def __init__(self, host: str = "localhost", port: int = 6333, path: str | None = None):
        """`path` usa o modo local embarcado do qdrant-client (sem servidor/Docker),
        persistindo em disco — util para dev/teste. Se omitido, conecta em host:port."""
        self._local_path = path
        self.client = QdrantClient(path=path) if path else QdrantClient(host=host, port=port)

    def create_collection(self, vector_size: int, recreate: bool = False) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            if not recreate:
                dbg(_ETAPA, f"coleção '{COLLECTION_NAME}' já existe, mantendo")
                return
            dbg(_ETAPA, f"recriando coleção '{COLLECTION_NAME}'")
            self.client.delete_collection(COLLECTION_NAME)
            if self._local_path:
                # No modo local (embarcado), delete_collection não limpa de fato
                # o storage.sqlite da coleção — sem isso, os pontos antigos
                # reaparecem na coleção "recriada" (bug observado em 2026-09-14:
                # recreate=True mantinha os dados antigos, resultado misturava
                # currículos de reingestões diferentes na mesma coleção).
                pasta = Path(self._local_path) / "collection" / COLLECTION_NAME
                shutil.rmtree(pasta, ignore_errors=True)
        else:
            dbg(_ETAPA, f"criando coleção '{COLLECTION_NAME}' (dim={vector_size})")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VEC_SKILLS: VectorParams(size=vector_size, distance=Distance.COSINE),
                VEC_PERFIL: VectorParams(size=vector_size, distance=Distance.COSINE),
            },
        )

    def upsert(
        self,
        curriculo: CurriculoEstruturado,
        skills_vector: list[float],
        perfil_vector: list[float],
        curriculo_id: str | None = None,
        extraction_model: str | None = None,
        embedding_model: str | None = None,
        payload_extra: dict | None = None,
    ) -> str:
        point_id = curriculo_id or str(uuid.uuid4())
        payload = {
            "nome": curriculo.nome,
            "email": curriculo.email,
            "localidade": curriculo.localidade,
            "senioridade_estimada": curriculo.senioridade_estimada,
            "anos_experiencia_estimados": curriculo.anos_experiencia_estimados,
            "area_principal": curriculo.area_principal,
            "skills_tecnicas": curriculo.skills_tecnicas,
            "modelo_trabalho_preferido": curriculo.modelo_trabalho_preferido,
            "extraction_model": extraction_model,
            "embedding_model": embedding_model,
        }
        # Rastreabilidade da origem (arquivo_fonte/conteudo_hash) — ver src/dedup.py.
        payload.update(payload_extra or {})
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector={VEC_SKILLS: skills_vector, VEC_PERFIL: perfil_vector},
                    payload=payload,
                )
            ],
        )
        dbg(_ETAPA, f"upsert: {curriculo.nome or 'Nome não informado'} (id={point_id})")
        return point_id

    def search(
        self,
        skills_vector: list[float],
        perfil_vector: list[float],
        limit: int = 10,
        senioridade: str | None = None,
        localidade: str | None = None,
        modelo_trabalho: str | None = None,
        fusion: Fusion = Fusion.RRF,
        excluir_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Busca por fusao dos dois vetores.

        Fusion.RRF  — Reciprocal Rank Fusion: combina rankings, robusto a diferenca de escala.
        Fusion.DBSF — Distribution-Based Score Fusion: normaliza scores antes de combinar,
                      util quando as distribuicoes de score dos dois vetores sao muito diferentes.
        """
        query_filter = self._build_filter(senioridade, localidade, modelo_trabalho, excluir_ids)
        prefetch_limit = max(limit * 3, 20)
        dbg(_ETAPA, f"search: fusion={fusion.name}, limit={limit}, filtro={'sim' if query_filter else 'não'}")

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=skills_vector,
                    using=VEC_SKILLS,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=perfil_vector,
                    using=VEC_PERFIL,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
        )
        resultados = self._to_results(response.points)
        dbg(_ETAPA, f"search: {len(resultados)} resultado(s)")
        return resultados

    def search_by(
        self,
        vector: list[float],
        using: str,
        limit: int = 10,
        senioridade: str | None = None,
        localidade: str | None = None,
        modelo_trabalho: str | None = None,
        excluir_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Busca por um unico vetor (skills ou perfil). Util para inspecao e comparacao."""
        query_filter = self._build_filter(senioridade, localidade, modelo_trabalho, excluir_ids)
        dbg(_ETAPA, f"search_by: using={using}, limit={limit}, filtro={'sim' if query_filter else 'não'}")
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        resultados = self._to_results(response.points)
        dbg(_ETAPA, f"search_by: {len(resultados)} resultado(s)")
        return resultados

    def delete(self, curriculo_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[curriculo_id]),
        )

    def remover_pontos(self, ids: list[str]) -> None:
        """Remove vários pontos de uma vez.

        ⚠️ Deleção em massa: só usar com autorização explícita do usuário no
        momento (ver a regra no topo do CLAUDE.md). Existe para a deduplicação
        (`scripts/dedup_curriculos.py`), que apaga duplicatas ponto a ponto —
        nunca a coleção.
        """
        if not ids:
            return
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=list(ids)),
        )
        dbg(_ETAPA, f"removidos {len(ids)} ponto(s)")

    def atualizar_payload(self, curriculo_id: str, campos: dict) -> None:
        """Acrescenta/atualiza campos do payload sem tocar nos vetores."""
        self.client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=campos,
            points=[curriculo_id],
        )

    def listar_pontos(self, com_payload: bool = True) -> list[tuple[str, dict]]:
        """Varre a coleção inteira (scroll paginado), devolvendo (id, payload).

        Usada pela deduplicação e pelo estrato C do plano de avaliação, cujo
        universo é "todos os IDs do banco menos os TOP_POOL recuperados"
        (`context/10_PLANO_AVALIACAO.md`, seção 4.2, passo 4).
        """
        pontos: list[tuple[str, dict]] = []
        offset = None
        while True:
            lote, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=com_payload,
                with_vectors=False,
            )
            pontos.extend((str(p.id), p.payload or {}) for p in lote)
            if offset is None:
                break
        dbg(_ETAPA, f"listar_pontos: {len(pontos)} ponto(s)")
        return pontos

    def listar_ids(self) -> list[str]:
        return [pid for pid, _ in self.listar_pontos(com_payload=False)]

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    # ------------------------------------------------------------------

    def _build_filter(
        self,
        senioridade: str | None,
        localidade: str | None,
        modelo_trabalho: str | None,
        excluir_ids: list[str] | None = None,
    ) -> Filter | None:
        """`excluir_ids` tira pontos da busca antes da fusão (o estudo exclui os
        currículos sem texto original de todas as configurações)."""
        conditions = []
        if senioridade:
            conditions.append(
                FieldCondition(key="senioridade_estimada", match=MatchValue(value=senioridade))
            )
        if localidade:
            conditions.append(
                FieldCondition(key="localidade", match=MatchValue(value=localidade))
            )
        if modelo_trabalho:
            conditions.append(
                FieldCondition(key="modelo_trabalho_preferido", match=MatchValue(value=modelo_trabalho))
            )
        must_not = [HasIdCondition(has_id=list(excluir_ids))] if excluir_ids else []
        if not conditions and not must_not:
            return None
        return Filter(must=conditions or None, must_not=must_not or None)

    def _to_results(self, points: list) -> list[SearchResult]:
        return [
            SearchResult(
                id=str(h.id),
                score=h.score,
                nome=h.payload.get("nome") or "Nome não informado",
                email=h.payload.get("email"),
                localidade=h.payload.get("localidade"),
                senioridade_estimada=h.payload.get("senioridade_estimada"),
                anos_experiencia_estimados=h.payload.get("anos_experiencia_estimados"),
                area_principal=h.payload.get("area_principal"),
                skills_tecnicas=h.payload.get("skills_tecnicas", []),
                modelo_trabalho_preferido=h.payload.get("modelo_trabalho_preferido"),
                extraction_model=h.payload.get("extraction_model"),
                embedding_model=h.payload.get("embedding_model"),
            )
            for h in points
        ]
