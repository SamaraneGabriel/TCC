import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
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


class CurriculoStore:
    def __init__(self, host: str = "localhost", port: int = 6333, path: str | None = None):
        """`path` usa o modo local embarcado do qdrant-client (sem servidor/Docker),
        persistindo em disco — util para dev/teste. Se omitido, conecta em host:port."""
        self.client = QdrantClient(path=path) if path else QdrantClient(host=host, port=port)

    def create_collection(self, vector_size: int, recreate: bool = False) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            if not recreate:
                dbg(_ETAPA, f"coleção '{COLLECTION_NAME}' já existe, mantendo")
                return
            dbg(_ETAPA, f"recriando coleção '{COLLECTION_NAME}'")
            self.client.delete_collection(COLLECTION_NAME)
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
        }
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
        dbg(_ETAPA, f"upsert: {curriculo.nome} (id={point_id})")
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
    ) -> list[SearchResult]:
        """
        Busca por fusao dos dois vetores.

        Fusion.RRF  — Reciprocal Rank Fusion: combina rankings, robusto a diferenca de escala.
        Fusion.DBSF — Distribution-Based Score Fusion: normaliza scores antes de combinar,
                      util quando as distribuicoes de score dos dois vetores sao muito diferentes.
        """
        query_filter = self._build_filter(senioridade, localidade, modelo_trabalho)
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
    ) -> list[SearchResult]:
        """Busca por um unico vetor (skills ou perfil). Util para inspecao e comparacao."""
        query_filter = self._build_filter(senioridade, localidade, modelo_trabalho)
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

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    # ------------------------------------------------------------------

    def _build_filter(
        self,
        senioridade: str | None,
        localidade: str | None,
        modelo_trabalho: str | None,
    ) -> Filter | None:
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
        return Filter(must=conditions) if conditions else None

    def _to_results(self, points: list) -> list[SearchResult]:
        return [
            SearchResult(
                id=str(h.id),
                score=h.score,
                nome=h.payload.get("nome", ""),
                email=h.payload.get("email"),
                localidade=h.payload.get("localidade"),
                senioridade_estimada=h.payload.get("senioridade_estimada"),
                anos_experiencia_estimados=h.payload.get("anos_experiencia_estimados"),
                area_principal=h.payload.get("area_principal"),
                skills_tecnicas=h.payload.get("skills_tecnicas", []),
                modelo_trabalho_preferido=h.payload.get("modelo_trabalho_preferido"),
            )
            for h in points
        ]
