"""
Tests for CurriculoStore — two named vectors (skills + perfil) with RRF fusion.
Uses QdrantClient in-memory mode, no Docker required.
"""
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion

from src.schemas.curriculo import CurriculoEstruturado, Experiencia
from src.vectorstore.store import COLLECTION_NAME, VEC_PERFIL, VEC_SKILLS, CurriculoStore

VECTOR_SIZE = 8


def make_store() -> CurriculoStore:
    store = CurriculoStore.__new__(CurriculoStore)
    store.client = QdrantClient(":memory:")
    return store


def make_curriculo(nome: str, senioridade: str, skills: list[str], modelo: str) -> CurriculoEstruturado:
    return CurriculoEstruturado(
        nome=nome,
        localidade="Belo Horizonte, MG",
        senioridade_estimada=senioridade,
        skills_tecnicas=skills,
        modelo_trabalho_preferido=modelo,
    )


def vec(seed: float, size: int = VECTOR_SIZE) -> list[float]:
    """Vetor constante normalizado a partir de um seed."""
    raw = [(seed + i * 0.1) for i in range(size)]
    norm = sum(x ** 2 for x in raw) ** 0.5
    return [x / norm for x in raw]


@pytest.fixture
def store() -> CurriculoStore:
    s = make_store()
    s.create_collection(vector_size=VECTOR_SIZE)
    return s


# ── collection ─────────────────────────────────────────────────────────────

def test_create_collection_has_both_vectors(store: CurriculoStore):
    info = store.client.get_collection(COLLECTION_NAME)
    vector_names = set(info.config.params.vectors.keys())
    assert VEC_SKILLS in vector_names
    assert VEC_PERFIL in vector_names


def test_create_collection_idempotent(store: CurriculoStore):
    store.create_collection(vector_size=VECTOR_SIZE)  # should not raise
    assert store.count() == 0


# ── upsert ──────────────────────────────────────────────────────────────────

def test_upsert_returns_id(store: CurriculoStore):
    c = make_curriculo("Maria", "pleno", ["Python"], "remoto")
    cid = store.upsert(c, skills_vector=vec(0.1), perfil_vector=vec(0.2))
    assert isinstance(cid, str) and len(cid) == 36  # UUID format
    assert store.count() == 1


def test_upsert_custom_id(store: CurriculoStore):
    c = make_curriculo("Carlos", "senior", ["Java"], "hibrido")
    fixed = "11111111-1111-1111-1111-111111111111"
    returned = store.upsert(c, vec(0.3), vec(0.4), curriculo_id=fixed)
    assert returned == fixed


def test_upsert_overwrites_same_id(store: CurriculoStore):
    c = make_curriculo("Ana", "junior", ["JS"], "remoto")
    fixed = "22222222-2222-2222-2222-222222222222"
    store.upsert(c, vec(0.1), vec(0.2), curriculo_id=fixed)
    store.upsert(c, vec(0.5), vec(0.6), curriculo_id=fixed)
    assert store.count() == 1


# ── search_by (vetor individual) ───────────────────────────────────────────

def test_search_by_skills(store: CurriculoStore):
    for i, nome in enumerate(["Dev A", "Dev B", "Dev C"]):
        c = make_curriculo(nome, "pleno", ["Python"], "remoto")
        store.upsert(c, skills_vector=vec(i * 0.1), perfil_vector=vec(0.9))

    results = store.search_by(vec(0.0), using=VEC_SKILLS, limit=3)
    assert len(results) == 3
    assert results[0].nome == "Dev A"  # vec(0.0) mais proximo de vec(0.0)


def test_search_by_perfil(store: CurriculoStore):
    for i, nome in enumerate(["Dev A", "Dev B", "Dev C"]):
        c = make_curriculo(nome, "pleno", ["Python"], "remoto")
        store.upsert(c, skills_vector=vec(0.9), perfil_vector=vec(i * 0.1))

    results = store.search_by(vec(0.2), using=VEC_PERFIL, limit=3)
    assert len(results) > 0
    assert results[0].nome == "Dev C"  # vec(0.2) mais proximo de vec(0.2)


# ── search (fusao RRF) ──────────────────────────────────────────────────────

def test_search_fused_returns_results(store: CurriculoStore):
    for i, nome in enumerate(["Dev A", "Dev B", "Dev C"]):
        c = make_curriculo(nome, "pleno", ["Python"], "remoto")
        store.upsert(c, vec(i * 0.1), vec(i * 0.1))

    results = store.search(vec(0.0), vec(0.0), limit=3)
    assert len(results) == 3


def test_search_fused_scores_are_positive(store: CurriculoStore):
    c = make_curriculo("Maria", "pleno", ["Python"], "remoto")
    store.upsert(c, vec(0.5), vec(0.5))
    results = store.search(vec(0.5), vec(0.5), limit=1)
    assert results[0].score > 0


def test_search_fused_dbsf(store: CurriculoStore):
    c = make_curriculo("Maria", "pleno", ["Python"], "remoto")
    store.upsert(c, vec(0.5), vec(0.5))
    results = store.search(vec(0.5), vec(0.5), limit=1, fusion=Fusion.DBSF)
    assert len(results) == 1


# ── filtros ────────────────────────────────────────────────────────────────

def test_filter_senioridade(store: CurriculoStore):
    for nome, seniority in [("Pleno A", "pleno"), ("Pleno B", "pleno"), ("Senior X", "senior")]:
        c = make_curriculo(nome, seniority, ["Python"], "remoto")
        store.upsert(c, vec(0.1), vec(0.1))

    results = store.search(vec(0.1), vec(0.1), limit=5, senioridade="pleno")
    assert len(results) == 2
    assert all(r.senioridade_estimada == "pleno" for r in results)


def test_filter_modelo_trabalho(store: CurriculoStore):
    store.upsert(make_curriculo("Remote", "pleno", ["Python"], "remoto"), vec(0.1), vec(0.1))
    store.upsert(make_curriculo("Office", "pleno", ["Python"], "presencial"), vec(0.1), vec(0.1))

    results = store.search(vec(0.1), vec(0.1), limit=5, modelo_trabalho="remoto")
    assert len(results) == 1
    assert results[0].modelo_trabalho_preferido == "remoto"


def test_filter_combined(store: CurriculoStore):
    store.upsert(make_curriculo("Match",    "pleno",  ["Python"], "remoto"),     vec(0.1), vec(0.1))
    store.upsert(make_curriculo("No Snr",   "junior", ["Python"], "remoto"),     vec(0.1), vec(0.1))
    store.upsert(make_curriculo("No Model", "pleno",  ["Python"], "presencial"), vec(0.1), vec(0.1))

    results = store.search(vec(0.1), vec(0.1), limit=5, senioridade="pleno", modelo_trabalho="remoto")
    assert len(results) == 1
    assert results[0].nome == "Match"


def test_filter_on_search_by(store: CurriculoStore):
    store.upsert(make_curriculo("Pleno",  "pleno",  ["Python"], "remoto"), vec(0.1), vec(0.1))
    store.upsert(make_curriculo("Senior", "senior", ["Python"], "remoto"), vec(0.1), vec(0.1))

    results = store.search_by(vec(0.1), using=VEC_SKILLS, limit=5, senioridade="pleno")
    assert len(results) == 1
    assert results[0].senioridade_estimada == "pleno"


# ── payload ────────────────────────────────────────────────────────────────

def test_search_result_fields(store: CurriculoStore):
    c = make_curriculo("Carlos Souza", "senior", ["Go", "Kubernetes"], "remoto")
    store.upsert(c, vec(0.5), vec(0.5))

    results = store.search(vec(0.5), vec(0.5), limit=1)
    r = results[0]
    assert r.nome == "Carlos Souza"
    assert r.senioridade_estimada == "senior"
    assert "Go" in r.skills_tecnicas
    assert r.modelo_trabalho_preferido == "remoto"
    assert r.localidade == "Belo Horizonte, MG"


# ── delete ─────────────────────────────────────────────────────────────────

def test_delete(store: CurriculoStore):
    c = make_curriculo("To Delete", "junior", ["C++"], "presencial")
    cid = store.upsert(c, vec(0.9), vec(0.9))
    assert store.count() == 1
    store.delete(cid)
    assert store.count() == 0
