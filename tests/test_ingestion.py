import json

from qdrant_client import QdrantClient

from src.embedding.base import BaseEmbedder
from src.extraction.base import BaseExtractor
from src.extraction.curriculo_extractor import CurriculoExtractor
from src.dedup import hash_conteudo, id_deterministico
from src.ingestion import ingerir_curriculo
from src.vectorstore.store import CurriculoStore

VECTOR_SIZE = 8


class MockExtractor(BaseExtractor):
    def __init__(self, response: dict):
        self._response = json.dumps(response)

    def call(self, prompt: str) -> str:
        return self._response


class FakeEmbedder(BaseEmbedder):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * VECTOR_SIZE for _ in texts]


def make_store() -> CurriculoStore:
    store = CurriculoStore.__new__(CurriculoStore)
    store.client = QdrantClient(":memory:")
    store.create_collection(vector_size=VECTOR_SIZE)
    return store


def test_ingerir_curriculo_extracts_embeds_and_stores():
    extractor = CurriculoExtractor(MockExtractor({"nome": "Ana Silva", "skills_tecnicas": ["Python"]}))
    store = make_store()

    curriculo = ingerir_curriculo("texto qualquer", extractor, FakeEmbedder(), store)

    assert curriculo.nome == "Ana Silva"
    assert store.count() == 1


def test_ingerir_curriculo_returns_the_extracted_curriculo():
    extractor = CurriculoExtractor(MockExtractor({"nome": "Bruno Costa"}))
    store = make_store()

    resultado = ingerir_curriculo("texto", extractor, FakeEmbedder(), store)

    assert resultado.nome == "Bruno Costa"


def test_ingerir_curriculo_e_idempotente():
    """Mesmo texto -> mesmo ponto. Antes, cada ingestão sorteava um UUID e o
    mesmo candidato acabava indexado várias vezes (ver src/dedup.py)."""
    store = make_store()
    for nome in ("Ana Silva", "Ana Silva Souza"):  # extração pode variar
        extractor = CurriculoExtractor(MockExtractor({"nome": nome, "skills_tecnicas": ["Python"]}))
        ingerir_curriculo("mesmo texto", extractor, FakeEmbedder(), store)

    assert store.count() == 1
    pid, payload = store.listar_pontos()[0]
    assert pid == id_deterministico(hash_conteudo("mesmo texto"))
    assert payload["nome"] == "Ana Silva Souza"  # a reingestão sobrescreve


def test_ingerir_curriculo_grava_a_origem_no_payload():
    extractor = CurriculoExtractor(MockExtractor({"nome": "Ana Silva"}))
    store = make_store()

    ingerir_curriculo("texto do cv", extractor, FakeEmbedder(), store, arquivo_fonte="cv_ana.txt")

    _, payload = store.listar_pontos()[0]
    assert payload["arquivo_fonte"] == "cv_ana.txt"
    assert payload["conteudo_hash"] == hash_conteudo("texto do cv")


def test_ingerir_textos_diferentes_geram_pontos_diferentes():
    store = make_store()
    for texto in ("cv da ana", "cv do bruno"):
        extractor = CurriculoExtractor(MockExtractor({"nome": "X"}))
        ingerir_curriculo(texto, extractor, FakeEmbedder(), store)
    assert store.count() == 2
