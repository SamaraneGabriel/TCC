import json

from qdrant_client import QdrantClient

from src.embedding.base import BaseEmbedder
from src.extraction.base import BaseExtractor
from src.extraction.curriculo_extractor import CurriculoExtractor
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
