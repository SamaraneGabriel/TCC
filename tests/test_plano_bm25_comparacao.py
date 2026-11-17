import math

import pytest

from src.evaluation.plano.bm25 import BM25, tokenizar
from src.evaluation.plano.comparacao import bootstrap_ic, comparar, dcg, holm, ndcg_at_k


def test_tokenizar_mantem_termos_tecnicos_e_tira_stopwords():
    assert tokenizar("The C++ and C# dev, with Node.js and ASP.NET.") == ["c++", "c#", "dev", "node.js", "asp.net"]


def test_bm25_ordena_por_relevancia_lexical_e_desempata_por_id():
    docs = {"b": "python django python", "a": "java spring", "c": "python", "d": "java spring"}
    idx = BM25(docs)
    r = idx.ranquear("python")
    assert r[:2] == ["b", "c"] or r[:2] == ["c", "b"]
    assert set(r[2:]) == {"a", "d"} and r[2:] == ["a", "d"]  # empate em 0 -> ordem do id


def test_bm25_idf_positivo_mesmo_para_termo_em_todos_os_docs():
    idx = BM25({"x": "java", "y": "java"})
    assert all(s > 0 for s in idx.pontuar("java").values())


def test_ndcg_linear():
    notas = {"a": 3, "b": 0, "c": 1, "d": 0}
    assert ndcg_at_k(["a", "c", "b", "d"], notas, k=2) == 1.0
    esperado = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["c", "a", "b", "d"], notas, k=2) == pytest.approx(esperado)
    assert dcg([2, 2]) == pytest.approx(2 + 2 / math.log2(3))


def test_ndcg_indefinido_sem_relevante_e_erro_sem_nota():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=2) is None
    with pytest.raises(KeyError):
        ndcg_at_k(["a", "z"], {"a": 1, "b": 0}, k=2)


def test_holm():
    aj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert aj["x"] == pytest.approx(0.03)
    assert aj["z"] == pytest.approx(0.06)
    assert aj["y"] == pytest.approx(0.06)  # monótono


def test_comparar_e_bootstrap():
    a = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55, 0.65]
    b = [x - 0.1 for x in a]
    r = comparar(a, b)
    assert r["diferenca_media"] == pytest.approx(0.1)
    assert r["melhor"] == 8 and r["p"] < 0.05
    lo, hi = bootstrap_ic([0.1] * 10)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
