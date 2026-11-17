"""Baseline lexical BM25 (configuração BM25 do `context/15_PRE_REGISTRO.md` §2).

Documento = texto original do currículo; consulta = texto original da vaga. Nada passa
pela extração do LLM nem pelos embeddings — é o piso "busca por palavras".

Parâmetros fixados em 2026-11-11, antes de cruzar este ranking com qualquer nota:

- **Tokenização:** minúsculas; tokens = sequências alfanuméricas, mantendo sufixos
  técnicos `+`/`#` e pontos internos (`c++`, `c#`, `node.js`, `asp.net`). Sem stemming.
- **Stopwords:** a lista padrão de inglês do Lucene (`EnglishAnalyzer`, 33 palavras) —
  currículos e vagas do estudo estão em inglês.
- **k1 = 1,2 e b = 0,75:** os defaults do Lucene/Elasticsearch (Robertson e Zaragoza, 2009,
  recomendam k1 entre 1,2 e 2,0 e b = 0,75).
- **IDF:** variante do Lucene, `ln(1 + (N - n + 0,5) / (n + 0,5))`, sempre positiva.
- **Consulta:** cada ocorrência de um termo na consulta soma de novo (peso linear na
  frequência do termo na consulta, como a `match` query do Elasticsearch).
- **Empates:** desempate pelo id do currículo, para a ordem ser determinística.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping

K1 = 1.2
B = 0.75

STOPWORDS_LUCENE_EN = frozenset("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[+#]+|(?:\.[a-z0-9]+)+)?")


def tokenizar(texto: str) -> list[str]:
    return [t for t in _TOKEN.findall(texto.lower()) if t not in STOPWORDS_LUCENE_EN]


class BM25:
    def __init__(self, documentos: Mapping[str, str], k1: float = K1, b: float = B):
        if not documentos:
            raise ValueError("sem documentos")
        self.k1, self.b = k1, b
        self._tf: dict[str, Counter] = {d: Counter(tokenizar(t)) for d, t in documentos.items()}
        self._len = {d: sum(tf.values()) for d, tf in self._tf.items()}
        self._avgdl = sum(self._len.values()) / len(self._len)
        df: Counter = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def pontuar(self, consulta: str) -> dict[str, float]:
        termos = tokenizar(consulta)
        scores = {}
        for d, tf in self._tf.items():
            norm = self.k1 * (1 - self.b + self.b * self._len[d] / self._avgdl)
            s = 0.0
            for t in termos:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            scores[d] = s
        return scores

    def ranquear(self, consulta: str) -> list[str]:
        scores = self.pontuar(consulta)
        return sorted(scores, key=lambda d: (-scores[d], d))


def ranquear_bm25(documentos: Mapping[str, str], consultas: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """(id_consulta, texto) -> ranking completo dos documentos."""
    idx = BM25(documentos)
    return {qid: idx.ranquear(texto) for qid, texto in consultas}
