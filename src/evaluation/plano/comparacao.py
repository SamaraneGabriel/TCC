"""Métricas e testes da comparação entre configurações (`context/15_PRE_REGISTRO.md` §4–6).

- nDCG@25 com ganho **linear** (= nota 0–3) e desconto log2(posição + 1); lista ideal =
  as 25 maiores notas entre todos os currículos julgados da vaga.
- Vaga com todas as notas 0 não tem nDCG definido e sai do teste.
- Wilcoxon pareado bilateral (scipy), empates pelo método de Pratt ([proposta] do §5).
- IC 95% da diferença média por bootstrap de vagas (2.000 reamostragens, percentis).
- Holm sobre a família de secundárias.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

K = 25
N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 20261111


def dcg(ganhos: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(ganhos))


def ndcg_at_k(ranking: Sequence[str], notas: Mapping[str, int], k: int = K) -> float | None:
    """None quando a vaga não tem nenhuma nota > 0 (nDCG indefinido).

    Exige nota para todo item do top-k; a lista ideal usa todas as notas de `notas`."""
    faltando = [c for c in ranking[:k] if c not in notas]
    if faltando:
        raise KeyError(f"{len(faltando)} item(ns) do top-{k} sem nota")
    ideal = dcg(sorted(notas.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([notas[c] for c in ranking[:k]]) / ideal


def bootstrap_ic(diferencas: Sequence[float], n: int = N_BOOTSTRAP, seed: int = SEED_BOOTSTRAP,
                 nivel: float = 0.95) -> tuple[float, float]:
    d = np.asarray(diferencas, dtype=float)
    rng = np.random.default_rng(seed)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    a = (1 - nivel) / 2 * 100
    return float(np.percentile(medias, a)), float(np.percentile(medias, 100 - a))


def comparar(a: Sequence[float], b: Sequence[float]) -> dict:
    """Diferença a − b, pareada por vaga."""
    from scipy.stats import wilcoxon

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ic(d)
    if np.all(d == 0):
        w, p = float("nan"), 1.0
    else:
        r = wilcoxon(d, zero_method="pratt", alternative="two-sided")
        w, p = float(r.statistic), float(r.pvalue)
    return {
        "n": int(len(d)),
        "media_a": float(np.mean(a)), "media_b": float(np.mean(b)),
        "diferenca_media": float(d.mean()), "ic95": [lo, hi],
        "W": w, "p": p,
        "melhor": int((d > 0).sum()), "pior": int((d < 0).sum()), "empate": int((d == 0).sum()),
    }


def holm(pvalores: Mapping[str, float]) -> dict[str, float]:
    """p ajustado de Holm-Bonferroni (monótono, limitado a 1)."""
    ordem = sorted(pvalores, key=pvalores.get)
    m, ajustados, maximo = len(ordem), {}, 0.0
    for i, nome in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - i) * pvalores[nome]))
        ajustados[nome] = maximo
    return ajustados
