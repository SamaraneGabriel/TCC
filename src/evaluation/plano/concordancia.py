"""Concordância juiz LLM × anotador humano (validação do juiz, `context/15_PRE_REGISTRO.md` §7).

Kappa de Cohen implementado aqui, sem scikit-learn: são poucas linhas e o projeto não
depende dele. Fórmula (Cohen, 1968), com pesos de discordância w_ij:

    kappa = 1 - sum(w * observado) / sum(w * esperado)

`esperado` é o produto das frequências marginais (o que dois anotadores independentes,
com as mesmas distribuições de notas, concordariam por acaso). Sem pesos, w_ij = 1 fora da
diagonal; quadrático, w_ij = (i - j)^2 / (k - 1)^2.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

NOTAS = (0, 1, 2, 3)


def matriz_confusao(humano: Sequence[int], juiz: Sequence[int], categorias=NOTAS) -> np.ndarray:
    """Linhas = nota do humano, colunas = nota do juiz."""
    if len(humano) != len(juiz):
        raise ValueError("listas de tamanhos diferentes")
    pos = {c: i for i, c in enumerate(categorias)}
    m = np.zeros((len(categorias), len(categorias)), dtype=int)
    for h, j in zip(humano, juiz):
        m[pos[h], pos[j]] += 1
    return m


def kappa(
    humano: Sequence[int],
    juiz: Sequence[int],
    pesos: Literal["nenhum", "quadratico"] = "nenhum",
    categorias=NOTAS,
) -> float:
    obs = matriz_confusao(humano, juiz, categorias).astype(float)
    n = obs.sum()
    if n == 0:
        raise ValueError("sem pares")
    obs /= n
    esperado = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    k = len(categorias)
    i, j = np.indices((k, k))
    if pesos == "quadratico":
        w = (i - j) ** 2 / (k - 1) ** 2
    else:
        w = (i != j).astype(float)
    denominador = (w * esperado).sum()
    if denominador == 0:
        return float("nan")  # os dois deram sempre a mesma única nota
    return float(1 - (w * obs).sum() / denominador)


def binarizar(notas: Sequence[int], corte: int) -> list[int]:
    return [int(n >= corte) for n in notas]


def resumo(humano: Sequence[int], juiz: Sequence[int], cortes=(1, 2)) -> dict:
    """Tudo o que o pré-registro pede para a concordância, num dicionário."""
    h, j = list(humano), list(juiz)
    dif = np.array(j) - np.array(h)
    saida = {
        "pares": len(h),
        "concordancia_exata": float(np.mean(dif == 0)),
        "kappa_quadratico": kappa(h, j, "quadratico"),
        "kappa_simples": kappa(h, j, "nenhum"),
        "erro_medio_juiz_menos_humano": float(dif.mean()),
        "juiz_mais_alto": int((dif > 0).sum()),
        "juiz_mais_baixo": int((dif < 0).sum()),
        "matriz_confusao": matriz_confusao(h, j).tolist(),
        "binarizado": {},
    }
    for c in cortes:
        hb, jb = binarizar(h, c), binarizar(j, c)
        saida["binarizado"][f">={c}"] = {
            "concordancia": float(np.mean(np.array(hb) == np.array(jb))),
            "kappa": kappa(hb, jb, "nenhum", categorias=(0, 1)),
            "positivos_humano": sum(hb),
            "positivos_juiz": sum(jb),
        }
    return saida
