"""Kappa de Cohen (src/evaluation/plano/concordancia.py), conferido contra contas à mão."""
import math

import pytest

from src.evaluation.plano.concordancia import binarizar, kappa, matriz_confusao, resumo


def test_concordancia_perfeita_da_1():
    assert kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert kappa([0, 1, 2, 3], [0, 1, 2, 3], "quadratico") == pytest.approx(1.0)


def test_kappa_simples_exemplo_a_mao():
    # Binário: 2x2 com [[20, 5], [10, 15]], n = 50.
    # po = 35/50 = 0,70; marginais humano (25, 25), juiz (30, 20)
    # pe = (25*30 + 25*20) / 2500 = 0,50  ->  kappa = (0,70 - 0,50) / 0,50 = 0,40
    h = [0] * 25 + [1] * 25
    j = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
    assert kappa(h, j, categorias=(0, 1)) == pytest.approx(0.40)


def test_kappa_quadratico_exemplo_a_mao():
    # 3 categorias (0,1,2), pesos (i-j)^2/4. Pares: (0,0) (1,1) (2,2) (0,1) (1,2) (0,2)
    # observado: cada célula 1/6. Ponderado obs = [1/4 + 1/4 + 1] / 6 = 1,5/6 = 0,25
    # marginais humano (3,2,1)/6, juiz (1,2,3)/6
    # esperado ponderado = sum_ij w_ij * h_i * j_j
    h = [0, 1, 2, 0, 1, 0]
    j = [0, 1, 2, 1, 2, 2]
    hm, jm = [3 / 6, 2 / 6, 1 / 6], [1 / 6, 2 / 6, 3 / 6]
    esp = sum(((a - b) ** 2 / 4) * hm[a] * jm[b] for a in range(3) for b in range(3))
    assert kappa(h, j, "quadratico", categorias=(0, 1, 2)) == pytest.approx(1 - 0.25 / esp)


def test_discordar_por_3_pesa_mais_que_por_1_no_quadratico():
    base = [0, 1, 2, 3] * 5
    perto = [min(n + 1, 3) for n in base]
    longe = [3 - n for n in base]
    assert kappa(base, perto, "quadratico") > kappa(base, longe, "quadratico")


def test_matriz_linhas_humano_colunas_juiz():
    m = matriz_confusao([0, 0, 1], [1, 0, 1])
    assert m[0, 1] == 1 and m[0, 0] == 1 and m[1, 1] == 1 and m.sum() == 3


def test_uma_so_nota_dos_dois_da_nan():
    assert math.isnan(kappa([0, 0, 0], [0, 0, 0]))


def test_binarizar():
    assert binarizar([0, 1, 2, 3], 2) == [0, 0, 1, 1]


def test_resumo_erro_medio_e_direcao():
    r = resumo([0, 1, 2], [1, 1, 1])
    assert r["erro_medio_juiz_menos_humano"] == pytest.approx(0.0)
    assert r["juiz_mais_alto"] == 1 and r["juiz_mais_baixo"] == 1
    assert r["binarizado"][">=1"]["positivos_humano"] == 2
    assert r["binarizado"][">=1"]["positivos_juiz"] == 3


def test_tamanhos_diferentes_falham():
    with pytest.raises(ValueError):
        matriz_confusao([0], [0, 1])
