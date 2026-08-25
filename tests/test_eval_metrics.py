import pytest

from src.evaluation.metrics import grau_medio, pct_relevantes, spearman_rank_correlation

# ── spearman_rank_correlation ────────────────────────────────────────────────


def test_spearman_identical_orders_is_one():
    ordem = ["a", "b", "c", "d"]
    assert spearman_rank_correlation(ordem, ordem) == 1.0


def test_spearman_reversed_orders_is_minus_one():
    ordem_a = ["a", "b", "c", "d"]
    ordem_b = ["d", "c", "b", "a"]
    assert spearman_rank_correlation(ordem_a, ordem_b) == -1.0


def test_spearman_partial_agreement():
    # troca so Camila<->Rodrigo de posicao, resto igual -> rho = 0.8 (conta na mao)
    ordem_sistema = ["Rafael", "Camila", "Rodrigo", "Diego"]
    ordem_llm = ["Rafael", "Rodrigo", "Camila", "Diego"]
    assert spearman_rank_correlation(ordem_sistema, ordem_llm) == pytest.approx(0.8)


def test_spearman_single_item_is_one():
    assert spearman_rank_correlation(["a"], ["a"]) == 1.0


def test_spearman_empty_is_one():
    assert spearman_rank_correlation([], []) == 1.0


def test_spearman_raises_on_mismatched_sets():
    with pytest.raises(ValueError, match="mesmos ids"):
        spearman_rank_correlation(["a", "b"], ["a", "c"])


# ── pct_relevantes ───────────────────────────────────────────────────────────


def test_pct_relevantes_mixed():
    assert pct_relevantes([0, 1, 2]) == pytest.approx(2 / 3)


def test_pct_relevantes_none_relevant():
    assert pct_relevantes([0, 0, 0]) == 0.0


def test_pct_relevantes_all_relevant():
    assert pct_relevantes([2, 2, 2]) == 1.0


def test_pct_relevantes_empty_is_zero():
    assert pct_relevantes([]) == 0.0


def test_pct_relevantes_custom_limiar():
    # limiar=2 -> so grau 2 conta como "relevante"
    assert pct_relevantes([0, 1, 2], limiar=2) == pytest.approx(1 / 3)


# ── grau_medio ───────────────────────────────────────────────────────────────


def test_grau_medio_basic():
    assert grau_medio([0, 1, 2]) == 1.0


def test_grau_medio_empty_is_zero():
    assert grau_medio([]) == 0.0


def test_grau_medio_all_same():
    assert grau_medio([2, 2]) == 2.0
