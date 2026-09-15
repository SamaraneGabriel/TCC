"""Avaliação formal do ranqueamento (Precision@K / nDCG@K / Recall@K estimado).

Implementa o "Plano de Avaliação do Sistema de Ranqueamento de Currículos".
Distinto de `src/evaluation/{judge,metrics,runs}.py`, que são o pipeline
exploratório de LLM-as-judge já existente (ver context/08_AVALIACAO.md) — este
pacote é o estudo formal, com amostragem estratificada, calibração humana do
LLM-juiz e intervalos de confiança por bootstrap.

Implementado até agora: Fase 0 (parâmetros + confirmação de pré-requisitos).
"""
from .config import (
    ConfigAvaliacao,
    config,
    corrigir_populacao_finita,
    n0_amostra,
    tamanho_amostra,
)

__all__ = [
    "ConfigAvaliacao",
    "config",
    "corrigir_populacao_finita",
    "n0_amostra",
    "tamanho_amostra",
]
