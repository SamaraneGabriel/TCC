"""Avaliação formal do ranqueamento (Precision@K / nDCG@K / Recall@K estimado).

Implementa o "Plano de Avaliação do Sistema de Ranqueamento de Currículos".
Distinto de `src/evaluation/{judge,metrics,runs}.py`, que são o pipeline
exploratório de LLM-as-judge já existente (ver context/08_AVALIACAO.md) — este
pacote é o estudo formal, com amostragem estratificada, calibração humana do
LLM-juiz e intervalos de confiança por bootstrap.

Implementado até agora: Fase 0 (parâmetros + confirmação de pré-requisitos),
o carregador do pool real de vagas (`pool_vagas.py`, seção 2, sobre
`data/vagas/kaggle/postings.csv`), o sorteio/cobertura/calibração da seleção de
vagas (`selecao.py`, seção 2), a amostragem estratificada por vaga + estruturas
de dados de anotação/reprodutibilidade (`amostragem.py`, seções 5 e 7) e o
vocabulário O*NET de apoio à rubrica (`onet.py`, seção 3.1 / desvio D3) e a
rubrica de qualificação KSAO (`rubrica.py`, seções 3 e 7.3) e o estado da
anotação humana da calibração (`anotacao.py`, seção 4.3 — consumido por
`app_anotacao.py`).

Numeração de seções conforme a **v2** da spec (2026-09-22), que inseriu a
rubrica KSAO como seção 3 e empurrou as demais.
"""
from .amostragem import (
    CAMPOS_LOG_REPRODUTIBILIDADE,
    CAMPOS_PLANILHA_ANOTACAO,
    ItemAmostrado,
    LinhaAnotacao,
    LogReprodutibilidade,
    estrato_a,
    estrato_b,
    estrato_c,
    gerar_linhas_anotacao,
    juntar_estratos,
    salvar_log_reprodutibilidade,
    salvar_planilha_anotacao,
    separar_amostra_calibracao,
)
from .onet import (
    AREA_PARA_SOC,
    ARQUIVO_OCUPACOES,
    ARQUIVO_TECNOLOGIAS,
    BaseONetAusente,
    Tecnologia,
    VocabularioONet,
)
from .anotacao import (
    ANOTADORES,
    ATENDIMENTOS,
    ItemParaAnotar,
    LinhaChecklist,
    carregar_checklist,
    carregar_planilha,
    item_para_anotar,
    itens_do_anotador,
    nota_sugerida,
    progresso,
    registrar_checklist,
    registrar_nota,
    salvar_checklist,
    salvar_planilha,
    selecionar_double_label,
)
from .rubrica import (
    CAMPOS_RUBRICA,
    CATEGORIAS,
    GeradorRubrica,
    ItemQualificacao,
    RubricaVaga,
    carregar_rubricas,
    conferir_rubricas,
    gerar_item_id,
    salvar_rubricas,
)
from .pool_vagas import (
    AREA_PADRAO,
    AREAS_TECNICAS,
    amostrar_pool,
    area_tecnica,
    carregar_pool,
    carregar_postings,
    cobertura_pool,
    modelo_trabalho,
    salvar_pool,
    senioridade,
    vaga_id,
)
from .config import (
    ConfigAvaliacao,
    config,
    corrigir_populacao_finita,
    n0_amostra,
    tamanho_amostra,
)
from .selecao import (
    CoberturaInsuficiente,
    VagaPool,
    checar_cobertura,
    gerar_tabela_controle,
    salvar_tabela_controle,
    selecionar_vagas_calibracao,
    sortear_com_cobertura,
    sortear_vagas_estudo,
)

__all__ = [
    "ANOTADORES",
    "ATENDIMENTOS",
    "ItemParaAnotar",
    "LinhaChecklist",
    "carregar_checklist",
    "carregar_planilha",
    "item_para_anotar",
    "itens_do_anotador",
    "nota_sugerida",
    "progresso",
    "registrar_checklist",
    "registrar_nota",
    "salvar_checklist",
    "salvar_planilha",
    "selecionar_double_label",
    "CAMPOS_RUBRICA",
    "CATEGORIAS",
    "GeradorRubrica",
    "ItemQualificacao",
    "RubricaVaga",
    "carregar_rubricas",
    "conferir_rubricas",
    "gerar_item_id",
    "salvar_rubricas",
    "AREA_PARA_SOC",
    "ARQUIVO_OCUPACOES",
    "ARQUIVO_TECNOLOGIAS",
    "BaseONetAusente",
    "Tecnologia",
    "VocabularioONet",
    "AREAS_TECNICAS",
    "AREA_PADRAO",
    "amostrar_pool",
    "area_tecnica",
    "carregar_pool",
    "carregar_postings",
    "cobertura_pool",
    "modelo_trabalho",
    "salvar_pool",
    "senioridade",
    "vaga_id",
    "ConfigAvaliacao",
    "config",
    "corrigir_populacao_finita",
    "n0_amostra",
    "tamanho_amostra",
    "CoberturaInsuficiente",
    "VagaPool",
    "checar_cobertura",
    "gerar_tabela_controle",
    "salvar_tabela_controle",
    "selecionar_vagas_calibracao",
    "sortear_com_cobertura",
    "sortear_vagas_estudo",
    "CAMPOS_LOG_REPRODUTIBILIDADE",
    "CAMPOS_PLANILHA_ANOTACAO",
    "ItemAmostrado",
    "LinhaAnotacao",
    "LogReprodutibilidade",
    "estrato_a",
    "estrato_b",
    "estrato_c",
    "gerar_linhas_anotacao",
    "juntar_estratos",
    "salvar_log_reprodutibilidade",
    "salvar_planilha_anotacao",
    "separar_amostra_calibracao",
]
