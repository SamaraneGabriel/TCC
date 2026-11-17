"""Carregamento compartilhado entre as páginas do app de anotação.

Tudo aqui é cacheado: abrir o Qdrant e ler 385 payloads a cada rerun tornaria a
navegação item a item lenta o bastante para atrapalhar quem está anotando.

Os CSVs que **mudam durante a anotação** (planilha e checklist) ficam de fora do
cache de propósito — são lidos e reescritos a cada gravação.
"""
from pathlib import Path

import streamlit as st

from src.config import settings
from src.evaluation.plano.anotacao import DIRS_CURRICULOS, indexar_curriculos
from src.evaluation.plano.config import ConfigAvaliacao
from src.evaluation.plano.rubrica import carregar_rubricas
from src.vectorstore.store import CurriculoStore

SAIDA = Path(ConfigAvaliacao().saida_dir)
PLANILHA = SAIDA / "anotacoes_calibracao.csv"
CHECKLIST = SAIDA / "checklist_rubrica.csv"
RUBRICAS = SAIDA / "rubricas_vagas.csv"
VAGAS = SAIDA / "vagas_estudo.jsonl"
PROMPT_JUIZ = Path("src/evaluation/plano/prompts/juiz_prompt.txt")


@st.cache_resource
def config() -> ConfigAvaliacao:
    return ConfigAvaliacao()


@st.cache_resource
def payloads() -> dict[str, dict]:
    """curriculo_id -> payload do Qdrant. `cache_resource` porque abre o store
    em modo local, que trava a pasta — uma instância por processo."""
    store = CurriculoStore(path=settings.qdrant_local_path)
    return {pid: payload for pid, payload in store.listar_pontos()}


@st.cache_resource
def indice_curriculos() -> dict[str, Path]:
    return indexar_curriculos(DIRS_CURRICULOS)


@st.cache_data
def vagas_por_id() -> dict[str, dict]:
    import json

    if not VAGAS.exists():
        return {}
    with VAGAS.open(encoding="utf-8") as fh:
        return {(v := json.loads(l))["vaga_id"]: v for l in fh if l.strip()}


def rubricas(_versao: float = 0.0):
    """Rubricas do CSV. `_versao` existe para a página de revisão invalidar o
    cache depois de gravar: passa `RUBRICAS.stat().st_mtime`."""
    return carregar_rubricas(RUBRICAS)


def criterios_de_julgamento() -> str:
    """Trecho de critérios do prompt do juiz, mostrado ao anotador humano.

    Uma fonte só para as duas réguas: se humano e juiz seguissem textos
    diferentes, a concordância mediria a diferença entre os textos, não o juiz.
    Vai de "REGRA FUNDAMENTAL" até antes do schema de saída, que só interessa
    ao LLM. Sem cache: o arquivo é pequeno e pode mudar enquanto o app roda.
    """
    if not PROMPT_JUIZ.exists():
        return ""
    texto = PROMPT_JUIZ.read_text(encoding="utf-8")
    inicio, fim = texto.find("REGRA FUNDAMENTAL"), texto.find("Schema esperado:")
    if inicio < 0 or fim < inicio:
        return ""
    return texto[inicio:fim].strip()


def seletor_de_anotador() -> str:
    """Quem está anotando. Fica na sidebar, presente em todas as páginas.

    `humano_1` é o anotador principal (os 194); `humano_2` faz só os 30 do
    double-label (§4.3 passo 3) e **não pode ver** o que o primeiro respondeu,
    senão o kappa humano-humano vira concordância induzida.
    """
    st.session_state.setdefault("anotador", "humano_1")
    with st.sidebar:
        st.caption("Anotador")
        anotador = st.segmented_control(
            "Anotador",
            options=["humano_1", "humano_2"],
            format_func=lambda a: "Principal" if a == "humano_1" else "Segundo (double-label)",
            key="anotador",
            label_visibility="collapsed",
        )
    return anotador or "humano_1"


def aviso_se_faltar_arquivo() -> bool:
    """Falha cedo e com instrução, em vez de estourar no meio da anotação."""
    faltando = []
    if not PLANILHA.exists():
        faltando.append((PLANILHA, "python scripts/eval/montar_amostras.py"))
    if not RUBRICAS.exists():
        faltando.append((RUBRICAS, "python scripts/eval/gerar_rubricas.py"))
    for caminho, comando in faltando:
        st.error(f"`{caminho}` não existe. Rode antes:\n\n```\n{comando}\n```")
    return not faltando
