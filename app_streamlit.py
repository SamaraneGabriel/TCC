"""
Interface de teste do pipeline completo — não é a interface final do produto,
serve para QA manual (extração -> embedding -> retrieval -> ranking).

Duas abas:
  Buscar candidatos      — cola ou envia (PDF/TXT) o texto de uma vaga, mostra os
                            currículos indexados mais compatíveis, com score e rank.
  Ingestão de currículos — envia currículos (PDF/TXT) para extrair, vetorizar e
                            adicionar à base de busca.

Uso:
    streamlit run app_streamlit.py
"""
import streamlit as st

from src.config import settings
from src.embedding.factory import create_embedder
from src.extraction.curriculo_extractor import CurriculoExtractor
from src.extraction.factory import create_extractor
from src.extraction.vaga_extractor import VagaExtractor
from src.ingestion import ingerir_curriculo
from src.matching.ranker import rank
from src.matching.retriever import buscar_candidatos
from src.parsers.pdf_parser import parse_pdf
from src.vectorstore.store import CurriculoStore

st.set_page_config(page_title="TCC Matching — Teste", layout="wide")


@st.cache_resource
def get_pipeline():
    llm = create_extractor(settings)
    curriculo_extractor = CurriculoExtractor(llm)
    vaga_extractor = VagaExtractor(llm)
    embedder = create_embedder(settings)
    store = CurriculoStore(path=settings.qdrant_local_path)
    store.create_collection(vector_size=settings.embedding_dimension)
    return curriculo_extractor, vaga_extractor, embedder, store


curriculo_extractor, vaga_extractor, embedder, store = get_pipeline()


def extrair_texto(arquivo) -> str:
    """Extrai texto de um UploadedFile do Streamlit (PDF ou TXT)."""
    if arquivo.name.lower().endswith(".pdf"):
        return parse_pdf(arquivo.getvalue())
    return arquivo.getvalue().decode("utf-8")


st.title("TCC Matching — interface de teste")

tab_buscar, tab_ingestao = st.tabs(["Buscar candidatos", "Ingestão de currículos"])

# ── Buscar candidatos (vaga -> currículos) ──────────────────────────────────

with tab_buscar:
    st.caption(
        f"{store.count()} currículos indexados · LLM: {settings.llm_provider} · "
        f"Embedding: {settings.embedding_provider}"
    )

    arquivo_vaga = st.file_uploader("Envie a vaga em PDF/TXT", type=["pdf", "txt"], key="vaga_upload")
    texto_vaga = st.text_area(
        "...ou cole o texto da vaga",
        height=200,
        placeholder="Título, empresa, requisitos obrigatórios, diferenciais, senioridade...",
    )
    if arquivo_vaga is not None:
        texto_vaga = extrair_texto(arquivo_vaga)
        st.caption(f"Usando o texto extraído de **{arquivo_vaga.name}** ({len(texto_vaga)} caracteres).")

    top_k = st.slider("Quantos currículos retornar", 1, 15, 5)
    buscar = st.button("Buscar candidatos", type="primary")

    if buscar and texto_vaga.strip():
        with st.spinner("Extraindo dados estruturados da vaga..."):
            vaga = vaga_extractor.extract(texto_vaga)

        with st.expander("Vaga extraída (JSON)"):
            st.json(vaga.model_dump())

        with st.spinner("Buscando e ranqueando candidatos..."):
            candidatos = buscar_candidatos(vaga, embedder, store, top_k=top_k)
            ranqueados = rank(vaga, candidatos)

        if not ranqueados:
            st.warning("Nenhum currículo indexado ainda — use a aba 'Ingestão de currículos'.")
        else:
            st.subheader(f"{len(ranqueados)} candidato(s)")
            for r in ranqueados:
                c = r.candidato
                with st.container(border=True):
                    col_info, col_score = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**#{r.rank} — {c.nome}**")
                        st.caption(
                            f"{c.senioridade_estimada or '?'} · {c.localidade or '?'} · "
                            f"{c.modelo_trabalho_preferido or '?'}"
                        )
                        st.write(", ".join(c.skills_tecnicas) or "sem skills informadas")
                    with col_score:
                        st.metric("Score final", f"{r.score_final:.3f}")
                        st.caption(
                            f"Skills: {len(r.skills_match)}/{r.skills_total_vaga} · "
                            f"Senioridade: {'✓' if r.senioridade_match else '—'} · "
                            f"Local: {'✓' if r.localidade_match else '—'}"
                        )
    elif buscar:
        st.warning("Cole o texto de uma vaga ou envie um arquivo antes de buscar.")

# ── Ingestão de currículos ───────────────────────────────────────────────────

with tab_ingestao:
    st.caption("Envia currículos (PDF/TXT), extrai, vetoriza (skills+perfil) e adiciona à base de busca.")

    arquivos = st.file_uploader(
        "Currículos (PDF/TXT) — pode selecionar vários",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="curriculos_upload",
    )
    indexar = st.button("Vetorizar e indexar", type="primary")

    if indexar and arquivos:
        barra = st.progress(0.0, text="Processando...")
        sucesso = 0
        for i, arquivo in enumerate(arquivos, start=1):
            try:
                curriculo = ingerir_curriculo(extrair_texto(arquivo), curriculo_extractor, embedder, store)
                st.success(f"{arquivo.name} → {curriculo.nome}")
                sucesso += 1
            except Exception as e:
                st.error(f"{arquivo.name}: {e}")
            barra.progress(i / len(arquivos), text=f"{i}/{len(arquivos)}")

        if sucesso:
            st.info(f"{sucesso}/{len(arquivos)} indexado(s). Atualizando contagem...")
            st.rerun()
    elif indexar:
        st.warning("Selecione pelo menos um arquivo antes de indexar.")
