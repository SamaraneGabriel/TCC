"""Interface de anotação da amostra de calibração (Fase 5 do plano de avaliação).

    streamlit run app_anotacao.py

App separado do `app_streamlit.py` de propósito: aquele mostra ranking, score e
posição, que é exatamente o que o anotador **não** pode ver (§4.3 passo 2). Manter
os dois no mesmo app deixaria a informação a uma aba de distância.

Três páginas:
  Revisar rubricas  §3.2 passo 2 — revisão humana obrigatória nas vagas de calibração
  Anotar            §4.3 passos 2-3 — os 194 itens, contra a rubrica, escala 0-3
  Progresso         quanto falta, distribuição das notas, estado do double-label
"""
import streamlit as st

st.set_page_config(page_title="Anotação — plano de avaliação", page_icon=":material/rate_review:")

paginas = [
    st.Page("app_pages/anotar.py", title="Anotar", icon=":material/rate_review:", default=True),
    st.Page("app_pages/revisar_rubricas.py", title="Revisar rubricas", icon=":material/checklist:"),
    st.Page("app_pages/progresso.py", title="Progresso", icon=":material/monitoring:"),
]

st.navigation(paginas).run()
