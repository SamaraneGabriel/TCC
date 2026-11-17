"""Quanto falta anotar, como as notas estão distribuídas e o estado do double-label.

Mostra os dois anotadores lado a lado, mas **não** mostra nota por item: ver o que
o outro respondeu antes de terminar anularia a independência que o kappa
humano-humano mede (§4.3 passo 3). Só contagens agregadas.
"""
import pandas as pd
import streamlit as st

from app_pages._comum import CHECKLIST, PLANILHA, RUBRICAS, aviso_se_faltar_arquivo, config, rubricas
from src.evaluation.plano.anotacao import (
    ANOTADORES,
    ROTULOS_NOTA,
    carregar_checklist,
    carregar_planilha,
    chave,
    itens_do_anotador,
    nota_de,
    progresso,
    selecionar_double_label,
)

st.title("Progresso")

if not aviso_se_faltar_arquivo():
    st.stop()

cfg = config()
linhas = carregar_planilha(PLANILHA)
checklist = carregar_checklist(CHECKLIST)

st.subheader("Anotação")
colunas = st.columns(2)
progressos = {}
for coluna, anotador in zip(colunas, ANOTADORES):
    p = progresso(linhas, anotador, cfg, cfg.seed_mestre)
    progressos[anotador] = p
    with coluna, st.container(border=True):
        titulo = "Principal" if anotador == "humano_1" else "Segundo (double-label)"
        st.metric(titulo, f"{p['feitos']} / {p['total']}", f"{p['percentual']}%", border=False)
        st.progress(p["feitos"] / p["total"] if p["total"] else 0.0)

st.subheader("Distribuição das notas")
st.caption("Contagens agregadas apenas — nota por item não aparece aqui, para não quebrar a "
           "independência entre os dois anotadores (§4.3 passo 3).")
distribuicao = pd.DataFrame(
    {
        ("Principal" if a == "humano_1" else "Segundo"): [
            progressos[a]["distribuicao"][n] for n in (0, 1, 2, 3)
        ]
        for a in ANOTADORES
    },
    index=[f"{n} — {ROTULOS_NOTA[n]}" for n in (0, 1, 2, 3)],
)
st.bar_chart(distribuicao, horizontal=True)

st.subheader("Double-label")
alvo_dl = selecionar_double_label(linhas, cfg, cfg.seed_mestre)
feitos_dl = sum(
    1 for l in linhas if chave(l) in alvo_dl and nota_de(l, "humano_2") is not None
)
prontos_para_kappa = sum(
    1
    for l in linhas
    if chave(l) in alvo_dl
    and nota_de(l, "humano_1") is not None
    and nota_de(l, "humano_2") is not None
)
with st.container(horizontal=True):
    st.metric("Sorteados", len(alvo_dl), border=True)
    st.metric("Anotados pelo segundo", feitos_dl, border=True)
    st.metric("Com as duas notas", prontos_para_kappa, border=True,
              help="Só estes entram no cálculo do kappa humano-humano (limiar "
                   f"{cfg.kappa_limiar_humano}).")

st.subheader("Checklist da rubrica (§7.4)")
todas_rubricas = rubricas(RUBRICAS.stat().st_mtime)
esperado = {
    a: sum(
        len(todas_rubricas[l["vaga_id"]].itens)
        for l in itens_do_anotador(linhas, a, cfg, cfg.seed_mestre)
        if l["vaga_id"] in todas_rubricas
    )
    for a in ANOTADORES
}
registrado = {a: sum(1 for r in checklist.values() if r.anotador == a) for a in ANOTADORES}
st.dataframe(
    pd.DataFrame(
        {
            "Anotador": ["Principal", "Segundo"],
            "Marcações registradas": [registrado[a] for a in ANOTADORES],
            "Esperado ao terminar": [esperado[a] for a in ANOTADORES],
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Rubricas de calibração")
pendentes = [
    vid for vid in sorted({l["vaga_id"] for l in linhas})
    if vid in todas_rubricas and todas_rubricas[vid].origem != "humano_revisado"
]
if pendentes:
    st.warning(
        f"{len(pendentes)} vaga(s) de calibração ainda com a rubrica do LLM: "
        f"{', '.join(pendentes)}. A §3.2 passo 2 exige revisão humana antes da anotação valer.",
        icon=":material/warning:",
    )
else:
    st.success("Todas revisadas.", icon=":material/task_alt:")
