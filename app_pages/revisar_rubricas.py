"""Revisão humana das rubricas das vagas de calibração (§3.2 passo 2).

Obrigatória: enquanto uma vaga de calibração estiver com `origem=llm`, a
conferência em `conferir_rubricas()` bloqueia o estudo. A revisão pode corrigir
classificação errada, acrescentar item que o anúncio pede e o LLM não captou, e
remover item que não é qualificação do candidato.

Só as vagas de calibração aparecem por padrão. As outras 57 usam a rubrica do LLM
direto (§3.2 passo 3) — revisar todas seria trabalho que o protocolo não pede.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from app_pages._comum import RUBRICAS, SAIDA, aviso_se_faltar_arquivo, rubricas, vagas_por_id
from src.evaluation.plano.rubrica import (
    CATEGORIAS,
    ItemQualificacao,
    RubricaVaga,
    carregar_rubricas,
    gerar_item_id,
    salvar_rubricas,
)

st.title("Revisar rubricas")

if not aviso_se_faltar_arquivo():
    st.stop()

sorteio = SAIDA / "sorteio_vagas.json"
ids_calibracao = (
    json.loads(sorteio.read_text(encoding="utf-8")).get("ids_calibracao", [])
    if sorteio.exists()
    else []
)

todas = rubricas(RUBRICAS.stat().st_mtime)
vagas = vagas_por_id()

st.caption(
    "A revisão é obrigatória nas vagas de calibração (§3.2 passo 2): é contra esta rubrica "
    "que os 194 julgamentos humanos e os do LLM-juiz serão feitos."
)

mostrar_todas = st.toggle("Mostrar também as vagas sem revisão obrigatória", value=False)
ids = sorted(todas) if mostrar_todas else [i for i in ids_calibracao if i in todas]

if not ids:
    st.warning("Nenhuma rubrica para revisar.")
    st.stop()


def rotulo(vaga_id: str) -> str:
    r = todas[vaga_id]
    marca = "revisada" if r.origem == "humano_revisado" else "pendente"
    cargo = vagas.get(vaga_id, {}).get("cargo", vaga_id)
    return f"{'✓' if marca == 'revisada' else '•'} {cargo[:55]} ({marca})"


vaga_id = st.selectbox("Vaga", ids, format_func=rotulo)
rubrica = todas[vaga_id]
vaga = vagas.get(vaga_id, {})

st.subheader(vaga.get("cargo", vaga_id))
st.caption(f"{vaga.get('area_tecnica', '—')} · {vaga.get('senioridade', '—')} · "
           f"origem atual: `{rubrica.origem}` · gerada em {rubrica.data_geracao}")

for aviso in rubrica.avisos():
    st.warning(aviso, icon=":material/warning:")

with st.expander("Texto do anúncio", expanded=False):
    st.text(vaga.get("texto", "(indisponível)"))

st.markdown("**Itens da rubrica**")
st.caption(
    "Edite as células, use a última linha vazia para acrescentar item e a lixeira para remover. "
    "A ordem das linhas define os `item_id`, que ligam a rubrica ao checklist de anotação."
)

editado = st.data_editor(
    pd.DataFrame(
        [{"descricao": i.descricao, "tipo": i.tipo, "categoria": i.categoria} for i in rubrica.itens]
    ),
    num_rows="dynamic",
    width="stretch",
    column_config={
        "descricao": st.column_config.TextColumn(
            "Descrição", width="large", help="Frase curta e verificável contra um currículo"
        ),
        "tipo": st.column_config.SelectboxColumn(
            "Tipo", options=["obrigatoria", "desejavel"], required=True
        ),
        "categoria": st.column_config.SelectboxColumn(
            "Categoria", options=list(CATEGORIAS), required=True, width="medium"
        ),
    },
    key=f"editor_{vaga_id}",
)

with st.container(horizontal=True):
    salvar = st.button("Salvar como revisada", icon=":material/check:", type="primary")
    st.caption("Grava `origem=humano_revisado`, que é o que libera a vaga para anotação.")

if salvar:
    linhas = [
        r for _, r in editado.iterrows() if str(r["descricao"]).strip()
    ]
    nova = RubricaVaga(
        vaga_id=vaga_id,
        itens=[
            ItemQualificacao(
                item_id=gerar_item_id(vaga_id, n),
                descricao=str(r["descricao"]).strip(),
                tipo=r["tipo"],
                categoria=r["categoria"],
            )
            for n, r in enumerate(linhas, start=1)
        ],
        origem="humano_revisado",
        data_geracao=datetime.now().isoformat(timespec="seconds"),
        modelo=rubrica.modelo,
        ocupacoes_onet=rubrica.ocupacoes_onet,
    )
    problemas = nova.problemas()
    if problemas:
        for p in problemas:
            st.error(p)
    else:
        # Reescreve o CSV inteiro preservando a ordem das vagas.
        atuais = carregar_rubricas(RUBRICAS)
        atuais[vaga_id] = nova
        salvar_rubricas(list(atuais.values()), RUBRICAS)
        st.success(f"Rubrica de {vaga_id} salva como revisada ({len(nova.itens)} itens).")
        st.rerun()

pendentes = [i for i in ids_calibracao if i in todas and todas[i].origem != "humano_revisado"]
if pendentes:
    st.info(f"Ainda pendentes de revisão: {', '.join(pendentes)}", icon=":material/pending:")
else:
    st.success("Todas as vagas de calibração estão revisadas.", icon=":material/task_alt:")
