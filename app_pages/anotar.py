"""Anotação dos 194 itens da amostra de calibração (§4.3 passos 2-3).

A tela é deliberadamente pobre em informação: rubrica da vaga, texto do currículo,
checklist e nota. Nada de estrato, posição no ranking, score ou nota do LLM — a
§4.3 passo 2 exige cegueira quanto a tudo isso, e `item_para_anotar()` é o filtro
que garante que a página não tem como exibir o que não deve.
"""
import streamlit as st

from app_pages._comum import (
    CHECKLIST,
    PLANILHA,
    RUBRICAS,
    aviso_se_faltar_arquivo,
    config,
    criterios_de_julgamento,
    indice_curriculos,
    payloads,
    rubricas,
    seletor_de_anotador,
    vagas_por_id,
)
from src.evaluation.plano.anotacao import (
    ATENDIMENTOS,
    ROTULOS_NOTA,
    carregar_checklist,
    carregar_planilha,
    checklist_do_item,
    item_para_anotar,
    itens_do_anotador,
    nota_sugerida,
    progresso,
    proximo_pendente,
    registrar_checklist,
    registrar_nota,
    salvar_checklist,
    salvar_planilha,
    texto_do_curriculo,
)

ROTULO_ATENDIMENTO = {"atende": "Atende", "parcial": "Parcial", "nao_atende": "Não atende"}

st.title("Anotar")

if not aviso_se_faltar_arquivo():
    st.stop()

cfg = config()
anotador = seletor_de_anotador()

# Estes dois mudam a cada gravação, então são lidos do disco a cada rerun.
linhas = carregar_planilha(PLANILHA)
checklist = carregar_checklist(CHECKLIST)
alvo = itens_do_anotador(linhas, anotador, cfg, cfg.seed_mestre)

if not alvo:
    st.warning("Nada para anotar com este anotador.")
    st.stop()

# --- posição atual ----------------------------------------------------------

chave_indice = f"indice_{anotador}"
if chave_indice not in st.session_state:
    # Abre onde o trabalho parou, não no começo.
    st.session_state[chave_indice] = proximo_pendente(linhas, anotador, cfg, cfg.seed_mestre) or 0

indice = min(st.session_state[chave_indice], len(alvo) - 1)
linha = alvo[indice]
item = item_para_anotar(linha, anotador)

# Deixado pelo callback de gravação, que roda antes deste rerun.
aviso = st.session_state.pop("aviso_anotar", None)
if aviso and aviso[0] == "toast":
    st.toast(aviso[1], icon=aviso[2])

prog = progresso(linhas, anotador, cfg, cfg.seed_mestre)
st.progress(
    prog["feitos"] / prog["total"] if prog["total"] else 0.0,
    text=f"{prog['feitos']} de {prog['total']} anotados · item {indice + 1} na sua ordem",
)

# --- navegação --------------------------------------------------------------


def ir_para(novo: int) -> None:
    st.session_state[chave_indice] = max(0, min(novo, len(alvo) - 1))


def ir_para_pendente() -> None:
    proximo = proximo_pendente(linhas, anotador, cfg, cfg.seed_mestre)
    if proximo is not None:
        st.session_state[chave_indice] = proximo


with st.container(horizontal=True):
    st.button("Anterior", icon=":material/arrow_back:", on_click=ir_para, args=(indice - 1,),
              disabled=indice == 0)
    st.button("Próximo", icon=":material/arrow_forward:", on_click=ir_para, args=(indice + 1,),
              disabled=indice >= len(alvo) - 1)
    st.button("Ir para o próximo pendente", icon=":material/skip_next:", on_click=ir_para_pendente)

# --- vaga e rubrica ---------------------------------------------------------

vaga = vagas_por_id().get(item.vaga_id, {})
rubrica = rubricas(RUBRICAS.stat().st_mtime).get(item.vaga_id)

if rubrica is None:
    st.error(f"Vaga {item.vaga_id} está sem rubrica — gere em `scripts/eval/gerar_rubricas.py`.")
    st.stop()

# Chaves dos widgets do julgamento: o callback de gravação lê os valores por elas.
chave_chk = {i.item_id: f"chk_{anotador}_{item.curriculo_id}_{i.item_id}" for i in rubrica.itens}
chave_nota = f"nota_{anotador}_{item.vaga_id}_{item.curriculo_id}"
chave_obs = f"obs_{anotador}_{item.vaga_id}_{item.curriculo_id}"


def salvar_e_avancar() -> None:
    """Grava e avança antes do rerun, que assim já desenha o item seguinte."""
    nota = st.session_state.get(chave_nota)
    if nota is None:
        st.session_state["aviso_anotar"] = ("erro", "Escolha a nota antes de salvar.")
        return
    preenchidos = {iid: v for iid, k in chave_chk.items() if (v := st.session_state.get(k))}
    registrar_checklist(checklist, item.vaga_id, item.curriculo_id, anotador, preenchidos)
    salvar_checklist(checklist, CHECKLIST)
    registrar_nota(linhas, item.vaga_id, item.curriculo_id, anotador, nota,
                   st.session_state.get(chave_obs, ""))
    salvar_planilha(linhas, PLANILHA)
    if len(preenchidos) < len(rubrica.itens):
        st.session_state["aviso_anotar"] = (
            "toast", f"Salvo com {len(rubrica.itens) - len(preenchidos)} item(ns) da rubrica "
            "em branco", ":material/warning:")
    else:
        st.session_state["aviso_anotar"] = ("toast", "Salvo", ":material/check:")
    ir_para(indice + 1)


# Dois slots que se alternam a cada troca de item. No navegador, um bloco que
# chega na posição de outro bloco do mesmo tipo herda os filhos dele, e as
# marcações do item anterior ficariam na tela até o rerun alcançar o formulário —
# tempo de sobra para clicar nelas achando que são do item novo. Alternando, o
# slot do item anterior vira um elemento vazio logo no início do rerun e o item
# novo nasce num bloco sem herança. (Um `st.empty()` só não basta: a fila funde o
# vazio com o bloco seguinte na mesma posição antes de enviar.)
item_exibido = (anotador, item.vaga_id, item.curriculo_id)
if st.session_state.get("item_exibido") != item_exibido:
    st.session_state["item_exibido"] = item_exibido
    st.session_state["slot_item"] = 1 - st.session_state.get("slot_item", 0)
slots = [st.empty(), st.empty()]

with slots[st.session_state["slot_item"]].container():
    st.subheader(vaga.get("cargo", item.vaga_id))
    st.caption(
        f"{vaga.get('area_tecnica', '—')} · {vaga.get('senioridade', '—')} · "
        f"{vaga.get('empresa', '—')} · {vaga.get('localidade', '—')}"
    )
    if rubrica.origem != "humano_revisado":
        st.info("Rubrica ainda não revisada por humano — revise em **Revisar rubricas** antes de anotar "
                "esta vaga (§3.2 passo 2).", icon=":material/info:")

    with st.expander("Texto do anúncio"):
        st.text(vaga.get("texto", "(indisponível)"))

    # --- currículo ----------------------------------------------------------

    payload = payloads().get(item.curriculo_id, {})
    texto = texto_do_curriculo(payload, indice_curriculos())

    st.subheader("Currículo")
    if not texto:
        st.warning(f"Texto original não encontrado (`{payload.get('arquivo_fonte', '?')}`). "
                   "Julgue pelos campos extraídos abaixo, e anote isso em observações.")
        st.json({k: payload.get(k) for k in
                 ("nome", "area_principal", "senioridade_estimada", "localidade", "skills_tecnicas")})
    else:
        with st.container(border=True, height=420):
            st.text(texto)

    # --- checklist + nota ---------------------------------------------------

    st.subheader("Julgamento")
    st.caption(
        "Conta evidência **equivalente**, não o termo literal: tecnologia, ferramenta ou "
        "experiência análoga à exigida atende o item, desde que a competência esteja demonstrada "
        "no currículo (§3.3)."
    )

    criterios = criterios_de_julgamento()
    if criterios:
        with st.expander("Critérios de julgamento — os mesmos do juiz LLM", icon=":material/gavel:"):
            # Quebra simples vira quebra de linha no markdown, senão a escala 0-3 cola numa linha só.
            st.markdown(criterios.replace("\n", "  \n"))

    marcados = checklist_do_item(checklist, item.curriculo_id, anotador, rubrica)

    with st.form(f"form_{item.vaga_id}_{item.curriculo_id}_{anotador}", border=False):
        for qual, titulo in (("obrigatorios", "Obrigatórios"), ("desejaveis", "Desejáveis")):
            itens_rubrica = getattr(rubrica, qual)
            if not itens_rubrica:
                continue
            st.markdown(f"**{titulo}**")
            for item_rubrica in itens_rubrica:
                with st.container(border=True):
                    st.markdown(f"{item_rubrica.descricao}  \n`{item_rubrica.categoria}`")
                    st.segmented_control(
                        item_rubrica.descricao,
                        options=list(ATENDIMENTOS),
                        format_func=lambda a: ROTULO_ATENDIMENTO[a],
                        default=marcados.get(item_rubrica.item_id),
                        key=chave_chk[item_rubrica.item_id],
                        label_visibility="collapsed",
                    )

        st.markdown("**Nota de relevância (§6)**")
        st.segmented_control(
            "Nota",
            options=[0, 1, 2, 3],
            format_func=lambda n: f"{n} — {ROTULOS_NOTA[n]}",
            default=item.nota_atual,
            key=chave_nota,
            label_visibility="collapsed",
        )
        st.text_input("Observações (opcional)", value=item.observacoes, key=chave_obs)

        st.form_submit_button("Salvar e avançar", icon=":material/save:", type="primary",
                              on_click=salvar_e_avancar)

    if aviso and aviso[0] == "erro":
        st.error(aviso[1])

    # A sugestão fica depois do formulário: aparece a partir do segundo rerun, como
    # conferência do que foi marcado, sem induzir a nota antes do julgamento.
    if marcados:
        sugerida = nota_sugerida(rubrica, marcados)
        if sugerida is not None and item.nota_atual is not None and sugerida != item.nota_atual:
            st.caption(
                f":material/info: o checklist salvo implicaria nota **{sugerida}** pela escala da §6, "
                f"e você deu **{item.nota_atual}**. Divergir é legítimo — a escala diz "
                '"todos ou quase todos" —, mas vale conferir se não foi desatenção.'
            )
