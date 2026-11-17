"""Testa a amostragem estratificada por vaga (seção 4) e as estruturas de
dados de anotação/reprodutibilidade (seção 6) com universos pequenos e
determinísticos — não é o dataset real, só fixtures pra exercitar a lógica."""
import csv

import pytest

from src.evaluation.plano.amostragem import (
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
    completar_censo_estrato_a,
    escolher_vaga_censo_a,
    separar_amostra_calibracao,
)
from src.evaluation.plano.config import ConfigAvaliacao
from src.evaluation.plano.selecao import CoberturaInsuficiente


def _cfg(**overrides) -> ConfigAvaliacao:
    return ConfigAvaliacao(_env_file=None, **overrides)


# --- estrato_a ----------------------------------------------------------------


def test_estrato_a_pega_as_primeiras_k_posicoes():
    resultados = [f"c{i}" for i in range(1, 11)]  # 10 IDs, posições 1..10
    itens = estrato_a(resultados, k=5)
    assert [i.curriculo_id for i in itens] == ["c1", "c2", "c3", "c4", "c5"]
    assert [i.posicao_sistema for i in itens] == [1, 2, 3, 4, 5]
    assert all(i.estrato == "A" for i in itens)


def test_estrato_a_com_menos_resultados_que_k_nao_e_erro():
    resultados = [f"c{i}" for i in range(1, 4)]  # só 3 IDs
    itens = estrato_a(resultados, k=25)
    assert len(itens) == 3  # censo do que existe, sem levantar erro


def test_estrato_a_lista_vazia():
    assert estrato_a([], k=25) == []


# --- estrato_b ----------------------------------------------------------------


def test_estrato_b_amostra_dentro_do_intervalo_k_mais_um_ate_top_pool():
    resultados = [f"c{i}" for i in range(1, 31)]  # 30 IDs, posições 1..30
    itens = estrato_b(resultados, k=5, n_b=5, seed=42)
    assert len(itens) == 5
    assert len(set(i.curriculo_id for i in itens)) == 5
    for item in itens:
        assert item.estrato == "B"
        assert 6 <= item.posicao_sistema <= 30
        assert item.curriculo_id == f"c{item.posicao_sistema}"


def test_estrato_b_e_determinado_pela_seed():
    resultados = [f"c{i}" for i in range(1, 31)]
    a = estrato_b(resultados, k=5, n_b=5, seed=42)
    b = estrato_b(resultados, k=5, n_b=5, seed=42)
    assert [i.curriculo_id for i in a] == [i.curriculo_id for i in b]


def test_estrato_b_erro_se_n_b_maior_que_universo():
    resultados = [f"c{i}" for i in range(1, 31)]  # candidatos: 25 (posições 6..30)
    with pytest.raises(ValueError, match="estrato B"):
        estrato_b(resultados, k=5, n_b=999, seed=1)


def test_estrato_b_universo_vazio_com_n_b_zero():
    resultados = [f"c{i}" for i in range(1, 6)]  # k=5 esgota a lista -> sem candidatos
    assert estrato_b(resultados, k=5, n_b=0, seed=1) == []


def test_estrato_b_universo_vazio_com_n_b_positivo_e_erro():
    resultados = [f"c{i}" for i in range(1, 6)]
    with pytest.raises(ValueError, match="estrato B"):
        estrato_b(resultados, k=5, n_b=1, seed=1)


# --- estrato_c ------------------------------------------------------------------


def test_estrato_c_amostra_do_complemento_de_conjunto():
    todos = {f"c{i}" for i in range(1, 101)}
    top_pool = {f"c{i}" for i in range(1, 21)}
    itens = estrato_c(todos, top_pool, n_c=10, seed=7)
    assert len(itens) == 10
    assert len(set(i.curriculo_id for i in itens)) == 10
    for item in itens:
        assert item.estrato == "C"
        assert item.posicao_sistema is None
        assert item.curriculo_id not in top_pool
        assert item.curriculo_id in todos


def test_estrato_c_e_determinado_pela_seed():
    todos = {f"c{i}" for i in range(1, 101)}
    top_pool = {f"c{i}" for i in range(1, 21)}
    a = estrato_c(todos, top_pool, n_c=10, seed=7)
    b = estrato_c(todos, top_pool, n_c=10, seed=7)
    assert [i.curriculo_id for i in a] == [i.curriculo_id for i in b]


def test_estrato_c_erro_se_n_c_maior_que_universo():
    todos = {f"c{i}" for i in range(1, 21)}
    top_pool = {f"c{i}" for i in range(1, 16)}  # universo: só 5 IDs
    with pytest.raises(ValueError, match="estrato C"):
        estrato_c(todos, top_pool, n_c=10, seed=1)


# --- juntar_estratos --------------------------------------------------------------


def test_juntar_estratos_preserva_todos_os_itens():
    a = estrato_a([f"a{i}" for i in range(1, 6)], k=5)
    b = [ItemAmostrado(curriculo_id="b1", estrato="B", posicao_sistema=10)]
    c = [ItemAmostrado(curriculo_id="c1", estrato="C", posicao_sistema=None)]
    juntos = juntar_estratos(a, b, c, seed=1)
    assert len(juntos) == 7
    assert {i.curriculo_id for i in juntos} == {i.curriculo_id for i in a + b + c}
    # a "cegueira" é só de apresentação: os campos continuam presentes
    assert {i.estrato for i in juntos} == {"A", "B", "C"}


def test_juntar_estratos_e_determinado_pela_seed():
    a = estrato_a([f"a{i}" for i in range(1, 6)], k=5)
    b = [ItemAmostrado(curriculo_id=f"b{i}", estrato="B", posicao_sistema=10 + i) for i in range(5)]
    c = [ItemAmostrado(curriculo_id=f"c{i}", estrato="C", posicao_sistema=None) for i in range(5)]
    x = juntar_estratos(a, b, c, seed=99)
    y = juntar_estratos(a, b, c, seed=99)
    assert [i.curriculo_id for i in x] == [i.curriculo_id for i in y]


# --- gerar_linhas_anotacao -----------------------------------------------------------


def test_gerar_linhas_anotacao_mapeia_campos_e_zera_notas():
    itens = [
        ItemAmostrado(curriculo_id="c1", estrato="A", posicao_sistema=1),
        ItemAmostrado(curriculo_id="c2", estrato="C", posicao_sistema=None),
    ]
    linhas = gerar_linhas_anotacao("v1", itens, papel="producao_llm")
    assert linhas == [
        LinhaAnotacao(vaga_id="v1", curriculo_id="c1", estrato="A", posicao_sistema=1, papel="producao_llm"),
        LinhaAnotacao(vaga_id="v1", curriculo_id="c2", estrato="C", posicao_sistema=None, papel="producao_llm"),
    ]
    assert all(l.nota_humano_1 is None and l.nota_llm_1 is None for l in linhas)


def test_salvar_planilha_anotacao_escreve_csv_legivel(tmp_path):
    linhas = gerar_linhas_anotacao(
        "v1", [ItemAmostrado(curriculo_id="c1", estrato="A", posicao_sistema=1)], papel="producao_llm"
    )
    caminho = tmp_path / "sub" / "anotacoes.csv"
    salvar_planilha_anotacao(linhas, caminho)
    with caminho.open(encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == CAMPOS_PLANILHA_ANOTACAO
        linha = next(leitor)
    assert linha["vaga_id"] == "v1"
    assert linha["curriculo_id"] == "c1"
    assert linha["nota_humano_1"] == ""  # None vira string vazia no CSV


# --- separar_amostra_calibracao ------------------------------------------------------


def _itens_calibracao(vaga_id: str, n_a: int, n_b: int, n_c: int) -> list[ItemAmostrado]:
    itens = [ItemAmostrado(curriculo_id=f"{vaga_id}_a{i}", estrato="A", posicao_sistema=i) for i in range(n_a)]
    itens += [ItemAmostrado(curriculo_id=f"{vaga_id}_b{i}", estrato="B", posicao_sistema=100 + i) for i in range(n_b)]
    itens += [ItemAmostrado(curriculo_id=f"{vaga_id}_c{i}", estrato="C", posicao_sistema=None) for i in range(n_c)]
    return itens


def test_separar_amostra_calibracao_respeita_minimo_por_substrato_e_tamanho_do_grupo():
    # e_calib=0.3 -> n_calibracao_por_grupo=11 (mais rápido de montar que o default 97)
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)
    assert cfg.n_calibracao_por_grupo == 11

    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=3, n_b=3, n_c=6),
        "v2": _itens_calibracao("v2", n_a=3, n_b=3, n_c=6),
        "v3": _itens_calibracao("v3", n_a=10, n_b=10, n_c=10),  # não é vaga de calibração
    }
    linhas_ab, linhas_c = separar_amostra_calibracao(itens_por_vaga, ["v1", "v2"], cfg, seed=1)

    assert len(linhas_ab) == 11
    assert len(linhas_c) == 11
    assert sum(1 for l in linhas_ab if l.estrato == "A") >= 3
    assert sum(1 for l in linhas_ab if l.estrato == "B") >= 3
    assert all(l.estrato == "C" for l in linhas_c)
    assert all(l.papel == "calibracao" for l in linhas_ab + linhas_c)
    assert all(l.vaga_id in ("v1", "v2") for l in linhas_ab + linhas_c)  # v3 fica de fora


def test_separar_amostra_calibracao_e_determinada_pela_seed():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)
    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=3, n_b=3, n_c=6),
        "v2": _itens_calibracao("v2", n_a=3, n_b=3, n_c=6),
    }
    x = separar_amostra_calibracao(itens_por_vaga, ["v1", "v2"], cfg, seed=5)
    y = separar_amostra_calibracao(itens_por_vaga, ["v1", "v2"], cfg, seed=5)
    assert x == y


def test_separar_amostra_calibracao_com_censo_inclui_estrato_a_inteiro_da_vaga():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)  # grupo de 11
    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=5, n_b=3, n_c=6),
        "v2": _itens_calibracao("v2", n_a=5, n_b=3, n_c=6),
    }
    for seed in range(20):
        linhas_ab, linhas_c = separar_amostra_calibracao(
            itens_por_vaga, ["v1", "v2"], cfg, seed=seed, vaga_censo_a="v2"
        )
        a_v2 = {l.curriculo_id for l in linhas_ab if l.vaga_id == "v2" and l.estrato == "A"}
        assert a_v2 == {f"v2_a{i}" for i in range(5)}
        assert len(linhas_ab) == 11 and len(linhas_c) == 11
        assert sum(1 for l in linhas_ab if l.estrato == "B") >= 3
        assert len({(l.vaga_id, l.curriculo_id) for l in linhas_ab}) == 11  # sem repetição


def test_separar_amostra_calibracao_censo_que_nao_cabe_no_grupo_e_erro():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)  # grupo de 11: 9 de A + 3 de B não cabe
    itens_por_vaga = {"v1": _itens_calibracao("v1", n_a=9, n_b=3, n_c=11)}
    with pytest.raises(ValueError, match="censo"):
        separar_amostra_calibracao(itens_por_vaga, ["v1"], cfg, seed=1, vaga_censo_a="v1")


def test_separar_amostra_calibracao_censo_em_vaga_que_nao_e_de_calibracao_e_erro():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)
    itens_por_vaga = {"v1": _itens_calibracao("v1", n_a=6, n_b=6, n_c=11)}
    with pytest.raises(ValueError, match="não é vaga de calibração"):
        separar_amostra_calibracao(itens_por_vaga, ["v1"], cfg, seed=1, vaga_censo_a="v9")


def test_separar_amostra_calibracao_independe_da_ordem_das_vagas():
    # Regressão: a função recebia um set e o percorria na ordem de iteração, que
    # muda entre processos — a mesma seed dava amostras diferentes.
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)
    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=5, n_b=3, n_c=6),
        "v2": _itens_calibracao("v2", n_a=5, n_b=3, n_c=6),
    }
    x = separar_amostra_calibracao(itens_por_vaga, ["v1", "v2"], cfg, seed=5, vaga_censo_a="v1")
    y = separar_amostra_calibracao(itens_por_vaga, ["v2", "v1"], cfg, seed=5, vaga_censo_a="v1")
    assert x == y


def test_completar_censo_estrato_a_acrescenta_so_os_itens_de_a_que_faltavam():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)  # grupo de 11
    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=5, n_b=3, n_c=6),
        "v2": _itens_calibracao("v2", n_a=5, n_b=3, n_c=6),
        "v3": _itens_calibracao("v3", n_a=5, n_b=3, n_c=6),  # não é vaga de calibração
    }
    linhas_ab, _ = separar_amostra_calibracao(itens_por_vaga, ["v1", "v2"], cfg, seed=3)
    completo = completar_censo_estrato_a(linhas_ab, itens_por_vaga, ["v1", "v2"])

    assert completo[: len(linhas_ab)] == linhas_ab  # o que já estava não muda
    chaves = [(l.vaga_id, l.curriculo_id) for l in completo]
    assert len(chaves) == len(set(chaves))
    for v in ("v1", "v2"):
        a = {l.curriculo_id for l in completo if l.vaga_id == v and l.estrato == "A"}
        assert a == {f"{v}_a{i}" for i in range(5)}
    assert all(l.vaga_id != "v3" for l in completo)
    novos = completo[len(linhas_ab):]
    assert all(l.estrato == "A" and l.papel == "calibracao" for l in novos)
    assert completar_censo_estrato_a(completo, itens_por_vaga, ["v1", "v2"]) == completo


def test_escolher_vaga_censo_a_e_deterministica_e_independe_da_ordem():
    assert escolher_vaga_censo_a(["v1", "v2", "v3"], 7) == escolher_vaga_censo_a({"v3", "v1", "v2"}, 7)
    assert escolher_vaga_censo_a(["v1", "v2", "v3"], 7) in {"v1", "v2", "v3"}


def test_separar_amostra_calibracao_levanta_cobertura_insuficiente():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=3)  # precisa de 3 no estrato A
    itens_por_vaga = {
        "v1": _itens_calibracao("v1", n_a=1, n_b=10, n_c=10),  # só 1 item em A
    }
    with pytest.raises(CoberturaInsuficiente) as exc:
        separar_amostra_calibracao(itens_por_vaga, ["v1"], cfg, seed=1)
    assert any("estrato A" in b for b in exc.value.checagem["bloqueios"])


def test_separar_amostra_calibracao_erro_de_config_quando_minimo_excede_o_grupo():
    cfg = _cfg(e_calib=0.3, min_substrato_calibracao=10)  # 2*10=20 > n_calibracao_por_grupo=11
    with pytest.raises(ValueError, match="configuração inconsistente"):
        separar_amostra_calibracao({}, ["v1"], cfg, seed=1)


# --- log de reprodutibilidade -----------------------------------------------------


def test_salvar_log_reprodutibilidade_escreve_csv_legivel(tmp_path):
    linhas = [
        LogReprodutibilidade(
            vaga_id="v1",
            seed_estrato_b=1,
            seed_estrato_c=2,
            seed_embaralhamento=3,
            data_coleta="2026-09-15",
            k_usado=25,
            n_total_usado=4817,
            is_calibracao=True,
        )
    ]
    caminho = tmp_path / "sub" / "log_reprodutibilidade.csv"
    salvar_log_reprodutibilidade(linhas, caminho)
    with caminho.open(encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == CAMPOS_LOG_REPRODUTIBILIDADE
        linha = next(leitor)
    assert linha == {
        "vaga_id": "v1",
        "seed_estrato_b": "1",
        "seed_estrato_c": "2",
        "seed_embaralhamento": "3",
        "data_coleta": "2026-09-15",
        "k_usado": "25",
        "n_total_usado": "4817",
        "is_calibracao": "True",
    }
