"""Lógica da anotação humana da calibração (§4.3) — sem Streamlit.

O foco é nas três invariantes do protocolo: cegueira do anotador, independência
entre os dois anotadores e durabilidade do que já foi anotado.
"""
import csv

import pytest

from src.evaluation.plano.amostragem import CAMPOS_PLANILHA_ANOTACAO
from src.evaluation.plano.anotacao import (
    ATENDIMENTOS,
    ItemParaAnotar,
    carregar_checklist,
    carregar_planilha,
    checklist_do_item,
    chave,
    indexar_curriculos,
    item_para_anotar,
    itens_do_anotador,
    nota_de,
    nota_sugerida,
    ordem_do_anotador,
    progresso,
    proximo_pendente,
    registrar_checklist,
    registrar_nota,
    salvar_checklist,
    salvar_planilha,
    selecionar_double_label,
    texto_do_curriculo,
)
from src.evaluation.plano.config import ConfigAvaliacao
from src.evaluation.plano.rubrica import ItemQualificacao, RubricaVaga, gerar_item_id

SEED = 20260915


@pytest.fixture
def cfg():
    return ConfigAvaliacao(_env_file=None)


def _linha(vaga_id: str, curriculo_id: str, estrato: str = "A", posicao: int | None = 1) -> dict:
    return {
        "vaga_id": vaga_id,
        "curriculo_id": curriculo_id,
        "estrato": estrato,
        "posicao_sistema": str(posicao) if posicao else "",
        "papel": "calibracao",
        "nota_humano_1": "",
        "nota_humano_2": "",
        "nota_llm_1": "",
        "nota_llm_2": "",
        "fonte_nota_final": "",
        "observacoes": "",
    }


@pytest.fixture
def linhas():
    """40 itens em 2 vagas, estratos misturados — como sai de montar_amostras.py."""
    return [
        _linha("v1" if i < 20 else "v2", f"cur_{i:03d}", "A" if i % 3 else "C", i + 1)
        for i in range(40)
    ]


@pytest.fixture
def rubrica():
    itens = [
        ("Python em produção", "obrigatoria", "ferramenta_tecnologia"),
        ("5 anos em backend", "obrigatoria", "experiencia_relacionada"),
        ("Kubernetes", "desejavel", "ferramenta_tecnologia"),
    ]
    return RubricaVaga(
        vaga_id="v1",
        itens=[
            ItemQualificacao(item_id=gerar_item_id("v1", n), descricao=d, tipo=t, categoria=c)
            for n, (d, t, c) in enumerate(itens, start=1)
        ],
        data_geracao="2026-09-22T10:00:00",
    )


# --- cegueira ---------------------------------------------------------------


def test_item_para_anotar_nao_expoe_estrato_posicao_nem_nota_do_llm():
    """§4.3 passo 2: o anotador é cego quanto a estrato, posição e à nota do LLM,
    mesmo quando ela já existe. Este é o filtro que a interface usa."""
    linha = _linha("v1", "cur_1", estrato="A", posicao=3)
    linha["nota_llm_1"] = "3"
    item = item_para_anotar(linha, "humano_1")

    assert isinstance(item, ItemParaAnotar)
    campos = set(vars(item))
    assert campos == {"vaga_id", "curriculo_id", "nota_atual", "observacoes"}
    assert "3" not in str(vars(item).values()) or item.nota_atual is None


def test_item_para_anotar_mostra_a_propria_nota_do_anotador():
    """Rever o que você mesmo respondeu é permitido — e necessário para corrigir."""
    linha = _linha("v1", "cur_1")
    linha["nota_humano_1"] = "2"
    assert item_para_anotar(linha, "humano_1").nota_atual == 2
    assert item_para_anotar(linha, "humano_2").nota_atual is None


# --- ordem e independência entre anotadores ---------------------------------


def test_ordem_e_deterministica(linhas):
    assert [chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED)] == [
        chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED)
    ]


def test_cada_anotador_recebe_ordem_diferente(linhas):
    """Se os dois percorressem a mesma sequência, fadiga e ordem ficariam
    correlacionadas entre eles e o kappa humano-humano herdaria esse efeito."""
    a = [chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED)]
    b = [chave(l) for l in ordem_do_anotador(linhas, "humano_2", SEED)]
    assert a != b
    assert set(a) == set(b)


def test_ordem_nao_segue_a_do_arquivo(linhas):
    """O CSV vem agrupado (AB e depois C), então usar a ordem do arquivo deixaria
    resíduo de estrato na sequência exibida."""
    assert [chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED)] != [
        chave(l) for l in linhas
    ]


def test_seed_mestre_diferente_muda_a_ordem(linhas):
    assert [chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED)] != [
        chave(l) for l in ordem_do_anotador(linhas, "humano_1", SEED + 1)
    ]


# --- double-label -----------------------------------------------------------


def test_double_label_respeita_a_fracao(linhas, cfg):
    alvo = selecionar_double_label(linhas, cfg, SEED)
    assert len(alvo) == min(cfg.n_double_label, len(linhas))
    assert alvo <= {chave(l) for l in linhas}


def test_double_label_e_estavel_entre_execucoes(linhas, cfg):
    """O segundo anotador começa noutro dia, possivelmente noutro processo."""
    assert selecionar_double_label(linhas, cfg, SEED) == selecionar_double_label(linhas, cfg, SEED)


def test_segundo_anotador_so_recebe_os_sorteados(linhas, cfg):
    alvo = selecionar_double_label(linhas, cfg, SEED)
    do_segundo = itens_do_anotador(linhas, "humano_2", cfg, SEED)
    assert {chave(l) for l in do_segundo} == alvo
    assert len(itens_do_anotador(linhas, "humano_1", cfg, SEED)) == len(linhas)


# --- gravação ---------------------------------------------------------------


def test_registrar_nota_marca_fonte_humano(linhas):
    linha = registrar_nota(linhas, "v1", "cur_000", "humano_1", 2)
    assert linha["nota_humano_1"] == "2"
    assert linha["fonte_nota_final"] == "humano"


def test_registrar_nota_de_um_anotador_nao_toca_no_outro(linhas):
    registrar_nota(linhas, "v1", "cur_000", "humano_1", 3)
    registrar_nota(linhas, "v1", "cur_000", "humano_2", 1)
    linha = next(l for l in linhas if chave(l) == ("v1", "cur_000"))
    assert (linha["nota_humano_1"], linha["nota_humano_2"]) == ("3", "1")


def test_nota_fora_da_escala_e_rejeitada(linhas):
    with pytest.raises(ValueError, match="escala"):
        registrar_nota(linhas, "v1", "cur_000", "humano_1", 4)


def test_par_inexistente_e_rejeitado(linhas):
    with pytest.raises(KeyError):
        registrar_nota(linhas, "v9", "cur_999", "humano_1", 1)


def test_ida_e_volta_preserva_notas_e_colunas(linhas, tmp_path):
    registrar_nota(linhas, "v1", "cur_000", "humano_1", 2, observacoes="currículo truncado")
    caminho = tmp_path / "anotacoes.csv"
    salvar_planilha(linhas, caminho)
    lidas = carregar_planilha(caminho)

    assert list(csv.DictReader(caminho.open(encoding="utf-8")).fieldnames) == CAMPOS_PLANILHA_ANOTACAO
    linha = next(l for l in lidas if chave(l) == ("v1", "cur_000"))
    assert nota_de(linha, "humano_1") == 2
    assert linha["observacoes"] == "currículo truncado"


def test_planilha_inexistente_devolve_vazio(tmp_path):
    assert carregar_planilha(tmp_path / "nao-existe.csv") == []


# --- checklist (§7.4) -------------------------------------------------------


def test_checklist_ida_e_volta(tmp_path, rubrica):
    registros = {}
    registrar_checklist(registros, "v1", "cur_1", "humano_1",
                        {"v1#01": "atende", "v1#02": "parcial", "v1#03": "nao_atende"})
    caminho = tmp_path / "checklist.csv"
    salvar_checklist(registros, caminho)

    lidos = carregar_checklist(caminho)
    assert len(lidos) == 3
    assert checklist_do_item(lidos, "cur_1", "humano_1", rubrica) == {
        "v1#01": "atende", "v1#02": "parcial", "v1#03": "nao_atende"
    }


def test_checklist_separa_por_anotador(rubrica):
    registros = {}
    registrar_checklist(registros, "v1", "cur_1", "humano_1", {"v1#01": "atende"})
    registrar_checklist(registros, "v1", "cur_1", "humano_2", {"v1#01": "nao_atende"})
    assert checklist_do_item(registros, "cur_1", "humano_1", rubrica) == {"v1#01": "atende"}
    assert checklist_do_item(registros, "cur_1", "humano_2", rubrica) == {"v1#01": "nao_atende"}


def test_checklist_regravado_sobrescreve(rubrica):
    registros = {}
    registrar_checklist(registros, "v1", "cur_1", "humano_1", {"v1#01": "atende"})
    registrar_checklist(registros, "v1", "cur_1", "humano_1", {"v1#01": "nao_atende"})
    assert len(registros) == 1
    assert checklist_do_item(registros, "cur_1", "humano_1", rubrica) == {"v1#01": "nao_atende"}


def test_atendimento_invalido_e_rejeitado():
    with pytest.raises(ValueError, match="atendimento"):
        registrar_checklist({}, "v1", "cur_1", "humano_1", {"v1#01": "talvez"})


# --- nota sugerida ----------------------------------------------------------


def test_sugestao_exige_checklist_completo(rubrica):
    assert nota_sugerida(rubrica, {"v1#01": "atende"}) is None


def test_sugestao_3_com_obrigatorios_e_desejavel(rubrica):
    assert nota_sugerida(rubrica, {"v1#01": "atende", "v1#02": "atende", "v1#03": "atende"}) == 3


def test_sugestao_2_com_obrigatorios_sem_desejavel(rubrica):
    """§6: nota 3 exige "também atende itens desejáveis"."""
    assert nota_sugerida(rubrica, {"v1#01": "atende", "v1#02": "atende", "v1#03": "nao_atende"}) == 2


def test_sugestao_1_quando_falta_obrigatorio(rubrica):
    assert nota_sugerida(rubrica, {"v1#01": "atende", "v1#02": "nao_atende", "v1#03": "atende"}) == 1


def test_sugestao_0_quando_nada_e_atendido(rubrica):
    assert nota_sugerida(
        rubrica, {"v1#01": "nao_atende", "v1#02": "nao_atende", "v1#03": "nao_atende"}
    ) == 0


def test_sugestao_e_none_sem_obrigatorios():
    """Rubrica sem obrigatórios não sustenta a escala (ver RubricaVaga.problemas)."""
    r = RubricaVaga(
        vaga_id="v1",
        itens=[ItemQualificacao(item_id="v1#01", descricao="x", tipo="desejavel", categoria="outro")],
        data_geracao="2026-09-22T10:00:00",
    )
    assert nota_sugerida(r, {"v1#01": "atende"}) is None


# --- progresso --------------------------------------------------------------


def test_progresso_conta_so_o_alvo_do_anotador(linhas, cfg):
    registrar_nota(linhas, "v1", "cur_000", "humano_1", 2)
    p = progresso(linhas, "humano_1", cfg, SEED)
    assert (p["total"], p["feitos"], p["faltam"]) == (40, 1, 39)
    assert p["distribuicao"][2] == 1


def test_proximo_pendente_pula_o_que_ja_foi_feito(linhas, cfg):
    ordem = itens_do_anotador(linhas, "humano_1", cfg, SEED)
    assert proximo_pendente(linhas, "humano_1", cfg, SEED) == 0
    registrar_nota(linhas, *chave(ordem[0]), "humano_1", 1)
    assert proximo_pendente(linhas, "humano_1", cfg, SEED) == 1


def test_proximo_pendente_none_quando_tudo_anotado(linhas, cfg):
    for l in linhas:
        registrar_nota(linhas, *chave(l), "humano_1", 0)
    assert proximo_pendente(linhas, "humano_1", cfg, SEED) is None


# --- texto do currículo -----------------------------------------------------


def test_texto_do_curriculo_resolve_pelo_arquivo_fonte(tmp_path):
    d = tmp_path / "curriculos"
    d.mkdir()
    (d / "curriculo_0001.txt").write_text("Engenheiro de software, Python.", encoding="utf-8")
    indice = indexar_curriculos([d])
    assert texto_do_curriculo({"arquivo_fonte": "curriculo_0001.txt"}, indice).startswith("Engenheiro")


def test_texto_ausente_devolve_vazio_em_vez_de_explodir(tmp_path):
    """Currículo cujo arquivo sumiu não pode derrubar a sessão de anotação —
    a interface cai para os campos extraídos e avisa."""
    indice = indexar_curriculos([tmp_path])
    assert texto_do_curriculo({"arquivo_fonte": "sumiu.txt"}, indice) == ""
    assert texto_do_curriculo({}, indice) == ""


def test_atendimentos_sao_os_tres_do_desvio_d4():
    assert set(ATENDIMENTOS) == {"atende", "parcial", "nao_atende"}
