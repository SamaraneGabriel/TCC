"""Rubrica de qualificação por vaga (seção 3 do plano de avaliação).

Sem chamada de API: o LLM é um `BaseExtractor` falso que devolve o JSON que o
teste quiser. O que se verifica aqui é o contrato em volta do modelo — identidade
dos itens, validação da escala, persistência ida e volta, e as checagens que
impedem uma rubrica inutilizável de chegar na anotação.
"""
import json

import pytest

from src.evaluation.plano.onet import VocabularioONet
from src.evaluation.plano.rubrica import (
    CAMPOS_RUBRICA,
    GeradorRubrica,
    ItemQualificacao,
    RubricaVaga,
    carregar_rubricas,
    conferir_rubricas,
    gerar_item_id,
    remover_itens,
    salvar_rubricas,
)
from src.evaluation.plano.selecao import VagaPool
from src.extraction.base import BaseExtractor
from src.extraction.parsing import ExtractionError


class LLMFalso(BaseExtractor):
    def __init__(self, resposta: str, model: str = "modelo-de-teste"):
        self.resposta = resposta
        self.model = model
        self.prompts: list[str] = []

    def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.resposta


def resposta(*itens: tuple[str, str, str]) -> str:
    return json.dumps(
        {"itens": [{"descricao": d, "tipo": t, "categoria": c} for d, t, c in itens]}
    )


RESPOSTA_PADRAO = resposta(
    ("experiência com Python em produção", "obrigatoria", "ferramenta_tecnologia"),
    ("mínimo 5 anos em desenvolvimento backend", "obrigatoria", "experiencia_relacionada"),
    ("modelagem de dados relacionais", "obrigatoria", "habilidade_tecnica"),
    ("experiência com Kubernetes ou orquestração equivalente", "desejavel", "ferramenta_tecnologia"),
    ("graduação em Ciência da Computação ou correlata", "desejavel", "formacao"),
)


@pytest.fixture
def vaga():
    return VagaPool(
        vaga_id="vaga_123",
        cargo="Senior Backend Engineer",
        texto="Buscamos engenheiro backend sênior.\nRequisitos: Python, 5 anos, SQL.",
        area_tecnica="Backend",
        senioridade="senior",
    )


@pytest.fixture
def gerador():
    return GeradorRubrica(LLMFalso(RESPOSTA_PADRAO))


# --- geração ----------------------------------------------------------------


def test_gera_rubrica_com_itens_tipados(gerador, vaga):
    r = gerador.gerar(vaga)
    assert r.vaga_id == "vaga_123"
    assert len(r.itens) == 5
    assert len(r.obrigatorios) == 3
    assert len(r.desejaveis) == 2
    assert r.origem == "llm"
    assert r.modelo == "modelo-de-teste"


def test_item_id_e_atribuido_pelo_codigo_nao_pelo_llm(gerador, vaga):
    """O LLM não devolve item_id — ele é a chave que liga rubrica e checklist
    (§7.4), então precisa ser determinístico e não inventado a cada chamada."""
    r = gerador.gerar(vaga)
    assert [i.item_id for i in r.itens] == [f"vaga_123#{n:02d}" for n in range(1, 6)]
    assert gerar_item_id("vaga_9", 7) == "vaga_9#07"


def test_gerar_duas_vezes_produz_os_mesmos_ids(gerador, vaga):
    assert [i.item_id for i in gerador.gerar(vaga).itens] == [
        i.item_id for i in gerador.gerar(vaga).itens
    ]


def test_categoria_invalida_e_rejeitada(vaga):
    g = GeradorRubrica(LLMFalso(resposta(("x", "obrigatoria", "categoria_inventada"))))
    with pytest.raises(ExtractionError, match="Schema validation failed"):
        g.gerar(vaga)


def test_tipo_invalido_e_rejeitado(vaga):
    g = GeradorRubrica(LLMFalso(resposta(("x", "talvez", "conhecimento"))))
    with pytest.raises(ExtractionError):
        g.gerar(vaga)


def test_json_quebrado_vira_erro_de_extracao(vaga):
    with pytest.raises(ExtractionError, match="invalid JSON"):
        GeradorRubrica(LLMFalso("não é json")).gerar(vaga)


def test_resposta_com_cercas_de_markdown_funciona(vaga):
    """Modelos às vezes embrulham em ```json apesar da instrução."""
    g = GeradorRubrica(LLMFalso(f"```json\n{RESPOSTA_PADRAO}\n```"))
    assert len(g.gerar(vaga).itens) == 5


# --- prompt -----------------------------------------------------------------


def test_prompt_leva_a_vaga_e_nao_deixa_placeholder(gerador, vaga):
    gerador.gerar(vaga)
    prompt = gerador._llm.prompts[0]
    assert vaga.texto in prompt
    assert "Senior Backend Engineer" in prompt
    for marcador in ("{texto}", "{cargo}", "{area}", "{senioridade}",
                     "{tecnologias}", "{ocupacoes}", "{categorias_onet}"):
        assert marcador not in prompt


def test_prompt_sem_vocabulario_diz_que_esta_indisponivel(gerador, vaga):
    gerador.gerar(vaga)
    assert "vocabulário O*NET indisponível" in gerador._llm.prompts[0]


def test_prompt_com_vocabulario_injeta_termos_da_area(vaga, tmp_path, monkeypatch):
    vocab = VocabularioONet(
        tecnologias_por_soc={
            "15-1252.00": [
                type("T", (), {"nome": "Python", "categoria": "Dev software", "soc": "15-1252.00",
                               "hot": True, "em_demanda": True})(),
                type("T", (), {"nome": "Kubernetes", "categoria": "Cloud software",
                               "soc": "15-1252.00", "hot": True, "em_demanda": False})(),
            ]
        },
        titulos={"15-1252.00": "Software Developers"},
    )
    g = GeradorRubrica(LLMFalso(RESPOSTA_PADRAO), vocabulario=vocab)
    g.gerar(vaga)
    prompt = g._llm.prompts[0]
    assert "Python" in prompt and "Kubernetes" in prompt
    assert "15-1252.00 Software Developers" in prompt
    # O vocabulário não pode virar fonte de requisito — a instrução tem que estar lá.
    assert "NÃO é fonte de requisito" in prompt


def test_ocupacoes_onet_ficam_registradas_na_rubrica(vaga):
    vocab = VocabularioONet(
        tecnologias_por_soc={"15-1252.00": []}, titulos={"15-1252.00": "Software Developers"}
    )
    g = GeradorRubrica(LLMFalso(RESPOSTA_PADRAO), vocabulario=vocab)
    assert "15-1252.00" in g.gerar(vaga).ocupacoes_onet


# --- validação da rubrica ---------------------------------------------------


def _rubrica(*itens: tuple[str, str, str], vaga_id: str = "v1", origem: str = "llm") -> RubricaVaga:
    return RubricaVaga(
        vaga_id=vaga_id,
        itens=[
            ItemQualificacao(item_id=gerar_item_id(vaga_id, n), descricao=d, tipo=t, categoria=c)
            for n, (d, t, c) in enumerate(itens, start=1)
        ],
        origem=origem,
        data_geracao="2026-09-22T10:00:00",
    )


def test_rubrica_valida_nao_tem_problemas(gerador, vaga):
    assert gerador.gerar(vaga).problemas() == []


def test_rubrica_sem_obrigatorios_e_bloqueio():
    """Sem obrigatórios, "atende todos os obrigatórios" é vacuamente verdadeiro
    e as notas 2 e 3 da §6 param de discriminar."""
    r = _rubrica(("a", "desejavel", "outro"), ("b", "desejavel", "outro"))
    assert any("nenhum item obrigatório" in p for p in r.problemas())


def test_rubrica_vazia_e_bloqueio():
    assert len(_rubrica().problemas()) >= 1


def test_descricao_repetida_e_bloqueio():
    r = _rubrica(("Python", "obrigatoria", "ferramenta_tecnologia"),
                 ("python", "desejavel", "ferramenta_tecnologia"))
    assert any("descrição repetida" in p for p in r.problemas())


def test_sem_desejaveis_vira_aviso_nao_bloqueio():
    """A nota 3 exige "também atende itens desejáveis" — sem desejáveis ela é
    inalcançável, mas a rubrica continua utilizável para 0-2."""
    r = _rubrica(("a", "obrigatoria", "outro"))
    assert r.problemas() == []
    assert any("nota 3" in a for a in r.avisos())


def test_habilidade_basica_obrigatoria_vira_aviso():
    """§3.1: habilidade_basica é desejável por padrão, salvo exigência explícita."""
    r = _rubrica(("inglês fluente", "obrigatoria", "habilidade_basica"))
    assert any("habilidade_basica" in a for a in r.avisos())


def test_poucos_itens_vira_aviso():
    r = _rubrica(("a", "obrigatoria", "outro"), ("b", "desejavel", "outro"))
    assert any("mínimo sugerido" in a for a in r.avisos())


def test_resumo_categorias_conta_so_o_que_existe(gerador, vaga):
    resumo = gerador.gerar(vaga).resumo_categorias()
    assert resumo["ferramenta_tecnologia"] == 2
    assert "senioridade" not in resumo


# --- persistência (§7.3) ----------------------------------------------------


def test_csv_tem_uma_linha_por_item_no_formato_da_spec(gerador, vaga, tmp_path):
    caminho = salvar_rubricas([gerador.gerar(vaga)], tmp_path / "rubricas_vagas.csv")
    linhas = caminho.read_text(encoding="utf-8").strip().splitlines()
    assert linhas[0].split(",") == CAMPOS_RUBRICA
    assert len(linhas) == 6  # cabeçalho + 5 itens


def test_ida_e_volta_preserva_itens_e_ordem(gerador, vaga, tmp_path):
    original = gerador.gerar(vaga)
    caminho = salvar_rubricas([original], tmp_path / "r.csv")
    lido = carregar_rubricas(caminho)["vaga_123"]
    assert [i.model_dump() for i in lido.itens] == [i.model_dump() for i in original.itens]
    assert lido.origem == original.origem
    assert lido.data_geracao == original.data_geracao


def test_ida_e_volta_preserva_revisao_humana(tmp_path):
    r = _rubrica(("a", "obrigatoria", "outro"), origem="humano_revisado")
    lido = carregar_rubricas(salvar_rubricas([r], tmp_path / "r.csv"))["v1"]
    assert lido.origem == "humano_revisado"


def test_arquivo_inexistente_devolve_vazio(tmp_path):
    assert carregar_rubricas(tmp_path / "nao-existe.csv") == {}


def test_descricao_com_virgula_sobrevive_ao_csv(tmp_path):
    r = _rubrica(("Python, Django ou Flask", "obrigatoria", "ferramenta_tecnologia"))
    lido = carregar_rubricas(salvar_rubricas([r], tmp_path / "r.csv"))["v1"]
    assert lido.itens[0].descricao == "Python, Django ou Flask"


# --- conferência de prontidão -----------------------------------------------


def _vaga(vaga_id: str) -> VagaPool:
    return VagaPool(vaga_id=vaga_id, cargo="Dev", texto="linha um\nlinha dois", area_tecnica="Backend")


def test_conferencia_aprova_quando_tudo_pronto():
    vagas = [_vaga("v1"), _vaga("v2")]
    rubricas = {
        "v1": _rubrica(("a", "obrigatoria", "outro"), ("b", "desejavel", "outro"),
                       ("c", "desejavel", "outro"), vaga_id="v1", origem="humano_revisado"),
        "v2": _rubrica(("a", "obrigatoria", "outro"), ("b", "desejavel", "outro"),
                       ("c", "desejavel", "outro"), vaga_id="v2"),
    }
    r = conferir_rubricas(rubricas, vagas, {"v1"})
    assert r["ok"] and r["bloqueios"] == []
    assert r["rubricas"] == 2 and r["revisadas"] == 1


def test_conferencia_bloqueia_vaga_sem_rubrica():
    r = conferir_rubricas({}, [_vaga("v1")], set())
    assert not r["ok"]
    assert any("sem rubrica" in b for b in r["bloqueios"])


def test_conferencia_bloqueia_calibracao_sem_revisao_humana():
    """§3.2 passo 2: nas vagas de calibração a revisão humana é obrigatória —
    e é a rubrica revisada que os 194 julgamentos humanos vão usar."""
    rubricas = {"v1": _rubrica(("a", "obrigatoria", "outro"), vaga_id="v1", origem="llm")}
    r = conferir_rubricas(rubricas, [_vaga("v1")], {"v1"})
    assert not r["ok"]
    assert any("revisão humana" in b for b in r["bloqueios"])


def test_conferencia_propaga_problema_de_rubrica_invalida():
    rubricas = {"v1": _rubrica(("a", "desejavel", "outro"), vaga_id="v1")}
    r = conferir_rubricas(rubricas, [_vaga("v1")], set())
    assert not r["ok"]
    assert any("nenhum item obrigatório" in b for b in r["bloqueios"])


# ----------------------------------------------------------------------
# Remoção de itens por regra (desvio D9)


def _duas_rubricas() -> dict[str, RubricaVaga]:
    return {
        "v1": _rubrica(("python", "obrigatoria", "ferramenta_tecnologia"),
                       ("visto de trabalho", "obrigatoria", "outro"),
                       ("docker", "desejavel", "ferramenta_tecnologia"), vaga_id="v1"),
        "v2": _rubrica(("java", "obrigatoria", "ferramenta_tecnologia"), vaga_id="v2",
                       origem="humano_revisado"),
    }


def test_remover_itens_tira_so_os_pedidos_e_nao_renumera():
    """Renumerar faria `v1#03` passar a apontar para outro item sem nada acusar."""
    novas = remover_itens(_duas_rubricas(), {"v1#02"})
    assert [i.item_id for i in novas["v1"].itens] == ["v1#01", "v1#03"]
    assert [i.descricao for i in novas["v1"].itens] == ["python", "docker"]
    assert len(novas["v2"].itens) == 1


def test_remover_itens_preserva_origem_e_nao_altera_a_entrada():
    original = _duas_rubricas()
    novas = remover_itens(original, {"v1#02"})
    assert novas["v1"].origem == "llm" and novas["v2"].origem == "humano_revisado"
    assert len(original["v1"].itens) == 3


def test_remover_item_inexistente_e_erro():
    with pytest.raises(ValueError, match="inexistente"):
        remover_itens(_duas_rubricas(), {"v1#99"})


def test_remover_ultimo_obrigatorio_e_erro():
    with pytest.raises(ValueError, match="nenhum item obrigatório"):
        remover_itens(_duas_rubricas(), {"v2#01"})


def test_remocao_sobrevive_ao_csv(tmp_path):
    novas = remover_itens(_duas_rubricas(), {"v1#02"})
    caminho = salvar_rubricas(list(novas.values()), tmp_path / "r.csv")
    lidas = carregar_rubricas(caminho)
    assert [i.item_id for i in lidas["v1"].itens] == ["v1#01", "v1#03"]
