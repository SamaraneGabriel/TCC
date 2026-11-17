"""Juiz LLM item a item (src/evaluation/plano/juiz.py).

Sem chamada de API: o LLM é um `BaseExtractor` falso que devolve, em sequência, as
respostas que o teste quiser. Verifica-se o contrato em volta do modelo — o que
entra no prompt (e o que não pode entrar), validação da resposta contra a
rubrica, novas tentativas e persistência com retomada.
"""
import asyncio
import json

import pytest

from src.evaluation.plano.juiz import (
    MAX_TENTATIVAS,
    PROMPT_JUIZ,
    JuizError,
    JuizRubrica,
    carregar_julgamentos,
    gravar_julgamento,
    hash_prompt,
)
from src.evaluation.plano.rubrica import ItemQualificacao, RubricaVaga
from src.extraction.base import BaseExtractor


class LLMFalso(BaseExtractor):
    def __init__(self, *respostas: str, model: str = "modelo-de-teste"):
        self.respostas = list(respostas)
        self.model = model
        self.prompts: list[str] = []

    def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.respostas[min(len(self.prompts), len(self.respostas)) - 1]


VAGA = {
    "vaga_id": "vaga_1",
    "cargo": "Senior Backend Engineer",
    "area_tecnica": "Backend",
    "senioridade": "senior",
    "texto": "TEXTO COMPLETO DO ANUNCIO",
}


def rubrica(com_desejavel: bool = True) -> RubricaVaga:
    itens = [
        ItemQualificacao(item_id="vaga_1#01", descricao="Python", tipo="obrigatoria",
                         categoria="ferramenta_tecnologia"),
        ItemQualificacao(item_id="vaga_1#02", descricao="5 anos de backend", tipo="obrigatoria",
                         categoria="experiencia_relacionada"),
    ]
    if com_desejavel:
        itens.append(ItemQualificacao(item_id="vaga_1#03", descricao="Kubernetes",
                                      tipo="desejavel", categoria="ferramenta_tecnologia"))
    return RubricaVaga(vaga_id="vaga_1", itens=itens, origem="humano_revisado",
                       data_geracao="2026-11-10T00:00:00")


def resposta(nota=2, atend=("atende", "atende", "nao_atende"), ids=None, justificativa="ok") -> str:
    ids = ids or ["vaga_1#01", "vaga_1#02", "vaga_1#03"][: len(atend)]
    return json.dumps({
        "itens": [{"item_id": i, "evidencia": "", "atendimento": a} for i, a in zip(ids, atend)],
        "justificativa": justificativa,
        "nota": nota,
    })


# --- prompt -------------------------------------------------------------------


def test_prompt_traz_rubrica_curriculo_e_contexto_da_vaga():
    juiz = JuizRubrica(LLMFalso(resposta()))
    p = juiz.montar_prompt(VAGA, rubrica(), "CURRICULO XYZ")
    assert "Senior Backend Engineer" in p and "Backend" in p and "senior" in p
    assert "- vaga_1#01 [ferramenta_tecnologia] Python" in p
    assert "- vaga_1#03 [ferramenta_tecnologia] Kubernetes" in p
    assert "<curriculo>\nCURRICULO XYZ\n</curriculo>" in p
    assert "{" + "curriculo" + "}" not in p and "{itens_obrigatorios}" not in p


def test_prompt_nao_vaza_ranking_nem_texto_do_anuncio():
    juiz = JuizRubrica(LLMFalso(resposta()))
    p = juiz.montar_prompt({**VAGA, "posicao": 1, "score": 0.99}, rubrica(), "cv")
    for proibido in ("posicao", "score", "estrato", "0.99", "TEXTO COMPLETO DO ANUNCIO"):
        assert proibido not in p


def test_rubrica_sem_desejaveis_aparece_como_nenhum():
    p = JuizRubrica(LLMFalso(resposta())).montar_prompt(VAGA, rubrica(False), "cv")
    assert "RUBRICA — itens desejáveis\n(nenhum)" in p


def test_chaves_dentro_do_curriculo_nao_sao_substituidas():
    p = JuizRubrica(LLMFalso(resposta())).montar_prompt(VAGA, rubrica(), "texto com {cargo} literal")
    assert "texto com {cargo} literal" in p


def test_prompt_e_o_arquivo_congelado_sem_alteracao():
    juiz = JuizRubrica(LLMFalso(resposta()))
    assert juiz.prompt_sha256 == hash_prompt(PROMPT_JUIZ.read_text(encoding="utf-8"))


# --- julgamento ---------------------------------------------------------------


def test_julgamento_valido_na_primeira_tentativa():
    llm = LLMFalso(resposta(nota=2))
    j = JuizRubrica(llm).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert j.nota == 2 and j.tentativas == 1 and len(llm.prompts) == 1
    assert j.vaga_id == "vaga_1" and j.curriculo_id == "cv_1"
    assert j.modelo == "modelo-de-teste"
    assert j.atendimentos() == {"vaga_1#01": "atende", "vaga_1#02": "atende", "vaga_1#03": "nao_atende"}


def test_nota_e_do_juiz_e_a_sugerida_fica_ao_lado():
    # Checklist completo nos obrigatórios + desejável atendido sugeriria 3; o juiz deu 2.
    j = JuizRubrica(LLMFalso(resposta(nota=2, atend=("atende", "atende", "atende")))).julgar(
        VAGA, rubrica(), "cv_1", "cv")
    assert j.nota == 2
    assert j.nota_sugerida_checklist == 3


def test_itens_voltam_na_ordem_da_rubrica():
    bruto = resposta(atend=("parcial", "atende", "nao_atende"), ids=["vaga_1#03", "vaga_1#01", "vaga_1#02"])
    j = JuizRubrica(LLMFalso(bruto)).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert [i.item_id for i in j.itens] == ["vaga_1#01", "vaga_1#02", "vaga_1#03"]
    assert j.atendimentos()["vaga_1#03"] == "parcial"


def test_grafia_nao_atendimento_vira_nao_atende():
    llm = LLMFalso(resposta(atend=("atende", "nao_atendimento", "nao_atende")))
    j = JuizRubrica(llm).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert j.tentativas == 1 and j.atendimentos()["vaga_1#02"] == "nao_atende"


def test_nota_em_string_e_aceita():
    j = JuizRubrica(LLMFalso(resposta(nota="1"))).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert j.nota == 1


@pytest.mark.parametrize("ruim", [
    "isto não é json",
    resposta(nota=4),
    resposta(atend=("atende", "atende")),                                   # item faltando
    resposta(ids=["vaga_1#01", "vaga_1#02", "vaga_1#99"]),                  # item fora da rubrica
    resposta(ids=["vaga_1#01", "vaga_1#01", "vaga_1#02"]),                  # item repetido
    resposta(atend=("atende", "talvez", "atende")),                          # atendimento inválido
])
def test_resposta_invalida_gera_nova_tentativa(ruim):
    llm = LLMFalso(ruim, resposta(nota=1))
    j = JuizRubrica(llm).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert j.nota == 1 and j.tentativas == 2 and len(llm.prompts) == 2


def test_desiste_depois_de_max_tentativas():
    llm = LLMFalso("lixo")
    with pytest.raises(JuizError):
        JuizRubrica(llm).julgar(VAGA, rubrica(), "cv_1", "cv")
    assert len(llm.prompts) == MAX_TENTATIVAS


def test_markdown_em_volta_do_json_e_tolerado():
    j = JuizRubrica(LLMFalso("```json\n" + resposta(nota=3, atend=("atende",) * 3) + "\n```")).julgar(
        VAGA, rubrica(), "cv_1", "cv")
    assert j.nota == 3


def test_versao_assincrona():
    llm = LLMFalso("lixo", resposta(nota=0, atend=("nao_atende",) * 3))
    j = asyncio.run(JuizRubrica(llm).ajulgar(VAGA, rubrica(), "cv_1", "cv"))
    assert j.nota == 0 and j.tentativas == 2


# --- persistência -------------------------------------------------------------


def test_gravar_e_carregar_ida_e_volta(tmp_path):
    arq = tmp_path / "julgamentos.jsonl"
    juiz = JuizRubrica(LLMFalso(resposta(nota=2)))
    a = juiz.julgar(VAGA, rubrica(), "cv_1", "cv")
    b = juiz.julgar(VAGA, rubrica(), "cv_2", "cv")
    gravar_julgamento(arq, a)
    gravar_julgamento(arq, b)
    lidos = carregar_julgamentos(arq)
    assert set(lidos) == {("vaga_1", "cv_1"), ("vaga_1", "cv_2")}
    assert lidos[("vaga_1", "cv_1")] == a


def test_linha_corrompida_e_ignorada_para_o_par_ser_refeito(tmp_path):
    arq = tmp_path / "julgamentos.jsonl"
    gravar_julgamento(arq, JuizRubrica(LLMFalso(resposta())).julgar(VAGA, rubrica(), "cv_1", "cv"))
    with arq.open("a", encoding="utf-8") as fh:
        fh.write('{"vaga_id": "vaga_1", "curriculo_id": "cv_2", "no')  # gravação interrompida
    assert set(carregar_julgamentos(arq)) == {("vaga_1", "cv_1")}


def test_arquivo_inexistente_e_vazio(tmp_path):
    assert carregar_julgamentos(tmp_path / "nao_existe.jsonl") == {}
