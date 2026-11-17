"""Vocabulário O*NET da rubrica (seção 3.1 do plano, desvio D3).

Os testes rodam sobre uma base falsa escrita em tmp_path — não dependem dos CSVs
reais estarem baixados, nem da rede. O único teste que toca a base real é
`test_base_real_se_baixada`, que pula se ela não existir.
"""
import csv
from pathlib import Path

import pytest

from src.evaluation.plano.onet import (
    ARQUIVO_OCUPACOES,
    ARQUIVO_TECNOLOGIAS,
    AREA_PARA_SOC,
    BaseONetAusente,
    DIR_PADRAO,
    VocabularioONet,
)
from src.evaluation.plano.pool_vagas import AREA_PADRAO, AREAS_TECNICAS

# Ocupações que o mapa realmente usa, para a base falsa não precisar ser completa.
SOCS_DO_MAPA = sorted({soc for socs in AREA_PARA_SOC.values() for soc in socs})


def _escrever_base(diretorio: Path, linhas_tecnologia: list[dict], socs: list[str]) -> Path:
    diretorio.mkdir(parents=True, exist_ok=True)

    with (diretorio / ARQUIVO_OCUPACOES).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["O*NET-SOC Code", "Title", "Description"])
        w.writeheader()
        for soc in socs:
            w.writerow({"O*NET-SOC Code": soc, "Title": f"Ocupação {soc}", "Description": "..."})

    campos = ["O*NET-SOC Code", "Title", "Workplace Example", "Element ID",
              "Element Name", "Hot Technology", "In Demand"]
    with (diretorio / ARQUIVO_TECNOLOGIAS).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for linha in linhas_tecnologia:
            w.writerow({c: linha.get(c, "") for c in campos})
    return diretorio


def _tec(soc: str, nome: str, categoria: str = "Cat", hot: str = "N", demanda: str = "N") -> dict:
    return {
        "O*NET-SOC Code": soc,
        "Title": f"Ocupação {soc}",
        "Workplace Example": nome,
        "Element ID": "43.23.00.00",
        "Element Name": categoria,
        "Hot Technology": hot,
        "In Demand": demanda,
    }


@pytest.fixture
def base(tmp_path):
    """Base mínima cobrindo todos os SOCs do mapa, com conteúdo controlado em
    Backend (15-1252.00 / 15-1251.00) e Frontend (15-1254.00)."""
    linhas = [_tec(soc, f"Generica {soc}", "Categoria generica") for soc in SOCS_DO_MAPA]
    linhas += [
        _tec("15-1252.00", "Python", "Development environment software", hot="Y", demanda="Y"),
        _tec("15-1252.00", "Docker", "Development environment software", hot="Y"),
        _tec("15-1252.00", "Cobol", "Compiler software"),
        _tec("15-1251.00", "Python", "Development environment software"),   # repetida, sem flags
        _tec("15-1251.00", "Assembler", "Compiler software"),
        _tec("15-1254.00", "React", "Web platform development software", hot="Y"),
    ]
    return _escrever_base(tmp_path / "onet", linhas, SOCS_DO_MAPA)


@pytest.fixture
def vocab(base):
    return VocabularioONet.carregar(base)


# --- carga ------------------------------------------------------------------


def test_base_ausente_da_erro_acionavel(tmp_path):
    with pytest.raises(BaseONetAusente, match="baixar_onet"):
        VocabularioONet.carregar(tmp_path / "nao-existe")


def test_carrega_titulos_e_tecnologias(vocab):
    assert vocab.titulo("15-1252.00") == "Ocupação 15-1252.00"
    assert "Python" in vocab.termos("Backend")


def test_titulo_desconhecido_devolve_o_proprio_codigo(vocab):
    assert vocab.titulo("99-9999.99") == "99-9999.99"


# --- consulta por área ------------------------------------------------------


def test_area_junta_as_ocupacoes_mapeadas(vocab):
    """Backend = Software Developers + Computer Programmers."""
    termos = vocab.termos("Backend", apenas_hot=False)
    assert "Docker" in termos      # só de 15-1252.00
    assert "Assembler" in termos   # só de 15-1251.00


def test_apenas_hot_e_o_padrao(vocab):
    assert vocab.termos("Backend") == ["Python", "Docker"]
    assert "Cobol" in vocab.termos("Backend", apenas_hot=False)


def test_tecnologia_repetida_entre_ocupacoes_aparece_uma_vez(vocab):
    """Python está em 15-1252.00 (hot) e 15-1251.00 (sem flag): uma entrada só,
    mantendo as flags mais fortes — senão a área perderia a marcação de hot
    dependendo da ordem do arquivo."""
    pythons = [t for t in vocab.tecnologias("Backend", apenas_hot=False) if t.nome == "Python"]
    assert len(pythons) == 1
    assert pythons[0].hot and pythons[0].em_demanda


def test_ordem_e_deterministica_para_o_limite_cortar_sempre_igual(vocab):
    """hot+demanda primeiro, depois hot, depois alfabética: o corte por `limite`
    não pode depender da ordem em que o CSV foi lido."""
    todos = vocab.tecnologias("Backend", apenas_hot=False)
    assert [t.nome for t in todos][:2] == ["Python", "Docker"]
    assert vocab.termos("Backend", apenas_hot=False, limite=1) == ["Python"]


def test_limite_corta_sem_reordenar(vocab):
    completo = vocab.termos("Backend", apenas_hot=False)
    assert vocab.termos("Backend", apenas_hot=False, limite=3) == completo[:3]


def test_categorias_sao_as_classes_unspsc(vocab):
    assert vocab.categorias("Backend") == [
        "Categoria generica",
        "Compiler software",
        "Development environment software",
    ]


def test_area_desconhecida_cai_no_padrao(vocab):
    assert vocab.ocupacoes("Área Inventada") == AREA_PARA_SOC[AREA_PADRAO]


# --- consistência do mapa ---------------------------------------------------


def test_mapa_cobre_todas_as_areas_do_pool():
    """Se `pool_vagas.AREAS_TECNICAS` ganhar uma área, o mapa tem que ganhar
    junto — este teste é o que impede a área nova de cair no padrão em silêncio."""
    areas_do_pool = {nome for nome, _ in AREAS_TECNICAS} | {AREA_PADRAO}
    assert areas_do_pool == set(AREA_PARA_SOC)


def test_base_completa_nao_tem_problemas(vocab):
    assert vocab.problemas() == []


def test_soc_fora_da_base_vira_problema(tmp_path):
    socs_incompletos = [s for s in SOCS_DO_MAPA if s != "15-1253.00"]  # QA/Testes
    base = _escrever_base(
        tmp_path / "parcial",
        [_tec(soc, f"Generica {soc}") for soc in socs_incompletos],
        socs_incompletos,
    )
    problemas = VocabularioONet.carregar(base).problemas()
    assert any("15-1253.00" in p for p in problemas)


def test_soc_sem_tecnologia_vira_problema(tmp_path):
    """O SOC existe no occupation_data mas nenhuma tecnologia o referencia —
    silenciosamente daria vocabulário vazio para a área."""
    base = _escrever_base(
        tmp_path / "vazio",
        [_tec(soc, f"Generica {soc}") for soc in SOCS_DO_MAPA if soc != "15-1253.00"],
        SOCS_DO_MAPA,
    )
    problemas = VocabularioONet.carregar(base).problemas()
    assert any("não tem tecnologia listada" in p for p in problemas)


def test_resumo_lista_todas_as_areas(vocab):
    resumo = vocab.resumo()
    assert set(resumo["areas"]) == set(AREA_PARA_SOC)
    assert resumo["areas"]["Backend"]["tecnologias_hot"] == 2
    assert resumo["areas"]["Backend"]["socs"][0].startswith("15-1252.00")


# --- base real (pula se não estiver baixada) --------------------------------


@pytest.mark.skipif(
    not (DIR_PADRAO / ARQUIVO_TECNOLOGIAS).exists(),
    reason="base O*NET não baixada (scripts/eval/baixar_onet.py)",
)
def test_base_real_se_baixada():
    """Sanidade contra a base de verdade: mapa íntegro e vocabulário reconhecível.
    Os termos abaixo são o mínimo que uma taxonomia de tecnologia útil para
    triagem de currículo de engenharia precisa conter."""
    vocab = VocabularioONet.carregar()
    assert vocab.problemas() == []

    devops = [t.lower() for t in vocab.termos("DevOps/Infra", apenas_hot=False)]
    assert any("kubernetes" in t for t in devops)
    assert any("terraform" in t for t in devops)

    frontend = [t.lower() for t in vocab.termos("Frontend", apenas_hot=False)]
    assert any("react" in t for t in frontend)

    # O corte por hot precisa continuar sendo um vocabulário utilizável, não 5 itens.
    assert 50 <= len(vocab.termos("Backend")) <= 300
