"""Testa o sorteio estratificado, a checagem de cobertura e a marcação de
vagas de calibração (seção 2 do plano de avaliação) com pools pequenos e
determinísticos — não é o dataset real, só fixtures pra exercitar a lógica."""
import csv

import pytest

from src.evaluation.plano.config import ConfigAvaliacao
from src.evaluation.plano.selecao import (
    CoberturaInsuficiente,
    VagaPool,
    checar_cobertura,
    gerar_tabela_controle,
    salvar_tabela_controle,
    selecionar_vagas_calibracao,
    sortear_com_cobertura,
    sortear_vagas_estudo,
)

DESC_OK = "Requisito 1: Python\nRequisito 2: SQL\nRequisito 3: comunicação"
DESC_TRIVIAL = "vaga de python"


def _vaga(i: int, area: str | None, senioridade: str | None, texto: str = DESC_OK) -> VagaPool:
    return VagaPool(vaga_id=f"v{i}", cargo=f"Cargo {i}", texto=texto, area_tecnica=area, senioridade=senioridade)


def _cfg(**overrides) -> ConfigAvaliacao:
    return ConfigAvaliacao(_env_file=None, **overrides)


def _pool_com_metadados(n_por_estrato: int = 10) -> list[VagaPool]:
    """3 áreas x 2 senioridades, `n_por_estrato` vagas cada — pool "rico"."""
    pool = []
    i = 0
    for area in ("backend", "frontend", "dados"):
        for senioridade in ("junior", "senior"):
            for _ in range(n_por_estrato):
                pool.append(_vaga(i, area, senioridade))
                i += 1
    return pool


# --- sortear_vagas_estudo ----------------------------------------------------


def test_erro_se_pool_menor_que_n_vagas_estudo():
    pool = _pool_com_metadados(n_por_estrato=1)  # 6 vagas
    cfg = _cfg(n_vagas_estudo=10)
    with pytest.raises(ValueError, match="N_VAGAS_ESTUDO"):
        sortear_vagas_estudo(pool, cfg, seed=1)


def test_sorteio_estratificado_retorna_exatamente_n():
    pool = _pool_com_metadados()
    cfg = _cfg(n_vagas_estudo=30)
    vagas = sortear_vagas_estudo(pool, cfg, seed=42)
    assert len(vagas) == 30
    assert len(set(v.vaga_id for v in vagas)) == 30  # sem repetição


def test_sorteio_estratificado_cobre_proporcao_dos_estratos():
    # estratos desbalanceados: "dados" tem 5x mais vagas que os outros
    pool = _pool_com_metadados(n_por_estrato=10)
    pool += [_vaga(1000 + i, "dados", "pleno") for i in range(200)]
    cfg = _cfg(n_vagas_estudo=50)
    vagas = sortear_vagas_estudo(pool, cfg, seed=7)
    assert len(vagas) == 50
    n_dados = sum(1 for v in vagas if v.area_tecnica == "dados")
    # "dados" é ~85% do pool (260/320); a cota proporcional deve refletir isso,
    # não ficar igual às demais áreas
    assert n_dados > 30


def test_sorteio_e_determinado_pela_seed():
    pool = _pool_com_metadados()
    cfg = _cfg(n_vagas_estudo=20)
    a = sortear_vagas_estudo(pool, cfg, seed=123)
    b = sortear_vagas_estudo(pool, cfg, seed=123)
    assert [v.vaga_id for v in a] == [v.vaga_id for v in b]


def test_metadados_insuficientes_cai_pro_sorteio_simples():
    # só 2 das 20 vagas têm área/senioridade preenchidas -> não estratifica
    pool = [_vaga(i, "backend" if i < 2 else None, "pleno" if i < 2 else None) for i in range(20)]
    cfg = _cfg(n_vagas_estudo=10)
    vagas = sortear_vagas_estudo(pool, cfg, seed=1)
    assert len(vagas) == 10  # sorteio aleatório simples ainda funciona


# --- checar_cobertura ---------------------------------------------------------


def test_cobertura_ok_com_areas_senioridades_e_descricoes_variadas():
    vagas = [_vaga(i, area, "junior" if i % 2 == 0 else "senior") for i, area in enumerate(["backend", "frontend", "dados"])]
    cfg = _cfg()
    checagem = checar_cobertura(vagas, cfg)
    assert checagem["ok"]
    assert checagem["areas"] == ["backend", "dados", "frontend"]


def test_cobertura_falha_por_poucas_areas():
    vagas = [_vaga(i, "backend", "junior") for i in range(5)]
    cfg = _cfg(min_areas_tecnicas=3)
    checagem = checar_cobertura(vagas, cfg)
    assert not checagem["ok"]
    assert any("área" in b for b in checagem["bloqueios"])


def test_cobertura_falha_por_poucas_senioridades():
    vagas = [_vaga(i, area, "junior") for i, area in enumerate(["backend", "frontend", "dados"])]
    cfg = _cfg(min_senioridades=2)
    checagem = checar_cobertura(vagas, cfg)
    assert not checagem["ok"]
    assert any("senioridade" in b for b in checagem["bloqueios"])


def test_cobertura_falha_por_descricao_trivial():
    vagas = [
        _vaga(0, "backend", "junior", texto=DESC_OK),
        _vaga(1, "frontend", "senior", texto=DESC_TRIVIAL),
        _vaga(2, "dados", "pleno", texto=DESC_OK),
    ]
    cfg = _cfg()
    checagem = checar_cobertura(vagas, cfg)
    assert not checagem["ok"]
    assert any("trivial" in b and "v1" in b for b in checagem["bloqueios"])


# --- sortear_com_cobertura -----------------------------------------------------


def test_sortear_com_cobertura_encontra_amostra_valida():
    pool = _pool_com_metadados()
    cfg = _cfg(n_vagas_estudo=30, min_areas_tecnicas=3, min_senioridades=2)
    vagas, checagem, seed_usada = sortear_com_cobertura(pool, cfg, seed=99)
    assert len(vagas) == 30
    assert checagem["ok"]
    assert isinstance(seed_usada, int)


def test_sortear_com_cobertura_levanta_quando_pool_nao_tem_diversidade():
    # pool inteiro é da mesma área -> nenhuma tentativa vai cobrir min_areas_tecnicas=3
    pool = [_vaga(i, "backend", "junior" if i % 2 == 0 else "senior") for i in range(50)]
    cfg = _cfg(n_vagas_estudo=20, min_areas_tecnicas=3)
    with pytest.raises(CoberturaInsuficiente) as exc:
        sortear_com_cobertura(pool, cfg, seed=1, max_tentativas=5)
    assert "área" in exc.value.checagem["bloqueios"][0]


# --- selecionar_vagas_calibracao -----------------------------------------------


def test_calibracao_cobre_areas_minimas():
    vagas_estudo = _pool_com_metadados(n_por_estrato=3)  # 3 áreas, 18 vagas
    cfg = _cfg(n_vagas_calibracao=3, min_areas_calibracao=2)
    ids = selecionar_vagas_calibracao(vagas_estudo, cfg, seed=5)
    assert len(ids) == 3
    areas_escolhidas = {v.area_tecnica for v in vagas_estudo if v.vaga_id in ids}
    assert len(areas_escolhidas) >= 2


def test_calibracao_erro_se_estudo_menor_que_n_calibracao():
    vagas_estudo = _pool_com_metadados(n_por_estrato=1)[:2]
    cfg = _cfg(n_vagas_calibracao=3)
    with pytest.raises(ValueError, match="N_VAGAS_CALIBRACAO"):
        selecionar_vagas_calibracao(vagas_estudo, cfg, seed=1)


def test_calibracao_levanta_quando_so_uma_area_disponivel():
    vagas_estudo = [_vaga(i, "backend", "junior") for i in range(5)]
    cfg = _cfg(n_vagas_calibracao=3, min_areas_calibracao=2)
    with pytest.raises(CoberturaInsuficiente):
        selecionar_vagas_calibracao(vagas_estudo, cfg, seed=1)


# --- tabela de controle ---------------------------------------------------------


def test_gerar_tabela_controle_marca_calibracao_corretamente():
    vagas_estudo = [_vaga(0, "backend", "junior"), _vaga(1, "frontend", "senior")]
    tabela = gerar_tabela_controle(vagas_estudo, ids_calibracao={"v0"})
    assert tabela == [
        {"vaga_id": "v0", "cargo": "Cargo 0", "area_tecnica": "backend", "senioridade": "junior", "is_calibracao": True},
        {"vaga_id": "v1", "cargo": "Cargo 1", "area_tecnica": "frontend", "senioridade": "senior", "is_calibracao": False},
    ]


def test_salvar_tabela_controle_escreve_csv_legivel(tmp_path):
    vagas_estudo = [_vaga(0, "backend", "junior")]
    tabela = gerar_tabela_controle(vagas_estudo, ids_calibracao={"v0"})
    caminho = tmp_path / "sub" / "vagas_estudo.csv"

    salvar_tabela_controle(tabela, caminho)

    with caminho.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert linhas == [
        {"vaga_id": "v0", "cargo": "Cargo 0", "area_tecnica": "backend", "senioridade": "junior", "is_calibracao": "True"}
    ]
