"""Garante que as fórmulas de tamanho de amostra reproduzem os números citados
no plano de avaliação, e que nada nelas depende de K/N fixados no código."""
import pytest

from src.evaluation.plano.config import (
    ConfigAvaliacao,
    corrigir_populacao_finita,
    n0_amostra,
    tamanho_amostra,
)


@pytest.fixture
def cfg():
    """Defaults do plano, isolados do .env do ambiente."""
    return ConfigAvaliacao(_env_file=None)


# --- fórmulas puras (seções 3.1 e 4.1) ---------------------------------------


def test_n0_bate_com_o_plano():
    assert n0_amostra(1.96, 0.10) == pytest.approx(96.04, abs=0.01)
    assert n0_amostra(1.96, 0.25) == pytest.approx(15.37, abs=0.01)


def test_n0_rejeita_margem_invalida():
    with pytest.raises(ValueError):
        n0_amostra(1.96, 0.0)


def test_fpc_reduz_o_tamanho():
    n0 = n0_amostra(1.96, 0.10)
    assert corrigir_populacao_finita(n0, 4617) == pytest.approx(94.10, abs=0.01)
    assert corrigir_populacao_finita(n0, 4617) < n0


def test_amostra_nunca_excede_o_universo():
    # universo minúsculo: a fórmula pediria 97, mas só existem 10 itens
    assert tamanho_amostra(1.96, 0.10, n_universo=10) == 10


def test_fpc_pulada_quando_universo_omitido():
    assert tamanho_amostra(1.96, 0.10) == 97


# --- valores derivados com os defaults do plano ------------------------------


def test_calibracao_97_por_grupo_194_no_total(cfg):
    assert cfg.n_calibracao_por_grupo == 97
    assert cfg.n_calibracao_total == 194


def test_universos_e_amostras_dos_estratos(cfg):
    assert (cfg.n_universo_b, cfg.n_b) == (175, 15)
    assert (cfg.n_universo_c, cfg.n_c) == (4617, 95)


def test_double_label_cobre_a_fracao(cfg):
    # 0,15 * 194 = 29,1 — o plano diz "~29"; arredondamos pra cima pra não ficar abaixo
    assert cfg.n_double_label == 30
    assert cfg.n_double_label >= cfg.double_label_fracao * cfg.n_calibracao_total


def test_auditoria_respeita_piso_e_teto(cfg):
    assert cfg.n_auditoria(25) == 5      # piso: ceil(0,10*25)=3 -> 5
    assert cfg.n_auditoria(95) == 10     # fração: ceil(0,10*95)=10
    assert cfg.n_auditoria(3) == 3       # teto: estrato menor que o piso


def test_defaults_do_plano_sao_consistentes(cfg):
    assert cfg.problemas() == []


# --- reatividade: nada de K/N fixado no código -------------------------------


def test_mudar_k_e_n_total_recalcula_tudo():
    cfg = ConfigAvaliacao(_env_file=None, k=10, n_total=1000, top_pool=100)
    assert cfg.n_universo_b == 90
    assert cfg.n_universo_c == 900
    assert cfg.n_b == tamanho_amostra(cfg.z, cfg.e_b, 90)
    assert cfg.n_c == tamanho_amostra(cfg.z, cfg.e_c, 900)
    assert cfg.n_b != 15 and cfg.n_c != 95


def test_base_pequena_demais_vira_problema():
    # cenário real de hoje: 30 currículos indexados, TOP_POOL=200
    cfg = ConfigAvaliacao(_env_file=None, n_total=30)
    problemas = " ".join(cfg.problemas())
    assert "estrato C" in problemas


def test_top_pool_menor_que_k_vira_problema():
    cfg = ConfigAvaliacao(_env_file=None, k=25, top_pool=25)
    assert any("estrato B" in p for p in cfg.problemas())


def test_calibracao_maior_que_estudo_vira_problema():
    cfg = ConfigAvaliacao(_env_file=None, n_vagas_estudo=3, n_vagas_calibracao=5)
    assert any("subconjunto" in p for p in cfg.problemas())


def test_afrouxar_e_calib_vira_problema():
    cfg = ConfigAvaliacao(_env_file=None, e_calib=0.20)
    assert any("E_CALIB" in p for p in cfg.problemas())


def test_n_vagas_fora_da_faixa_vira_aviso():
    cfg = ConfigAvaliacao(_env_file=None, n_vagas_estudo=10)
    assert any("N_VAGAS_ESTUDO" in a for a in cfg.avisos())
    assert cfg.problemas() == []
