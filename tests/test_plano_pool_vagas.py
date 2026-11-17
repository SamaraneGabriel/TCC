"""Testa o carregamento do pool de vagas reais (seção 2.1) com um CSV mínimo
em tmp_path — mesmas colunas do dataset do Kaggle, conteúdo de fixture."""
import csv

from src.evaluation.plano.pool_vagas import (
    AREA_PADRAO,
    amostrar_pool,
    area_tecnica,
    carregar_pool,
    carregar_postings,
    cobertura_pool,
    modelo_trabalho,
    salvar_pool,
    senioridade,
    vaga_id,
)
from src.evaluation.plano.selecao import VagaPool

COLUNAS = [
    "job_title", "company", "job_location", "job_link", "first_seen",
    "search_city", "search_country", "job level", "job_type", "job_summary", "job_skills",
]

DESCRICAO = (
    "We are looking for an engineer to join the team.\n"
    "Responsibilities include building services and reviewing code.\n"
    "Requirements: 5+ years of experience, strong communication skills, "
    "and a track record of shipping production software in a collaborative environment.\n"
) * 2


def _linha(**overrides) -> dict:
    base = {
        "job_title": "Senior Software Engineer",
        "company": "Acme",
        "job_location": "Austin, TX",
        "job_link": "https://www.linkedin.com/jobs/view/senior-software-engineer-at-acme-3780474203",
        "first_seen": "2023-12-25",
        "search_city": "Austin",
        "search_country": "United States",
        "job level": "Mid senior",
        "job_type": "Remote",
        "job_summary": DESCRICAO,
        "job_skills": "Java, Spring, PostgreSQL",
    }
    return {**base, **overrides}


def _csv(tmp_path, linhas: list[dict]):
    caminho = tmp_path / "postings.csv"
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)
    return caminho


# --- derivações ---------------------------------------------------------------


def test_modelo_trabalho_usa_o_vocabulario_do_projeto():
    assert modelo_trabalho("Remote") == "remoto"
    assert modelo_trabalho("Hybrid") == "híbrido"
    assert modelo_trabalho("Onsite") == "presencial"
    assert modelo_trabalho("") is None


def test_senioridade_prioriza_o_titulo_sobre_o_job_level():
    assert senioridade("Senior Software Engineer", "Associate") == "senior"
    assert senioridade("Junior Developer", "Mid senior") == "junior"


def test_senioridade_cai_no_job_level_quando_o_titulo_e_generico():
    assert senioridade("Software Engineer", "Associate") == "pleno"
    assert senioridade("Software Engineer", "Mid senior") == "senior"
    assert senioridade("Software Engineer", "") is None


def test_area_tecnica_prefere_a_especializacao_do_titulo():
    assert area_tecnica("Embedded Software Engineer") == "Embarcados/Firmware"
    assert area_tecnica("Full Stack Engineer (JavaScript)") == "Full-stack"
    assert area_tecnica("Machine Learning Engineer") == "IA/ML"


def test_area_tecnica_usa_as_skills_quando_o_titulo_e_generico():
    assert area_tecnica("Software Engineer") == AREA_PADRAO
    assert area_tecnica("Software Engineer", "Kubernetes, Terraform") == "DevOps/Infra"


def test_vaga_id_vem_do_numero_do_anuncio():
    assert vaga_id("https://linkedin.com/jobs/view/x-at-y-3780474203", "X", "Y", 0) == "vaga_3780474203"


def test_vaga_id_cai_no_hash_quando_nao_ha_numero():
    gerado = vaga_id("", "Dev", "Acme", 7)
    assert gerado.startswith("vaga_")
    assert gerado == vaga_id(None, "Dev", "Acme", 7)  # estável


# --- carregamento -------------------------------------------------------------


def test_carregar_postings_monta_a_vaga_completa(tmp_path):
    vagas = carregar_postings(_csv(tmp_path, [_linha()]))
    assert len(vagas) == 1
    v = vagas[0]
    assert (v.vaga_id, v.senioridade, v.modelo_trabalho) == ("vaga_3780474203", "senior", "remoto")
    assert v.area_tecnica == "Backend"
    assert v.empresa == "Acme" and v.localidade == "Austin, TX"
    assert v.skills == "Java, Spring, PostgreSQL"


def test_carregar_postings_descarta_descricao_trivial(tmp_path):
    linhas = [_linha(), _linha(job_link="...-111111111", job_summary="Vaga de dev.")]
    assert len(carregar_postings(_csv(tmp_path, linhas))) == 1


def test_carregar_postings_descarta_anuncio_repetido(tmp_path):
    # mesmo job_link capturado por duas buscas de cidade diferentes
    linhas = [_linha(), _linha(search_city="Dallas")]
    assert len(carregar_postings(_csv(tmp_path, linhas))) == 1


def test_carregar_postings_descarta_texto_repetido(tmp_path):
    """15% do dataset é a mesma vaga republicada por cidade — duas cópias no
    estudo quebrariam a independência que o bootstrap da seção 9 pressupõe."""
    linhas = [_linha(), _linha(job_link="...-222222222", job_location="Dallas, TX")]
    assert len(carregar_postings(_csv(tmp_path, linhas))) == 1
    assert len(carregar_postings(_csv(tmp_path, linhas), deduplicar=False)) == 2


def test_carregar_postings_descarta_cargo_fora_de_engenharia(tmp_path):
    linhas = [
        _linha(),
        _linha(
            job_link="...-333333333",
            job_title="Patent Attorney / Computer Software IP",
            job_summary=DESCRICAO + "\nPatentes de software.",  # texto próprio, não é duplicata
        ),
    ]
    assert [v.cargo for v in carregar_postings(_csv(tmp_path, linhas))] == ["Senior Software Engineer"]
    assert len(carregar_postings(_csv(tmp_path, linhas), excluir_nao_engenharia=False)) == 2


def test_carregar_postings_respeita_o_limite(tmp_path):
    linhas = [
        _linha(job_link=f"...-{100000000 + i}", job_summary=DESCRICAO + f"\nVaga {i}.")
        for i in range(5)
    ]
    assert len(carregar_postings(_csv(tmp_path, linhas), limite=3)) == 3


# --- amostragem e persistência ------------------------------------------------


def _pool(n: int) -> list[VagaPool]:
    return [VagaPool(vaga_id=f"v{i}", cargo=f"Dev {i}", texto="linha1\nlinha2") for i in range(n)]


def test_amostrar_pool_e_deterministico_com_a_mesma_seed():
    vagas = _pool(50)
    assert [v.vaga_id for v in amostrar_pool(vagas, 10, seed=7)] == [
        v.vaga_id for v in amostrar_pool(vagas, 10, seed=7)
    ]


def test_amostrar_pool_devolve_tudo_quando_n_excede_o_universo():
    assert len(amostrar_pool(_pool(5), 10, seed=1)) == 5


def test_salvar_e_carregar_pool_preserva_os_campos(tmp_path):
    original = [
        VagaPool(
            vaga_id="v1", cargo="Dev", texto="linha1\nlinha2", area_tecnica="Backend",
            senioridade="senior", empresa="Acme", localidade="Austin, TX",
            modelo_trabalho="remoto", skills="Java",
        )
    ]
    caminho = tmp_path / "pool.jsonl"
    salvar_pool(original, caminho)
    assert carregar_pool(caminho) == original


def test_cobertura_pool_conta_areas_e_senioridades():
    vagas = [
        VagaPool("v1", "Dev", "t", "Backend", "senior"),
        VagaPool("v2", "Dev", "t", "Backend", "pleno"),
        VagaPool("v3", "Dev", "t", "Frontend", "senior"),
    ]
    c = cobertura_pool(vagas)
    assert c["total"] == 3
    assert c["areas"] == {"Backend": 2, "Frontend": 1}
    assert c["senioridades"] == {"senior": 2, "pleno": 1}
