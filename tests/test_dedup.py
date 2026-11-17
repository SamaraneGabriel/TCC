"""Testa a detecção de duplicatas (`src/dedup.py`) com um corpus mínimo em
tmp_path — não toca Qdrant nem o dataset real."""
from src.dedup import (
    ArquivoFonte,
    PontoIndexado,
    chave_nome,
    corresponder,
    escolher_canonico,
    hash_conteudo,
    indexar_arquivos,
    montar_plano,
    normalizar,
    score_correspondencia,
)

TEXTO_ANA = (
    "ANA SILVA\nana.silva@example.com\nSão Paulo\n"
    "Desenvolvedora backend com Python, Django, PostgreSQL e Kubernetes."
)
TEXTO_BRUNO = (
    "BRUNO COSTA\nbruno@example.com\nCuritiba\n"
    "Analista de dados com Tableau, Power BI, Excel avançado e SQL Server."
)


def _arquivos(tmp_path, **conteudos) -> dict[str, ArquivoFonte]:
    for nome, texto in conteudos.items():
        (tmp_path / nome).write_text(texto, encoding="utf-8")
    return indexar_arquivos(tmp_path.glob("*.txt"))


def _ana(pid: str, **overrides) -> PontoIndexado:
    base = dict(
        nome="ANA SILVA",
        email="ana.silva@example.com",
        skills=("Python", "Django", "PostgreSQL", "Kubernetes"),
        localidade="São Paulo",
        senioridade="pleno",
    )
    return PontoIndexado(id=pid, **{**base, **overrides})


# --- normalização -------------------------------------------------------------


def test_normalizar_colapsa_espacos_e_caixa():
    assert normalizar("  Python\n\n  Django\t") == "python django"


def test_hash_conteudo_ignora_quebras_de_linha():
    assert hash_conteudo("Python\nDjango") == hash_conteudo("  python   django  ")


def test_hash_conteudo_difere_com_conteudo_diferente():
    assert hash_conteudo(TEXTO_ANA) != hash_conteudo(TEXTO_BRUNO)


def test_chave_nome_ignora_pontuacao_acento_e_caixa():
    assert chave_nome("Juan Antonio AGUILAR-GARIB") == chave_nome("juan antonio aguilar garib")
    assert chave_nome("MOHD. RASHID") == chave_nome("Mohd Rashid")
    assert chave_nome("José da Silva") == "jose da silva"
    assert chave_nome(None) == ""


# --- indexar_arquivos ---------------------------------------------------------


def test_indexar_arquivos_colapsa_arquivos_identicos(tmp_path):
    arquivos = _arquivos(
        tmp_path, **{"a.txt": TEXTO_ANA, "copia.txt": TEXTO_ANA, "b.txt": TEXTO_BRUNO}
    )
    assert len(arquivos) == 2
    hash_ana = hash_conteudo(TEXTO_ANA)
    assert arquivos[hash_ana].arquivos == ("a.txt", "copia.txt")


# --- score/correspondência ----------------------------------------------------


def test_score_soma_email_nome_skills_e_localidade():
    score = score_correspondencia(_ana("p1"), normalizar(TEXTO_ANA))
    assert score == 13.0  # 5 (email) + 3 (nome) + 4 (todas as skills) + 1 (localidade)


def test_score_ignora_skills_curtas():
    # "C" e "R" casariam com qualquer texto; só "Kubernetes" conta.
    ponto = PontoIndexado(id="p1", skills=("C", "R", "Kubernetes"))
    assert score_correspondencia(ponto, normalizar(TEXTO_ANA)) == 4.0


def test_corresponder_acha_o_arquivo_de_origem(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    c = corresponder([_ana("p1")], arquivos)["p1"]
    assert c.arquivos == ("ana.txt",)
    assert c.confiavel


def test_corresponder_marca_nao_confiavel_quando_evidencia_e_fraca(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    ponto = PontoIndexado(id="p1", skills=("Cobol", "Fortran"))  # não aparece em nenhum
    assert not corresponder([ponto], arquivos)["p1"].confiavel


# --- canônico -----------------------------------------------------------------


def test_escolher_canonico_prefere_payload_mais_completo():
    rico = _ana("p1")
    pobre = PontoIndexado(id="p2", nome="ANA SILVA", skills=("Python",))
    pontos = {"p1": rico, "p2": pobre}
    assert escolher_canonico(["p2", "p1"], pontos) == "p1"


def test_escolher_canonico_desempata_por_id():
    a = PontoIndexado(id="p1", nome="X", skills=("Python",))
    b = PontoIndexado(id="p2", nome="X", skills=("Python",))
    assert escolher_canonico(["p1", "p2"], {"p1": a, "p2": b}) == "p2"


# --- plano --------------------------------------------------------------------


def test_plano_agrupa_pontos_do_mesmo_arquivo_fonte(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    # dois pontos sem nome (o sinal nome+email não os pegaria), mesma origem
    p1 = PontoIndexado(id="p1", skills=("Python", "Django", "PostgreSQL", "Kubernetes"))
    p2 = PontoIndexado(id="p2", skills=("Python", "Django", "PostgreSQL"))
    plano = montar_plano([p1, p2], arquivos)
    assert len(plano.grupos) == 1
    assert plano.grupos[0].motivos == ("arquivo_fonte",)
    assert set(plano.remover) == {"p2"}  # p1 tem mais skills


def test_plano_agrupa_mesma_pessoa_em_arquivos_diferentes(tmp_path):
    variante = TEXTO_ANA + "\nExperiência adicional com Docker e Terraform."
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "ana_v2.txt": variante})
    plano = montar_plano([_ana("p1"), _ana("p2", skills=("Python",))], arquivos)
    assert len(plano.grupos) == 1
    assert "nome_email" in plano.grupos[0].motivos
    assert plano.remover == ("p2",)


def test_plano_nao_junta_pessoas_diferentes(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    bruno = PontoIndexado(
        id="p2",
        nome="BRUNO COSTA",
        email="bruno@example.com",
        skills=("Tableau", "Power BI", "SQL Server"),
        localidade="Curitiba",
    )
    plano = montar_plano([_ana("p1"), bruno], arquivos)
    assert plano.grupos == ()
    assert plano.total_apos == 2


def test_plano_nao_junta_dois_sem_nome_de_origens_diferentes(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    sem_nome_1 = PontoIndexado(id="p1", skills=("Python", "Django", "Kubernetes"))
    sem_nome_2 = PontoIndexado(id="p2", skills=("Tableau", "Power BI", "SQL Server"))
    assert montar_plano([sem_nome_1, sem_nome_2], arquivos).grupos == ()


def test_plano_conta_totais(tmp_path):
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    pontos = [_ana("p1"), _ana("p2"), _ana("p3", email=None)]
    plano = montar_plano(pontos, arquivos)
    assert plano.total_pontos == 3
    assert len(plano.remover) == 2
    assert plano.total_apos == 1


def test_plano_junta_grafias_diferentes_do_mesmo_nome(tmp_path):
    """Caso real do corpus: o mesmo candidato extraído como "AGUILAR-GARIB" e
    "AGUILAR GARIB" virava dois grupos de duplicatas em vez de um."""
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    p1 = PontoIndexado(id="p1", nome="ANA-SILVA", email="ana.silva@example.com", skills=("Python",))
    p2 = PontoIndexado(id="p2", nome="Ana Silva", email="ana.silva@example.com", skills=("Django",))
    plano = montar_plano([p1, p2], arquivos)
    assert len(plano.grupos) == 1
    assert plano.grupos[0].tamanho == 2


def test_plano_nao_junta_pelo_nome_quando_nao_ha_email(tmp_path):
    """Caso real: currículos anônimos em que o LLM extraiu o título do documento
    como nome ("INFORMATION TECHNOLOGY SPECIALIST") — mesmo "nome", pessoas
    diferentes. Sem e-mail, só o arquivo-fonte pode agrupar."""
    arquivos = _arquivos(tmp_path, **{"ana.txt": TEXTO_ANA, "bruno.txt": TEXTO_BRUNO})
    p1 = PontoIndexado(id="p1", nome="ESPECIALISTA EM TI", skills=("Python", "Django"))
    p2 = PontoIndexado(id="p2", nome="ESPECIALISTA EM TI", skills=("Tableau", "Power BI"))
    assert montar_plano([p1, p2], arquivos).grupos == ()
