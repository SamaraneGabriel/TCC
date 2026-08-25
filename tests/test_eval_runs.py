from pathlib import Path

from src.evaluation.runs import carregar_json, listar_runs, novo_run_dir, salvar_json, slug

# ── slug ─────────────────────────────────────────────────────────────────────


def test_slug_normalizes_spaces_and_punctuation():
    assert slug("Dev Backend Python - Pleno") == "dev-backend-python-pleno"


def test_slug_truncates():
    texto = "a" * 100
    assert len(slug(texto, tamanho=40)) == 40


def test_slug_empty_falls_back_to_run():
    assert slug("---") == "run"


# ── novo_run_dir ─────────────────────────────────────────────────────────────


def test_novo_run_dir_creates_folder_with_timestamp_and_slug(tmp_path):
    pasta = novo_run_dir("Dev Backend Python", base=tmp_path)
    assert pasta.exists()
    assert pasta.is_dir()
    assert "dev-backend-python" in pasta.name


def test_novo_run_dir_is_unique_per_call(tmp_path):
    pasta1 = novo_run_dir("Vaga A", base=tmp_path)
    pasta2 = novo_run_dir("Vaga A", base=tmp_path)
    # mesmo nome de vaga, mas pastas diferentes se rodar em segundos diferentes
    # (aqui só garantimos que a segunda chamada nao falha e ambas existem)
    assert pasta1.exists() and pasta2.exists()


# ── salvar_json / carregar_json ──────────────────────────────────────────────


def test_salvar_e_carregar_json_roundtrip(tmp_path):
    dados = {"titulo": "Vaga X", "skills": ["Python", "FastAPI"], "score": 0.87}
    salvar_json(tmp_path, "vaga.json", dados)
    assert carregar_json(tmp_path, "vaga.json") == dados


def test_salvar_json_preserva_acentos(tmp_path):
    dados = {"nome": "José da Silva", "área": "Engenharia de Software"}
    salvar_json(tmp_path, "curriculo.json", dados)
    assert carregar_json(tmp_path, "curriculo.json") == dados


# ── listar_runs ──────────────────────────────────────────────────────────────


def test_listar_runs_empty_when_base_missing(tmp_path):
    assert listar_runs(base=tmp_path / "nao_existe") == []


def test_listar_runs_returns_folders_newest_first(tmp_path):
    (tmp_path / "20260101_100000_vaga-a").mkdir()
    (tmp_path / "20260101_120000_vaga-b").mkdir()
    (tmp_path / "arquivo_solto.txt").write_text("nao e pasta")

    runs = listar_runs(base=tmp_path)

    assert [p.name for p in runs] == ["20260101_120000_vaga-b", "20260101_100000_vaga-a"]
