"""
Persistência dos runs de avaliação — uma pasta por execução em data/eval/runs/.

Puro: só cria pastas e serializa/desserializa JSON, não chama LLM nem toca o
vectorstore. Os nomes de arquivo esperados por convenção dentro de cada pasta
(escritos pelo orquestrador em scripts/eval/run_llm_judge_eval.py) são:
  vaga.json, resultados.json, julgamento.json, usage.json, metricas.json
"""
import json
import re
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("data/eval/runs")


def slug(texto: str, tamanho: int = 40) -> str:
    """Normaliza um texto pra usar em nome de pasta/arquivo."""
    s = re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")
    return s[:tamanho] or "run"


def novo_run_dir(titulo_vaga: str, base: Path = RUNS_DIR) -> Path:
    """Cria (e retorna) a pasta de um novo run: <timestamp>_<slug-da-vaga>/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta = base / f"{timestamp}_{slug(titulo_vaga)}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def salvar_json(pasta: Path, nome_arquivo: str, dados) -> None:
    (pasta / nome_arquivo).write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def carregar_json(pasta: Path, nome_arquivo: str):
    return json.loads((pasta / nome_arquivo).read_text(encoding="utf-8"))


def listar_runs(base: Path = RUNS_DIR) -> list[Path]:
    """Lista as pastas de run existentes, mais recentes primeiro (ordem lexicográfica
    do timestamp no nome já garante isso)."""
    if not base.exists():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
