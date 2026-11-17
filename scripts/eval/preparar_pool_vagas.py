"""
Fase 1 do plano de avaliação (seção 2.1): monta o pool de vagas do estudo a
partir do dataset real e o congela em disco, para que o sorteio (seção 2.2) e
tudo que vier depois sejam reproduzíveis.

Uso:
    python scripts/eval/preparar_pool_vagas.py [--csv data/vagas/kaggle/postings.csv]
                                               [--n 5000] [--seed N]
                                               [--saida data/eval/plano/pool_vagas.jsonl]

Não chama LLM nem toca o Qdrant: só lê o CSV, normaliza (área técnica,
senioridade, modelo de trabalho), descarta descrições triviais e anúncios
repetidos, sorteia `--n` vagas e grava o JSONL do pool + o resumo de cobertura.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.plano.config import config
from src.evaluation.plano.pool_vagas import (
    amostrar_pool,
    carregar_postings,
    cobertura_pool,
    salvar_pool,
)

CSV_PADRAO = Path("data/vagas/kaggle/postings.csv")
SAIDA_PADRAO = Path("data/eval/plano/pool_vagas.jsonl")


def _arg(nome: str, padrao: str) -> str:
    return sys.argv[sys.argv.index(nome) + 1] if nome in sys.argv else padrao


def main() -> None:
    csv_path = Path(_arg("--csv", str(CSV_PADRAO)))
    saida = Path(_arg("--saida", str(SAIDA_PADRAO)))
    n = int(_arg("--n", "5000"))
    seed = int(_arg("--seed", str(config.seed_mestre)))

    if not csv_path.exists():
        print(f"CSV não encontrado: {csv_path}")
        sys.exit(1)

    print(f"lendo {csv_path}...")
    vagas = carregar_postings(csv_path)
    print(f"vagas utilizáveis no dataset: {len(vagas)}")

    pool = amostrar_pool(vagas, n, seed)
    salvar_pool(pool, saida)

    cobertura = cobertura_pool(pool)
    print(f"\npool de {cobertura['total']} vaga(s) (seed={seed}) -> {saida}")
    print("\náreas técnicas:")
    for area, qtd in cobertura["areas"].items():
        print(f"  {qtd:5}  {area}")
    print("\nsenioridades:")
    for senioridade, qtd in cobertura["senioridades"].items():
        print(f"  {qtd:5}  {senioridade}")

    resumo = saida.with_suffix(".cobertura.json")
    resumo.write_text(
        json.dumps({"seed": seed, "csv": str(csv_path), **cobertura}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nresumo: {resumo}")

    minimos = []
    if len(cobertura["areas"]) < config.min_areas_tecnicas:
        minimos.append(f"áreas técnicas: {len(cobertura['areas'])} < {config.min_areas_tecnicas}")
    if len([s for s in cobertura["senioridades"] if s]) < config.min_senioridades:
        minimos.append("senioridades distintas abaixo do mínimo")
    if len(pool) < config.n_vagas_estudo:
        minimos.append(f"pool ({len(pool)}) menor que N_VAGAS_ESTUDO ({config.n_vagas_estudo})")
    print(
        "\ncobertura mínima do plano: OK"
        if not minimos
        else "\nATENÇÃO — cobertura insuficiente: " + "; ".join(minimos)
    )


if __name__ == "__main__":
    main()
