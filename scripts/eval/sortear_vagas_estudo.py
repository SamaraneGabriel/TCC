"""
Fase 2 do plano de avaliação (seção 2, passos 2-4): sorteia as N_VAGAS_ESTUDO
vagas do pool, checa a cobertura exigida e marca as vagas de calibração.

Uso:
    python scripts/eval/sortear_vagas_estudo.py [--pool data/eval/plano/pool_vagas.jsonl]
                                                [--seed N] [--saida-dir data/eval/plano]

Saídas (seção 13):
    vagas_estudo.csv     tabela de controle: vaga_id, cargo, área, senioridade, is_calibracao
    vagas_estudo.jsonl   as mesmas vagas com o texto completo, para a fase de busca
    sorteio_vagas.json   seeds usadas e resultado da checagem de cobertura

Não gasta chamada de API: só sorteia sobre o pool já congelado em disco. Rodar
de novo com a mesma seed reproduz exatamente o mesmo sorteio.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.plano.config import config
from src.evaluation.plano.pool_vagas import carregar_pool, cobertura_pool, salvar_pool
from src.evaluation.plano.selecao import (
    CoberturaInsuficiente,
    gerar_tabela_controle,
    salvar_tabela_controle,
    selecionar_vagas_calibracao,
    sortear_com_cobertura,
)

POOL_PADRAO = Path("data/eval/plano/pool_vagas.jsonl")


def _arg(nome: str, padrao: str) -> str:
    return sys.argv[sys.argv.index(nome) + 1] if nome in sys.argv else padrao


def main() -> int:
    pool_path = Path(_arg("--pool", str(POOL_PADRAO)))
    saida_dir = Path(_arg("--saida-dir", config.saida_dir))
    seed = int(_arg("--seed", str(config.seed_mestre)))

    if not pool_path.exists():
        print(f"pool não encontrado: {pool_path}\nRode antes: python scripts/eval/preparar_pool_vagas.py")
        return 1

    pool = carregar_pool(pool_path)
    print(f"pool: {len(pool)} vaga(s) ({pool_path})")

    try:
        vagas, checagem, seed_sorteio = sortear_com_cobertura(pool, config, seed)
    except CoberturaInsuficiente as e:
        print("\nCOBERTURA INSUFICIENTE — o plano manda completar manualmente neste ponto:")
        for b in e.checagem.get("bloqueios", []):
            print(f"  -> {b}")
        return 1

    # Seed própria para a calibração: mudar N_VAGAS_CALIBRACAO não deve
    # reembaralhar o sorteio das vagas de estudo.
    seed_calibracao = seed + 1
    ids_calibracao = selecionar_vagas_calibracao(vagas, config, seed_calibracao)

    tabela = gerar_tabela_controle(vagas, set(ids_calibracao))
    salvar_tabela_controle(tabela, saida_dir / "vagas_estudo.csv")
    salvar_pool(vagas, saida_dir / "vagas_estudo.jsonl")

    cobertura = cobertura_pool(vagas)
    registro = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "pool": str(pool_path),
        "pool_tamanho": len(pool),
        "seed_mestre": seed,
        "seed_sorteio": seed_sorteio,
        "seed_calibracao": seed_calibracao,
        "n_vagas_estudo": config.n_vagas_estudo,
        "n_vagas_calibracao": config.n_vagas_calibracao,
        "ids_calibracao": ids_calibracao,
        "checagem_cobertura": checagem,
        "cobertura": cobertura,
    }
    (saida_dir / "sorteio_vagas.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n{len(vagas)} vaga(s) sorteada(s) (seed do sorteio: {seed_sorteio})")
    print(f"cobertura: {len(cobertura['areas'])} área(s), "
          f"{len([s for s in cobertura['senioridades'] if s])} senioridade(s) — "
          f"{'OK' if checagem['ok'] else 'FALHA'}")
    print("\náreas:")
    for area, qtd in cobertura["areas"].items():
        print(f"  {qtd:3}  {area}")
    print("senioridades:")
    for senioridade, qtd in cobertura["senioridades"].items():
        print(f"  {qtd:3}  {senioridade}")

    print(f"\nvagas de calibração ({len(ids_calibracao)}, seed {seed_calibracao}):")
    por_id = {v.vaga_id: v for v in vagas}
    for vid in ids_calibracao:
        v = por_id[vid]
        print(f"  {vid}  [{v.area_tecnica} / {v.senioridade}]  {v.cargo[:60]}")

    print(f"\nsaídas em {saida_dir}: vagas_estudo.csv, vagas_estudo.jsonl, sorteio_vagas.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
