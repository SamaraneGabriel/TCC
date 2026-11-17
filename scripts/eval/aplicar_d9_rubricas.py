"""Desvio D9: remove das rubricas de produção os requisitos não verificáveis num currículo.

Uso:
    python scripts/eval/aplicar_d9_rubricas.py [--decisoes data/eval/plano/d9_decisoes.csv]
                                               [--aplicar]

Lê `d9_decisoes.csv`, em que cada linha é um item obrigatório com coluna `decisao`
preenchida à mão (`remover` ou `manter`). Sem `--aplicar`, só mostra o que faria.

Recusa rodar se:
  - alguma linha estiver sem decisão (a regra é do anotador, não do script);
  - o backup `rubricas_vagas.pre_d9.csv` não existir (é a única cópia do rascunho
    original do LLM, necessária para auditar o que a regra mudou);
  - algum item pertencer a vaga de calibração (essas são revisadas na interface);
  - já existir nota ou checklist do LLM para alguma vaga afetada (remover item
    depois de anotar invalidaria o checklist daquela vaga).

Grava `d9_itens_removidos.csv` com o registro completo do que saiu.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.plano.rubrica import (  # noqa: E402
    CAMPOS_RUBRICA,
    carregar_rubricas,
    remover_itens,
    salvar_rubricas,
)

SAIDA = Path("data/eval/plano")
RUBRICAS = SAIDA / "rubricas_vagas.csv"
BACKUP = SAIDA / "rubricas_vagas.pre_d9.csv"
REMOVIDOS = SAIDA / "d9_itens_removidos.csv"
DECISOES_VALIDAS = {"remover", "manter"}


def _ler_csv(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisoes", type=Path, default=SAIDA / "d9_decisoes.csv")
    ap.add_argument("--aplicar", action="store_true", help="grava as rubricas (default: só mostra)")
    args = ap.parse_args()

    if not BACKUP.exists():
        print(f"ERRO: backup {BACKUP} não existe — é a única cópia do rascunho original do LLM.")
        return 1

    decisoes = _ler_csv(args.decisoes)
    if not decisoes:
        print(f"ERRO: {args.decisoes} vazio ou inexistente.")
        return 1
    pendentes = [d["item_id"] for d in decisoes if d["decisao"].strip().lower() not in DECISOES_VALIDAS]
    if pendentes:
        print(f"ERRO: {len(pendentes)} linha(s) sem decisão válida (remover/manter): {', '.join(pendentes)}")
        return 1

    remover = {d["item_id"] for d in decisoes if d["decisao"].strip().lower() == "remover"}
    vagas_afetadas = {i.rsplit("#", 1)[0] for i in remover}

    calibracao = {
        v["vaga_id"] for v in _ler_csv(SAIDA / "vagas_estudo.csv") if v["is_calibracao"] == "True"
    }
    if vagas_afetadas & calibracao:
        print(f"ERRO: itens de vaga de calibração na lista: {sorted(vagas_afetadas & calibracao)} "
              "— essas são revisadas na interface.")
        return 1

    ja_julgadas = {
        l["vaga_id"] for l in _ler_csv(SAIDA / "anotacoes_producao.csv")
        if l["nota_llm_1"] or l["nota_llm_2"]
    } | {l["vaga_id"] for l in _ler_csv(SAIDA / "checklist_rubrica.csv")}
    if vagas_afetadas & ja_julgadas:
        print(f"ERRO: vagas afetadas já têm julgamento: {sorted(vagas_afetadas & ja_julgadas)}")
        return 1

    rubricas = carregar_rubricas(RUBRICAS)
    novas = remover_itens(rubricas, remover)

    por_id = {i.item_id: (r, i) for r in rubricas.values() for i in r.itens}
    linhas_removidas = [
        {
            "vaga_id": por_id[i][0].vaga_id,
            "item_id": i,
            "descricao": por_id[i][1].descricao,
            "tipo": por_id[i][1].tipo,
            "categoria": por_id[i][1].categoria,
            "origem": por_id[i][0].origem,
            "data_geracao": por_id[i][0].data_geracao,
        }
        for i in sorted(remover)
    ]

    antes = sum(len(r.itens) for r in rubricas.values())
    depois = sum(len(r.itens) for r in novas.values())
    print(f"Decisões: {len(decisoes)} ({len(remover)} remover, {len(decisoes) - len(remover)} manter)")
    print(f"Vagas afetadas: {len(vagas_afetadas)}")
    print(f"Itens nas rubricas: {antes} -> {depois}")
    for l in linhas_removidas:
        print(f"  - {l['item_id']}: {l['descricao']}")

    if not args.aplicar:
        print("\n(dry-run) nada gravado. Rode com --aplicar para gravar.")
        return 0

    salvar_rubricas(list(novas.values()), RUBRICAS)
    with REMOVIDOS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_RUBRICA)
        w.writeheader()
        w.writerows(linhas_removidas)
    print(f"\nGravado: {RUBRICAS} e {REMOVIDOS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
