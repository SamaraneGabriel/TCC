"""Comparação C1 × C3 × SC × BM25 pelo nDCG@25 com as notas do juiz (`15_PRE_REGISTRO.md` §4–6).

Uso:
    python scripts/eval/comparar_configuracoes.py [--julgamentos ARQ]

Lê os rankings congelados (`rankings_configuracoes.jsonl`, de `gerar_rankings.py`) e o censo
do juiz (`julgamentos_juiz.jsonl`). Só entram vagas com **todos** os 372 currículos julgados
(com o censo ainda rodando, o resultado é parcial e o script avisa). Vagas com todas as notas
0 saem do teste (nDCG indefinido).

Grava `comparacao_configuracoes.json` (testes) e `ndcg_por_vaga.csv` (uma linha por vaga).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.evaluation.plano.comparacao import (  # noqa: E402
    K, N_BOOTSTRAP, SEED_BOOTSTRAP, bootstrap_ic, comparar, holm, ndcg_at_k,
)
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.evaluation.plano.juiz import carregar_julgamentos  # noqa: E402

SAIDA_DIR = Path(cfg_padrao.saida_dir)
CONFIGS = ("C1", "C3", "SC", "BM25")
PRINCIPAL = ("C3", "C1")
SECUNDARIAS = {"S1: SC × C3": ("SC", "C3"), "S2: C3 × BM25": ("C3", "BM25"), "S3: C1 × BM25": ("C1", "BM25")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--julgamentos", default=str(SAIDA_DIR / "julgamentos_juiz.jsonl"))
    ap.add_argument("--rankings", default=str(SAIDA_DIR / "rankings_configuracoes.jsonl"))
    ap.add_argument("--saida", default=str(SAIDA_DIR / "comparacao_configuracoes.json"))
    ap.add_argument("--faltantes", choices=["excluir", "incompleta", "zero", "tres"], default="excluir",
                    help="pares sem nota do juiz: excluir = o currículo sai do ranking de todas as configurações daquela vaga (padrão, desvio D10); "
                         "incompleta = a vaga inteira fica de fora; "
                         "zero/tres = imputa a nota (limites para análise de sensibilidade)")
    args = ap.parse_args()

    with open(args.rankings, encoding="utf-8") as fh:
        rankings = {(r := json.loads(l))["vaga_id"]: r for l in fh if l.strip()}
    julg = carregar_julgamentos(args.julgamentos)

    notas_por_vaga: dict[str, dict[str, int]] = {v: {} for v in rankings}
    for (v, c), j in julg.items():
        if v in notas_por_vaga:
            notas_por_vaga[v][c] = j.nota

    linhas, incompletas, sem_relevante = [], [], []
    for v, rk in sorted(rankings.items()):
        universo = set(rk["C1"])
        notas = {c: n for c, n in notas_por_vaga[v].items() if c in universo}
        faltam = universo - set(notas)
        if faltam:
            if args.faltantes == "incompleta":
                incompletas.append((v, len(notas)))
                continue
            if args.faltantes == "excluir":
                rk = {cfg: [c for c in rk[cfg] if c not in faltam] for cfg in CONFIGS}
            else:
                notas.update({c: 0 if args.faltantes == "zero" else 3 for c in faltam})
        nd = {cfg: ndcg_at_k(rk[cfg], notas) for cfg in CONFIGS}
        if nd["C1"] is None:
            sem_relevante.append(v)
            continue
        dist = Counter(notas.values())
        linhas.append({"vaga_id": v, **{cfg: round(nd[cfg], 6) for cfg in CONFIGS},
                       **{f"n_nota{g}": dist.get(g, 0) for g in range(4)}})

    total = len(rankings)
    print(f"vagas: {total}  |  completas: {total - len(incompletas)}  |  incompletas: {len(incompletas)}"
          f"  |  sem nenhuma nota > 0 (excluídas): {len(sem_relevante)}  |  no teste: {len(linhas)}")
    if incompletas:
        print("  [PARCIAL] o censo ainda não fechou estas vagas: "
              + ", ".join(f"{v} ({n})" for v, n in incompletas[:5]) + (" ..." if len(incompletas) > 5 else ""))
    if len(linhas) < 6:
        print("poucas vagas completas para testar.")
        return 1

    col = {cfg: [l[cfg] for l in linhas] for cfg in CONFIGS}
    dist_total = Counter()
    for l in linhas:
        for g in range(4):
            dist_total[g] += l[f"n_nota{g}"]
    principal = comparar(col[PRINCIPAL[0]], col[PRINCIPAL[1]])
    sec = {nome: comparar(col[a], col[b]) for nome, (a, b) in SECUNDARIAS.items()}
    for nome, p in holm({n: s["p"] for n, s in sec.items()}).items():
        sec[nome]["p_holm"] = p

    resumo_cfg = {}
    for cfg in CONFIGS:
        x = np.asarray(col[cfg])
        resumo_cfg[cfg] = {"media": float(x.mean()), "dp": float(x.std(ddof=1)), "ic95": list(bootstrap_ic(x))}

    resultado = {
        "parcial": bool(incompletas), "faltantes": args.faltantes,
        "n_vagas_total": total, "n_vagas_incompletas": len(incompletas),
        "vagas_excluidas_sem_relevante": sem_relevante, "n_vagas_teste": len(linhas),
        "k": K, "ganho": "linear", "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED_BOOTSTRAP},
        "wilcoxon": "bilateral, zero_method=pratt",
        "distribuicao_notas_juiz": {str(g): dist_total[g] for g in range(4)},
        "ndcg_por_configuracao": resumo_cfg,
        "principal_C3_menos_C1": principal,
        "secundarias": sec,
    }
    Path(args.saida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    with (Path(args.saida).parent / "ndcg_por_vaga.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    n_pares = sum(dist_total.values())
    print("\nnotas do juiz nas vagas do teste: " + ", ".join(
        f"{g}: {dist_total[g]} ({dist_total[g]/n_pares:.1%})" for g in range(4)))
    print(f"\nnDCG@{K} (ganho linear), média ± DP [IC 95%]")
    for cfg in CONFIGS:
        r = resumo_cfg[cfg]
        print(f"  {cfg:5s} {r['media']:.3f} ± {r['dp']:.3f}  [{r['ic95'][0]:.3f}; {r['ic95'][1]:.3f}]")

    def linha(nome, s):
        extra = f"  p Holm = {s['p_holm']:.4f}" if "p_holm" in s else ""
        return (f"  {nome:16s} dif = {s['diferenca_media']:+.3f}  IC95 [{s['ic95'][0]:+.3f}; {s['ic95'][1]:+.3f}]"
                f"  W = {s['W']:.1f}  p = {s['p']:.4f}{extra}  (melhor/pior/empate: "
                f"{s['melhor']}/{s['pior']}/{s['empate']})")

    print("\nprincipal")
    print(linha("C3 − C1", principal))
    print("secundárias (Holm)")
    for nome, s in sec.items():
        print(linha(nome, s))
    print(f"\ngravado: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
