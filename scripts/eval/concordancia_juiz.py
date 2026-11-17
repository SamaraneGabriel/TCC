"""Concordância juiz LLM × autor nos pares de calibração (`context/15_PRE_REGISTRO.md` §7).

Uso:
    python scripts/eval/concordancia_juiz.py

Lê `anotacoes_calibracao.csv` (nota_humano_1) e `julgamentos_juiz.jsonl`, cruza pelos pares
que têm as duas notas e grava `concordancia_juiz.json`. Relata o total e, como descrição,
o recorte por estrato (A = top-25 do sistema; B = posições 26-100; C = fora do top-100):
o pré-registro pede o erro do juiz no topo.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.plano.concordancia import resumo  # noqa: E402
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.evaluation.plano.juiz import carregar_julgamentos  # noqa: E402

SAIDA_DIR = Path(cfg_padrao.saida_dir)


def main() -> int:
    with (SAIDA_DIR / "anotacoes_calibracao.csv").open(encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--julgamentos", default=str(SAIDA_DIR / "julgamentos_juiz.jsonl"))
    ap.add_argument("--saida", default=str(SAIDA_DIR / "concordancia_juiz.json"))
    args = ap.parse_args()
    juiz = carregar_julgamentos(args.julgamentos)

    pares, sem_juiz, sem_humano = [], 0, 0
    for l in linhas:
        # Estrato "T" = pares do top-25 de C1/BM25 anotados depois, só para a
        # conferência da ordem das configurações; fora da amostra pré-especificada.
        if l["estrato"] not in ("A", "B", "C"):
            continue
        chave = (l["vaga_id"], l["curriculo_id"])
        if not l["nota_humano_1"].strip():
            sem_humano += 1
        elif chave not in juiz:
            sem_juiz += 1
        else:
            pares.append((l["estrato"], l["vaga_id"], int(l["nota_humano_1"]), juiz[chave].nota))

    total = resumo([p[2] for p in pares], [p[3] for p in pares])
    por_estrato = {
        e: resumo([p[2] for p in pares if p[0] == e], [p[3] for p in pares if p[0] == e])
        for e in sorted({p[0] for p in pares})
    }
    por_vaga = {
        v: resumo([p[2] for p in pares if p[1] == v], [p[3] for p in pares if p[1] == v])
        for v in sorted({p[1] for p in pares})
    }
    resultado = {
        "pares_usados": len(pares),
        "fora_sem_nota_do_juiz": sem_juiz,
        "fora_sem_nota_humana": sem_humano,
        "total": total,
        "por_estrato": por_estrato,
        "por_vaga": por_vaga,
    }
    arq = Path(args.saida)
    arq.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")

    def linha(nome, r):
        b1, b2 = r["binarizado"][">=1"], r["binarizado"][">=2"]
        return (f"{nome:<18} n={r['pares']:>3}  exata={r['concordancia_exata']:.2f}  "
                f"k_quad={r['kappa_quadratico']:.3f}  k(>=1)={b1['kappa']:.3f}  "
                f"k(>=2)={b2['kappa']:.3f}  erro_medio={r['erro_medio_juiz_menos_humano']:+.2f}")

    print(f"pares: {len(pares)}  (fora: {sem_juiz} sem nota do juiz, {sem_humano} sem nota humana)\n")
    print(linha("TOTAL", total))
    for e, r in por_estrato.items():
        print(linha(f"estrato {e}", r))
    for v, r in por_vaga.items():
        print(linha(v, r))
    print("\nmatriz de confusão (linhas = você 0..3, colunas = juiz 0..3):")
    for nota, row in zip(range(4), total["matriz_confusao"]):
        print(f"  {nota}: {row}")
    print(f"\ngravado em {arq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
