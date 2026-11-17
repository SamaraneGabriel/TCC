"""Conferência da conclusão (seção de validação do juiz): nas vagas de calibração,
calcula o nDCG@25 de cada configuração com as notas do autor e com as do juiz e
compara a ordem das configurações.

Notas do autor: `anotacoes_calibracao.csv` (estratos A/B/C + estrato T, os pares do
top-25 de C1/BM25 anotados depois só para esta conferência).

Três colunas por configuração:
  autor        -> notas do autor; lista ideal sobre os pares que o autor anotou
  juiz_mesmos  -> notas do juiz nos MESMOS pares (mesmo universo da lista ideal do autor)
  juiz_censo   -> notas do juiz no banco inteiro, exatamente como no estudo principal

Pares sem nota do juiz saem do ranking de todas as configurações (mesma regra do estudo).
Grava `conferencia_ordem.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.plano.comparacao import ndcg_at_k  # noqa: E402
from src.evaluation.plano.config import ConfigAvaliacao  # noqa: E402
from src.evaluation.plano.juiz import carregar_julgamentos  # noqa: E402

SAIDA_DIR = Path(ConfigAvaliacao().saida_dir)
CONFIGS = ["C1", "C3", "SC", "BM25"]


def ordem(nd: dict[str, float]) -> list[str]:
    return sorted(nd, key=lambda c: -nd[c])


def main() -> int:
    with (SAIDA_DIR / "anotacoes_calibracao.csv").open(encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    autor: dict[str, dict[str, int]] = {}
    for l in linhas:
        if l["nota_humano_1"].strip():
            autor.setdefault(l["vaga_id"], {})[l["curriculo_id"]] = int(l["nota_humano_1"])

    with (SAIDA_DIR / "rankings_configuracoes.jsonl").open(encoding="utf-8") as fh:
        rankings = {(r := json.loads(l))["vaga_id"]: r for l in fh if l.strip()}
    julg = carregar_julgamentos(SAIDA_DIR / "julgamentos_juiz.jsonl")

    resultado = {}
    for v in sorted(autor):
        juiz_censo = {c: j.nota for (vv, c), j in julg.items() if vv == v}
        faltam = set(rankings[v]["C1"]) - set(juiz_censo)
        rk = {cfg: [c for c in rankings[v][cfg] if c not in faltam] for cfg in CONFIGS}
        # Só currículos do universo ranqueado (372) e com nota do juiz: a amostra
        # de calibração foi tirada dos 385 do índice, antes da dedup para 372.
        universo = set(rankings[v]["C1"]) - faltam
        notas_autor = {c: n for c, n in autor[v].items() if c in universo}
        juiz_mesmos = {c: juiz_censo[c] for c in notas_autor}

        nd = {
            "autor": {cfg: ndcg_at_k(rk[cfg], notas_autor) for cfg in CONFIGS},
            "juiz_mesmos": {cfg: ndcg_at_k(rk[cfg], juiz_mesmos) for cfg in CONFIGS},
            "juiz_censo": {cfg: ndcg_at_k(rk[cfg], juiz_censo) for cfg in CONFIGS},
        }
        resultado[v] = {
            "pares_autor": len(notas_autor),
            "excluidos_sem_juiz": len(faltam),
            "ndcg": {k: {c: round(x, 4) for c, x in d.items()} for k, d in nd.items()},
            "ordem": {k: ordem(d) for k, d in nd.items()},
        }

    medias = {
        k: {cfg: round(sum(resultado[v]["ndcg"][k][cfg] for v in resultado) / len(resultado), 4)
            for cfg in CONFIGS}
        for k in ("autor", "juiz_mesmos", "juiz_censo")
    }
    saida = {"por_vaga": resultado, "media_3_vagas": medias,
             "ordem_media": {k: ordem(d) for k, d in medias.items()}}
    (SAIDA_DIR / "conferencia_ordem.json").write_text(
        json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")

    for v, r in resultado.items():
        print(f"\n{v}  ({r['pares_autor']} pares do autor, {r['excluidos_sem_juiz']} excluído(s) sem juiz)")
        for k in ("autor", "juiz_mesmos", "juiz_censo"):
            vals = "  ".join(f"{c}={r['ndcg'][k][c]:.3f}" for c in CONFIGS)
            print(f"  {k:<12} {vals}   ordem: {' > '.join(r['ordem'][k])}")
    print("\nMédia nas 3 vagas")
    for k, d in medias.items():
        vals = "  ".join(f"{c}={d[c]:.3f}" for c in CONFIGS)
        print(f"  {k:<12} {vals}   ordem: {' > '.join(saida['ordem_media'][k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
