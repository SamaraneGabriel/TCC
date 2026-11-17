"""Rankings completos das 4 configurações do estudo (`context/15_PRE_REGISTRO.md` §2).

Uso:
    python scripts/eval/gerar_rankings.py [--qdrant-path CÓPIA] [--refazer]

Para cada uma das 60 vagas, ordena os 372 currículos com texto original:

  C1    busca só no vetor `perfil`
  C3    `perfil` + `skills`, fusão RRF do Qdrant (k = 2, padrão), sem o ranker
  SC    C3 + ranker (`src/matching/ranker.py`)
  BM25  texto original da vaga × texto original do currículo (`src/evaluation/plano/bm25.py`)

A vaga estruturada vem congelada de `busca_por_vaga.jsonl` (não é reextraída); só os dois
vetores da vaga são recalculados (gemini-embedding-001, determinístico). Os 13 currículos
sem texto original saem **antes** da fusão, por filtro no Qdrant.

Grava `rankings_configuracoes.jsonl` e não lê nenhuma nota: os rankings ficam congelados
antes de serem cruzados com o juiz. O Qdrant só é lido.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.embedding.factory import create_embedder  # noqa: E402
from src.embedding.templates import vaga_to_perfil_text, vaga_to_skills_text  # noqa: E402
from src.evaluation.plano.anotacao import DIRS_CURRICULOS, indexar_curriculos, texto_do_curriculo  # noqa: E402
from src.evaluation.plano.bm25 import B, K1, ranquear_bm25  # noqa: E402
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.matching.ranker import rank  # noqa: E402
from src.schemas.vaga import VagaEstruturada  # noqa: E402
from src.vectorstore.store import VEC_PERFIL, CurriculoStore  # noqa: E402

SAIDA_DIR = Path(cfg_padrao.saida_dir)
BUSCA = SAIDA_DIR / "busca_por_vaga.jsonl"
VAGAS = SAIDA_DIR / "vagas_estudo.jsonl"
SAIDA = SAIDA_DIR / "rankings_configuracoes.jsonl"
CONFIGS = ("C1", "C3", "SC", "BM25")


def _jsonl(caminho: Path) -> dict[str, dict]:
    with caminho.open(encoding="utf-8") as fh:
        return {(r := json.loads(l))["vaga_id"]: r for l in fh if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qdrant-path", default=settings.qdrant_local_path)
    ap.add_argument("--refazer", action="store_true", help="sobrescreve rankings já gravados")
    args = ap.parse_args()

    if SAIDA.exists() and not args.refazer:
        print(f"{SAIDA} já existe (congelado). Use --refazer para regerar.")
        return 1

    busca, vagas = _jsonl(BUSCA), _jsonl(VAGAS)
    store = CurriculoStore(path=args.qdrant_path)
    payloads = dict(store.listar_pontos())
    indice = indexar_curriculos(DIRS_CURRICULOS)
    textos = {cid: texto_do_curriculo(p, indice) for cid, p in payloads.items()}
    sem_texto = sorted(c for c, t in textos.items() if not t)
    com_texto = {c: t for c, t in textos.items() if t}
    n = len(com_texto)
    print(f"banco: {len(payloads)}  |  sem texto original (excluídos): {len(sem_texto)}  |  universo: {n}")

    ids_vagas = sorted(busca)
    estruturadas = {v: VagaEstruturada.model_validate(busca[v]["vaga_estruturada"]) for v in ids_vagas}

    embedder = create_embedder(settings)
    print(f"vetorizando {len(ids_vagas)} vagas ({embedder.model})...")
    vetores = {}
    for v in ids_vagas:
        e = estruturadas[v]
        skills_vec, perfil_vec = embedder.embed_batch([vaga_to_skills_text(e), vaga_to_perfil_text(e)])
        vetores[v] = (skills_vec, perfil_vec)

    bm25 = ranquear_bm25(com_texto, [(v, vagas[v]["texto"]) for v in ids_vagas])

    registros, sobreposicao = [], []
    for v in ids_vagas:
        skills_vec, perfil_vec = vetores[v]
        c1 = store.search_by(perfil_vec, using=VEC_PERFIL, limit=n, excluir_ids=sem_texto)
        c3 = store.search(skills_vec, perfil_vec, limit=n, excluir_ids=sem_texto)
        sc = rank(estruturadas[v], c3)
        rk = {
            "C1": [r.id for r in c1],
            "C3": [r.id for r in c3],
            "SC": [r.candidato.id for r in sc],
            "BM25": bm25[v],
        }
        for nome, lista in rk.items():
            assert len(lista) == n and set(lista) == set(com_texto), f"{v} {nome}: ranking incompleto"
        # Conferência: SC recalculado × top-25 congelado na amostragem (que incluía os 13).
        congelado = [c for c in busca[v]["ids_ordenados"] if c in com_texto][:25]
        sobreposicao.append(len(set(congelado) & set(rk["SC"][:25])))
        registros.append({"vaga_id": v, **rk})

    with SAIDA.open("w", encoding="utf-8") as fh:
        for r in registros:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "n_vagas": len(registros), "n_curriculos": n, "excluidos_sem_texto": sem_texto,
        "embedding_model": embedder.model, "bm25": {"k1": K1, "b": B},
        "qdrant_path": str(args.qdrant_path),
    }
    SAIDA.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ngravado: {SAIDA}")
    print(f"SC recalculado × top-25 da amostragem: sobreposição média {sum(sobreposicao)/len(sobreposicao):.1f}/25 "
          f"(mín {min(sobreposicao)}, máx {max(sobreposicao)})")
    for a, b in (("C1", "C3"), ("C3", "SC"), ("C3", "BM25"), ("C1", "BM25")):
        m = sum(len(set(r[a][:25]) & set(r[b][:25])) for r in registros) / len(registros)
        print(f"top-25 em comum {a} × {b}: {m:.1f}/25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
