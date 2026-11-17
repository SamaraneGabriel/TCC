"""
Fase 0 do plano de avaliação: confirmar os parâmetros e os pré-requisitos de
engenharia antes de sortear qualquer vaga.

Checa, sem gastar nenhuma chamada de API (o probe de busca usa um vetor
pseudoaleatório determinístico, não um embedding real):

  1. consistência interna da config (seção 1);
  2. `N_TOTAL` declarado x currículos realmente indexados;
  3. que o sistema aceita `limit=TOP_POOL` e devolve `min(TOP_POOL, N)` ids distintos;
  4. que dá pra enumerar todos os ids do banco — sem isso não existe universo do
     estrato C, que é definido por complemento de conjunto (seção 4.2 passo 4);
  5. que o pool de vagas existe, tem tamanho suficiente pra sortear
     `N_VAGAS_ESTUDO` e cobre o mínimo de áreas/senioridades da seção 2.

Sai com código != 0 se qualquer bloqueio for encontrado — a Fase 1 não deve rodar
antes disso passar.

Uso:
    python scripts/eval/fase0_confirmar_parametros.py [--vagas-dir DIR] [--qdrant-path DIR]
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.evaluation.plano.config import ConfigAvaliacao
from src.evaluation.plano.pool_vagas import carregar_pool, cobertura_pool
from src.vectorstore.store import COLLECTION_NAME, CurriculoStore

EXTENSOES_VAGA = (".txt", ".json", ".md")
SEED_PROBE = 0  # o probe só precisa ser determinístico, não representativo


def _vetor_probe(dimensao: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0, 1) for _ in range(dimensao)]


def checar_config(cfg: ConfigAvaliacao) -> dict:
    return {
        "nome": "config consistente",
        "ok": not cfg.problemas(),
        "bloqueios": cfg.problemas(),
        "avisos": cfg.avisos(),
    }


def checar_n_total(cfg: ConfigAvaliacao, store: CurriculoStore) -> dict:
    try:
        indexados = store.count()
    except Exception as e:
        return {
            "nome": "N_TOTAL confere com o banco",
            "ok": False,
            "bloqueios": [f"não consegui contar a coleção {COLLECTION_NAME!r}: {e}"],
            "indexados": None,
        }

    bloqueios = []
    if indexados != cfg.n_total:
        bloqueios.append(
            f"N_TOTAL={cfg.n_total} mas há {indexados} currículo(s) indexado(s) — "
            f"ajuste AVAL_N_TOTAL ou complete a ingestão (diferença: {cfg.n_total - indexados})"
        )
    if indexados <= cfg.top_pool:
        bloqueios.append(
            f"{indexados} indexado(s) <= TOP_POOL={cfg.top_pool}: o estrato C fica vazio, "
            "o sistema devolveria o banco inteiro e o Recall@K estimado perde sentido"
        )
    return {
        "nome": "N_TOTAL confere com o banco",
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "indexados": indexados,
        "n_total_declarado": cfg.n_total,
    }


def checar_limit_top_pool(cfg: ConfigAvaliacao, store: CurriculoStore, indexados: int | None) -> dict:
    rng = random.Random(SEED_PROBE)
    dim = settings.embedding_dimension
    try:
        resultados = store.search(
            _vetor_probe(dim, rng), _vetor_probe(dim, rng), limit=cfg.top_pool
        )
    except Exception as e:
        return {
            "nome": f"sistema aceita limit={cfg.top_pool}",
            "ok": False,
            "bloqueios": [f"busca com limit={cfg.top_pool} levantou exceção: {e}"],
            "retornados": None,
        }

    ids_distintos = len({r.id for r in resultados})
    esperado = min(cfg.top_pool, indexados) if indexados is not None else len(resultados)
    bloqueios = []
    if ids_distintos != len(resultados):
        bloqueios.append(
            f"busca devolveu {len(resultados)} itens mas só {ids_distintos} ids distintos — "
            "duplicata quebraria a amostragem por estrato"
        )
    if indexados is not None and len(resultados) != esperado:
        bloqueios.append(
            f"limit={cfg.top_pool} devolveu {len(resultados)}, esperado {esperado} "
            f"(min(TOP_POOL, {indexados})) — provável truncamento no prefetch da fusão"
        )
    return {
        "nome": f"sistema aceita limit={cfg.top_pool}",
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "retornados": len(resultados),
        "ids_distintos": ids_distintos,
        "esperado": esperado,
    }


def checar_enumeracao_ids(cfg: ConfigAvaliacao, store: CurriculoStore, indexados: int | None) -> dict:
    """O universo do estrato C é `{todos os ids} - {os TOP_POOL recuperados}`,
    então enumerar o banco inteiro é pré-requisito, não conveniência."""
    ids: set[str] = set()
    offset = None
    try:
        while True:
            pontos, offset = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(p.id) for p in pontos)
            if offset is None:
                break
    except Exception as e:
        return {
            "nome": "ids do banco enumeráveis (universo do estrato C)",
            "ok": False,
            "bloqueios": [f"scroll falhou: {e}"],
            "ids_enumerados": len(ids),
        }

    bloqueios = []
    if indexados is not None and len(ids) != indexados:
        bloqueios.append(
            f"scroll enumerou {len(ids)} ids mas count() diz {indexados} — "
            "universo do estrato C ficaria incompleto"
        )
    return {
        "nome": "ids do banco enumeráveis (universo do estrato C)",
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "ids_enumerados": len(ids),
    }


def checar_pool_vagas_jsonl(cfg: ConfigAvaliacao, caminho: Path) -> dict:
    """Pool montado por `scripts/eval/preparar_pool_vagas.py` (seção 2.1)."""
    try:
        pool = carregar_pool(caminho)
    except Exception as e:
        return {
            "nome": "pool de vagas disponível",
            "ok": False,
            "bloqueios": [f"não consegui ler {caminho}: {e}"],
            "vagas_encontradas": 0,
            "arquivo": str(caminho),
        }

    cobertura = cobertura_pool(pool)
    bloqueios = []
    if len(pool) < cfg.n_vagas_estudo:
        bloqueios.append(
            f"{len(pool)} vaga(s) no pool, menos que N_VAGAS_ESTUDO={cfg.n_vagas_estudo}"
        )
    if len(cobertura["areas"]) < cfg.min_areas_tecnicas:
        bloqueios.append(
            f"{len(cobertura['areas'])} área(s) técnica(s) no pool, "
            f"mínimo {cfg.min_areas_tecnicas} (seção 2, passo 3)"
        )
    if len([s for s in cobertura["senioridades"] if s]) < cfg.min_senioridades:
        bloqueios.append(f"senioridades distintas no pool abaixo de {cfg.min_senioridades}")
    return {
        "nome": "pool de vagas disponível",
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "vagas_encontradas": len(pool),
        "arquivo": str(caminho),
        "areas": cobertura["areas"],
        "senioridades": {str(k): v for k, v in cobertura["senioridades"].items()},
    }


def checar_pool_vagas(cfg: ConfigAvaliacao, vagas_dir: Path | None) -> dict:
    if vagas_dir is None:
        return {
            "nome": "pool de vagas disponível",
            "ok": False,
            "bloqueios": [
                "nenhum --vagas-dir informado e nenhum pool de vagas encontrado no repositório; "
                f"a seção 2 do plano pressupõe ~5000 vagas reais de onde sortear "
                f"N_VAGAS_ESTUDO={cfg.n_vagas_estudo}"
            ],
            "vagas_encontradas": 0,
            "diretorio": None,
        }

    if not vagas_dir.is_dir():
        return {
            "nome": "pool de vagas disponível",
            "ok": False,
            "bloqueios": [f"{vagas_dir} não existe ou não é um diretório"],
            "vagas_encontradas": 0,
            "diretorio": str(vagas_dir),
        }

    arquivos = [p for p in vagas_dir.rglob("*") if p.suffix.lower() in EXTENSOES_VAGA]
    bloqueios = []
    if len(arquivos) < cfg.n_vagas_estudo:
        bloqueios.append(
            f"{len(arquivos)} vaga(s) em {vagas_dir}, menos que N_VAGAS_ESTUDO={cfg.n_vagas_estudo}"
        )
    return {
        "nome": "pool de vagas disponível",
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "vagas_encontradas": len(arquivos),
        "diretorio": str(vagas_dir),
    }


def _derivados(cfg: ConfigAvaliacao) -> dict:
    return {
        "n_calibracao_por_grupo": cfg.n_calibracao_por_grupo,
        "n_calibracao_total": cfg.n_calibracao_total,
        "n_double_label": cfg.n_double_label,
        "n_universo_b": cfg.n_universo_b,
        "n_b": cfg.n_b,
        "n_universo_c": cfg.n_universo_c,
        "n_c": cfg.n_c,
        "itens_por_vaga": cfg.itens_por_vaga,
        "itens_no_estudo": cfg.itens_por_vaga * cfg.n_vagas_estudo,
        "itens_anotados_por_humano": cfg.n_calibracao_total,
        "itens_double_label": cfg.n_double_label,
    }


def _abrir_store(path: str) -> tuple[CurriculoStore | None, str | None]:
    """O Qdrant em modo local trava a pasta com um lock exclusivo — uma ingestão
    em andamento impede qualquer leitura. Vira uma checagem reprovada com
    mensagem clara, não um stack trace."""
    try:
        return CurriculoStore(path=path), None
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            return None, (
                f"storage {path} está travado por outro processo (ingestão em andamento?) — "
                "espere terminar, ou rode contra uma cópia do storage"
            )
        return None, f"não consegui abrir o storage {path}: {e}"
    except Exception as e:
        return None, f"não consegui abrir o storage {path}: {e}"


def _checks_sem_store(cfg: ConfigAvaliacao, erro: str) -> list[dict]:
    nomes = [
        "N_TOTAL confere com o banco",
        f"sistema aceita limit={cfg.top_pool}",
        "ids do banco enumeráveis (universo do estrato C)",
    ]
    return [{"nome": n, "ok": False, "bloqueios": [erro]} for n in nomes]


def _achar_vagas_dir(explicito: str | None) -> Path | None:
    if explicito:
        return Path(explicito)
    for candidato in (Path("data/vagas"), Path("data/vagas_pool")):
        if candidato.is_dir():
            return candidato
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 0 do plano de avaliação")
    parser.add_argument("--vagas-dir", help="diretório com o pool de vagas (.txt/.json/.md)")
    parser.add_argument(
        "--pool-vagas",
        default="data/eval/plano/pool_vagas.jsonl",
        help="JSONL do pool montado por scripts/eval/preparar_pool_vagas.py",
    )
    parser.add_argument(
        "--qdrant-path",
        default=settings.qdrant_local_path,
        help="storage local do Qdrant (default: config do projeto)",
    )
    parser.add_argument("--saida", help="onde escrever o JSON (default: <saida_dir>/fase0_confirmacao.json)")
    args = parser.parse_args()

    cfg = ConfigAvaliacao()
    store, erro_store = _abrir_store(args.qdrant_path)

    checks = [checar_config(cfg)]
    if store is None:
        checks += _checks_sem_store(cfg, erro_store)
    else:
        check_n = checar_n_total(cfg, store)
        indexados = check_n.get("indexados")
        checks += [
            check_n,
            checar_limit_top_pool(cfg, store, indexados),
            checar_enumeracao_ids(cfg, store, indexados),
        ]
    pool_jsonl = Path(args.pool_vagas)
    if pool_jsonl.exists() and not args.vagas_dir:
        checks.append(checar_pool_vagas_jsonl(cfg, pool_jsonl))
    else:
        checks.append(checar_pool_vagas(cfg, _achar_vagas_dir(args.vagas_dir)))

    derivados = _derivados(cfg)
    aprovado = all(c["ok"] for c in checks)

    print("=" * 72)
    print("FASE 0 — confirmação de parâmetros e pré-requisitos")
    print("=" * 72)
    print(f"\nPARÂMETROS   K={cfg.k}  N_TOTAL={cfg.n_total}  TOP_POOL={cfg.top_pool}  "
          f"N_VAGAS_ESTUDO={cfg.n_vagas_estudo}  N_VAGAS_CALIBRACAO={cfg.n_vagas_calibracao}")
    print(f"DERIVADOS    calibração {derivados['n_calibracao_por_grupo']}/grupo "
          f"({derivados['n_calibracao_total']} no total, {derivados['n_double_label']} double-label)")
    print(f"             estrato B: universo {derivados['n_universo_b']}, amostra {derivados['n_b']}")
    print(f"             estrato C: universo {derivados['n_universo_c']}, amostra {derivados['n_c']}")
    print(f"             {derivados['itens_por_vaga']} itens/vaga x {cfg.n_vagas_estudo} vagas = "
          f"{derivados['itens_no_estudo']} julgamentos")

    print("\nCHECAGENS")
    for c in checks:
        print(f"  [{'OK ' if c['ok'] else 'FALHA'}] {c['nome']}")
        for b in c["bloqueios"]:
            print(f"          -> {b}")
        for a in c.get("avisos", []):
            print(f"          (aviso) {a}")

    print(f"\nRESULTADO: {'pode seguir para a Fase 1' if aprovado else 'BLOQUEADO'}")

    saida = Path(args.saida) if args.saida else Path(cfg.saida_dir) / "fase0_confirmacao.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(
        json.dumps(
            {
                "data": datetime.now().isoformat(timespec="seconds"),
                "aprovado": aprovado,
                "parametros": cfg.model_dump(),
                "derivados": derivados,
                "checagens": checks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Relatório: {saida}")
    return 0 if aprovado else 1


if __name__ == "__main__":
    sys.exit(main())
