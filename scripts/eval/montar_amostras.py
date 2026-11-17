"""Fases 3 e 4 do plano de avaliação (§5 e §4.2): amostragem estratificada por vaga.

Uso:
    python scripts/eval/montar_amostras.py [--vagas data/eval/plano/vagas_estudo.jsonl]
                                           [--concurrency N] [--limite N] [--refazer]

Para cada vaga do estudo:
  1. extrai a `VagaEstruturada` do anúncio e gera os dois vetores (skills e perfil);
  2. consulta o sistema completo — busca por fusão RRF + re-ranking — com `limit=TOP_POOL`;
  3. estrato A = censo das K primeiras posições; B = amostra de `n_B` entre K+1 e TOP_POOL;
     C = amostra de `n_C` no complemento (banco inteiro menos o TOP_POOL);
  4. embaralha os três numa lista única, sem estrato/posição/score visíveis;
  5. separa, nas vagas de calibração, os grupos AB e C da amostra de calibração (§4.2), com o
     estrato A de uma delas (sorteada pela seed mestre) inteiro no grupo AB; depois completa o
     estrato A das demais, para que o top-K de todas as vagas de calibração seja anotado inteiro.

Saídas em `data/eval/plano/`:
    busca_por_vaga.jsonl        vaga estruturada + os TOP_POOL ids na ordem do sistema
    anotacoes_producao.csv      todas as vagas, papel=producao_llm (§7.1)
    anotacoes_calibracao.csv    os 194 itens das vagas de calibração, papel=calibracao
    log_reprodutibilidade.csv   seeds por vaga (§7.2)
    amostragem_resumo.json      contagens e parâmetros usados

**Determinismo.** As três seeds de cada vaga saem de `seeds_da_vaga(vaga_id, seed_mestre)`,
SHA-256 do par — rodar de novo com a mesma seed mestre e o mesmo banco reproduz exatamente as
mesmas amostras. O que *não* é determinístico é a extração da vaga pelo LLM; por isso o
passo 1 é persistido em `busca_por_vaga.jsonl` e reaproveitado em execuções seguintes, a
menos que se passe `--refazer`.

**Ordem das chamadas.** Extração e embedding (API) rodam concorrentes; a consulta ao Qdrant
roda sequencial depois, porque o modo local embarcado trabalha sobre um lock exclusivo de
pasta e não se beneficia de paralelismo.
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.embedding.factory import create_embedder  # noqa: E402
from src.embedding.templates import vaga_to_perfil_text, vaga_to_skills_text  # noqa: E402
from src.evaluation.plano.amostragem import (  # noqa: E402
    LogReprodutibilidade,
    estrato_a,
    estrato_b,
    estrato_c,
    gerar_linhas_anotacao,
    juntar_estratos,
    salvar_log_reprodutibilidade,
    salvar_planilha_anotacao,
    seeds_da_vaga,
    completar_censo_estrato_a,
    escolher_vaga_censo_a,
    separar_amostra_calibracao,
)
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.evaluation.plano.rubrica import carregar_rubricas  # noqa: E402
from src.evaluation.plano.selecao import CoberturaInsuficiente, VagaPool  # noqa: E402
from src.extraction.factory import create_extractor  # noqa: E402
from src.extraction.vaga_extractor import VagaExtractor  # noqa: E402
from src.matching.ranker import rank  # noqa: E402
from src.schemas.vaga import VagaEstruturada  # noqa: E402
from src.usage import formatar_resumo  # noqa: E402
from src.vectorstore.store import CurriculoStore  # noqa: E402

SAIDA = Path(cfg_padrao.saida_dir)
VAGAS_PADRAO = SAIDA / "vagas_estudo.jsonl"
SORTEIO_PADRAO = SAIDA / "sorteio_vagas.json"
BUSCA_PADRAO = SAIDA / "busca_por_vaga.jsonl"
INTERVALO_MIN = 1.0


class RateGate:
    """Pacea a taxa agregada de disparo (mesma ideia de scripts/ingest_curriculos.py)."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def espera(self) -> None:
        async with self._lock:
            agora = time.monotonic()
            espera = max(0.0, self._next_allowed - agora)
            self._next_allowed = max(agora, self._next_allowed) + self._min_interval
        if espera:
            await asyncio.sleep(espera)


# ----------------------------------------------------------------------
# Passo 1: vaga -> estruturada + vetores (API, concorrente)


async def _preparar_uma(vaga, extrator, embedder, gate, semaforo, prontas, erros):
    async with semaforo:
        for tentativa in range(2):
            try:
                await gate.espera()
                estruturada = await extrator.aextract(vaga.texto)
                await gate.espera()
                skills_vec = await asyncio.to_thread(embedder.embed, vaga_to_skills_text(estruturada))
                perfil_vec = await asyncio.to_thread(embedder.embed, vaga_to_perfil_text(estruturada))
                prontas[vaga.vaga_id] = (estruturada, skills_vec, perfil_vec)
                titulo = (estruturada.titulo or vaga.cargo or "(sem título)")[:45]
                print(f"[OK]    {vaga.vaga_id}  {titulo:45s} "
                      f"{len(estruturada.skills_obrigatorias)} skills obrig.")
                return
            except Exception as e:
                if tentativa == 0:
                    print(f"[RETRY] {vaga.vaga_id}: {e}")
                    await asyncio.sleep(10)
                    continue
                erros.append((vaga.vaga_id, str(e)))
                print(f"[ERRO]  {vaga.vaga_id}: {e}")


async def preparar_vagas(vagas, extrator, embedder, concorrencia):
    prontas, erros = {}, []
    gate, semaforo = RateGate(INTERVALO_MIN), asyncio.Semaphore(concorrencia)
    await asyncio.gather(
        *(_preparar_uma(v, extrator, embedder, gate, semaforo, prontas, erros) for v in vagas)
    )
    return prontas, erros


# ----------------------------------------------------------------------
# Persistência do passo 1 (para não repetir extração entre execuções)


def salvar_busca(caminho: Path, registros: dict[str, dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as fh:
        for vaga_id, reg in registros.items():
            fh.write(json.dumps({"vaga_id": vaga_id, **reg}, ensure_ascii=False) + "\n")


def carregar_busca(caminho: Path) -> dict[str, dict]:
    if not caminho.exists():
        return {}
    registros = {}
    with caminho.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                reg = json.loads(linha)
                registros[reg.pop("vaga_id")] = reg
    return registros


def carregar_vagas(caminho: Path) -> list[VagaPool]:
    if not caminho.exists():
        raise SystemExit(
            f"{caminho} não existe — rode antes: python scripts/eval/sortear_vagas_estudo.py"
        )
    with caminho.open(encoding="utf-8") as fh:
        return [VagaPool(**json.loads(l)) for l in fh if l.strip()]


# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vagas", default=str(VAGAS_PADRAO))
    ap.add_argument("--sorteio", default=str(SORTEIO_PADRAO))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limite", type=int, default=None, help="processa só as N primeiras vagas")
    ap.add_argument("--refazer", action="store_true", help="reextrai vagas já em busca_por_vaga.jsonl")
    args = ap.parse_args()

    cfg = cfg_padrao
    problemas = cfg.problemas()
    if problemas:
        print("configuração inconsistente:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    vagas = carregar_vagas(Path(args.vagas))
    if args.limite:
        vagas = vagas[: args.limite]
    ids_calibracao = set(
        json.loads(Path(args.sorteio).read_text(encoding="utf-8")).get("ids_calibracao", [])
    ) & {v.vaga_id for v in vagas}

    print(f"vagas: {len(vagas)}  |  calibração: {len(ids_calibracao)}")
    print(f"K={cfg.k}  TOP_POOL={cfg.top_pool}  n_B={cfg.n_b}  n_C={cfg.n_c}  "
          f"-> {cfg.itens_por_vaga} itens/vaga")
    print(f"modelo: {settings.gemini_model} ({settings.gcp_location})\n")

    rubricas = carregar_rubricas(SAIDA / "rubricas_vagas.csv")
    sem_rubrica = [v.vaga_id for v in vagas if v.vaga_id not in rubricas]
    if sem_rubrica:
        # Não bloqueia: amostrar não depende da rubrica, anotar depende (§5.2 passo 2).
        print(f"[AVISO] {len(sem_rubrica)} vaga(s) sem rubrica — rode scripts/eval/gerar_rubricas.py "
              f"antes de anotar\n")

    # --- passo 1: extrair + vetorizar (só o que falta) ---
    cache = {} if args.refazer else carregar_busca(BUSCA_PADRAO)
    pendentes = [v for v in vagas if v.vaga_id not in cache]
    prontas = {}
    if pendentes:
        print(f"extraindo e vetorizando {len(pendentes)} vaga(s)...")
        extrator = VagaExtractor(create_extractor(settings))
        embedder = create_embedder(settings)
        prontas, erros = asyncio.run(preparar_vagas(pendentes, extrator, embedder, args.concurrency))
        if erros:
            print(f"\n{len(erros)} vaga(s) falharam na extração e ficam de fora desta execução.")
        print()
    else:
        print("todas as vagas já extraídas (use --refazer para reextrair)\n")

    # --- passo 2: consultar o sistema (Qdrant local: sequencial) ---
    store = CurriculoStore(path=settings.qdrant_local_path)
    ids_banco = set(store.listar_ids())
    if len(ids_banco) != cfg.n_total:
        print(f"[AVISO] banco tem {len(ids_banco)} currículos, config diz N_TOTAL={cfg.n_total}. "
              f"Ajuste AVAL_N_TOTAL antes de rodar para valer.\n")

    registros = dict(cache)
    for vaga_id, (estruturada, skills_vec, perfil_vec) in prontas.items():
        resultados = store.search(skills_vec, perfil_vec, limit=cfg.top_pool)
        ordenado = [r.candidato.id for r in rank(estruturada, resultados)]
        registros[vaga_id] = {
            "vaga_estruturada": estruturada.model_dump(),
            "ids_ordenados": ordenado,
            "data_busca": datetime.now().isoformat(timespec="seconds"),
            "top_pool": cfg.top_pool,
        }
    if prontas:
        salvar_busca(BUSCA_PADRAO, registros)

    # --- passo 3: estratos, por vaga ---
    itens_por_vaga, log, curtas = {}, [], []
    for vaga in vagas:
        reg = registros.get(vaga.vaga_id)
        if not reg:
            continue
        ordenado = reg["ids_ordenados"]
        if len(ordenado) < cfg.top_pool:
            curtas.append((vaga.vaga_id, len(ordenado)))

        seed_b, seed_c, seed_shuffle = seeds_da_vaga(vaga.vaga_id, cfg.seed_mestre)
        try:
            itens_a = estrato_a(ordenado, cfg.k)
            itens_b = estrato_b(ordenado, cfg.k, cfg.n_b, seed_b)
            itens_c = estrato_c(ids_banco, set(ordenado), cfg.n_c, seed_c)
        except ValueError as e:
            print(f"[ERRO]  {vaga.vaga_id}: {e}")
            continue

        # Ordem interna (com estrato) para a amostra de calibração; a lista
        # embaralhada é o que vira planilha.
        itens_por_vaga[vaga.vaga_id] = [*itens_a, *itens_b, *itens_c]
        embaralhados = juntar_estratos(itens_a, itens_b, itens_c, seed_shuffle)
        registros[vaga.vaga_id]["itens_embaralhados"] = [i.curriculo_id for i in embaralhados]

        log.append(
            LogReprodutibilidade(
                vaga_id=vaga.vaga_id,
                seed_estrato_b=seed_b,
                seed_estrato_c=seed_c,
                seed_embaralhamento=seed_shuffle,
                data_coleta=reg["data_busca"],
                k_usado=cfg.k,
                n_total_usado=len(ids_banco),
                is_calibracao=vaga.vaga_id in ids_calibracao,
            )
        )

    if curtas:
        print(f"[AVISO] {len(curtas)} vaga(s) com menos de TOP_POOL resultados: "
              f"{', '.join(f'{v}({n})' for v, n in curtas[:5])}\n")

    # --- passo 4: planilhas ---
    linhas_producao = [
        linha
        for vaga_id, itens in itens_por_vaga.items()
        for linha in gerar_linhas_anotacao(
            vaga_id,
            juntar_estratos(
                [i for i in itens if i.estrato == "A"],
                [i for i in itens if i.estrato == "B"],
                [i for i in itens if i.estrato == "C"],
                seeds_da_vaga(vaga_id, cfg.seed_mestre)[2],
            ),
            "producao_llm",
        )
    ]
    salvar_planilha_anotacao(linhas_producao, SAIDA / "anotacoes_producao.csv")
    salvar_log_reprodutibilidade(log, SAIDA / "log_reprodutibilidade.csv")
    salvar_busca(BUSCA_PADRAO, registros)

    resumo = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "parametros": {
            "k": cfg.k, "top_pool": cfg.top_pool, "n_b": cfg.n_b, "n_c": cfg.n_c,
            "n_total_config": cfg.n_total, "n_total_banco": len(ids_banco),
            "seed_mestre": cfg.seed_mestre,
        },
        "vagas_processadas": len(itens_por_vaga),
        "itens_producao": len(linhas_producao),
        "vagas_com_pool_incompleto": [{"vaga_id": v, "recuperados": n} for v, n in curtas],
    }

    # --- passo 5: amostra de calibração (§4.2) ---
    if ids_calibracao:
        try:
            vaga_censo = escolher_vaga_censo_a(ids_calibracao, cfg.seed_mestre)
            ab, c = separar_amostra_calibracao(
                itens_por_vaga, ids_calibracao, cfg, cfg.seed_mestre, vaga_censo_a=vaga_censo
            )
            ab = completar_censo_estrato_a(ab, itens_por_vaga, ids_calibracao)
            salvar_planilha_anotacao(ab + c, SAIDA / "anotacoes_calibracao.csv")
            resumo["calibracao"] = {
                "grupo_ab": len(ab),
                "grupo_c": len(c),
                "total": len(ab) + len(c),
                "substrato_a": sum(1 for l in ab if l.estrato == "A"),
                "substrato_b": sum(1 for l in ab if l.estrato == "B"),
                "vagas": sorted(ids_calibracao),
                "vaga_censo_a": vaga_censo,
                "censo_estrato_a_completo": True,
            }
            print(f"amostra de calibração: {len(ab)} AB "
                  f"({resumo['calibracao']['substrato_a']} de A + "
                  f"{resumo['calibracao']['substrato_b']} de B) + {len(c)} C = {len(ab) + len(c)}")
        except CoberturaInsuficiente as e:
            resumo["calibracao"] = {"erro": e.checagem["bloqueios"]}
            print("[ERRO] amostra de calibração não fecha:")
            for b in e.checagem["bloqueios"]:
                print(f"  - {b}")

    (SAIDA / "amostragem_resumo.json").write_text(
        json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{len(itens_por_vaga)} vaga(s) amostrada(s), {len(linhas_producao)} itens de produção")
    print(f"saídas em {SAIDA}: anotacoes_producao.csv, anotacoes_calibracao.csv, "
          f"log_reprodutibilidade.csv, busca_por_vaga.jsonl, amostragem_resumo.json")
    print(formatar_resumo())
    return 0 if len(itens_por_vaga) == len(vagas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
