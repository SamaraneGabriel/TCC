"""Roda o juiz LLM item a item (src/evaluation/plano/juiz.py) sobre os pares vaga-currículo.

Uso:
    python scripts/eval/rodar_juiz.py --teste 10          # teste: só vagas de PRODUÇÃO
    python scripts/eval/rodar_juiz.py                     # rodada completa: 60 vagas × todo o banco
        [--concurrency 8] [--saida ...] [--qdrant-path ...]

Modos:
  --teste N  julga N pares sorteados (seed fixa) de 2 vagas de produção e imprime os
             julgamentos para conferência de formato. Nunca toca vaga de calibração. Grava
             em `juiz_teste.jsonl`, separado da rodada completa.
  (padrão)   censo: cada uma das 60 vagas contra todos os currículos do banco. Retoma de
             onde parou (pula pares já gravados). **Não imprime nenhuma nota** — só
             contagens —, porque inclui as vagas de calibração e a §4.3 proíbe o anotador
             humano de ver a nota do LLM antes de fechar a anotação.

O Qdrant só é lido (payloads, para achar o texto original de cada currículo). Com o app
de anotação aberto a pasta fica travada: use `--qdrant-path` apontando para uma CÓPIA.
"""
import argparse
import asyncio
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.evaluation.plano.anotacao import (  # noqa: E402
    DIRS_CURRICULOS,
    indexar_curriculos,
    texto_do_curriculo,
)
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.evaluation.plano.juiz import (  # noqa: E402
    JuizError,
    JuizRubrica,
    carregar_julgamentos,
    gravar_julgamento,
)
from src.evaluation.plano.rubrica import carregar_rubricas  # noqa: E402
from src.extraction.factory import create_extractor  # noqa: E402
from src.usage import formatar_resumo  # noqa: E402
from src.vectorstore.store import CurriculoStore  # noqa: E402

SAIDA_DIR = Path(cfg_padrao.saida_dir)
VAGAS = SAIDA_DIR / "vagas_estudo.jsonl"
SORTEIO = SAIDA_DIR / "sorteio_vagas.json"
RUBRICAS = SAIDA_DIR / "rubricas_vagas.csv"
SAIDA_CENSO = SAIDA_DIR / "julgamentos_juiz.jsonl"
SAIDA_TESTE = SAIDA_DIR / "juiz_teste.jsonl"
ERROS = SAIDA_DIR / "julgamentos_juiz_erros.jsonl"

SEED_TESTE = 20261110
INTERVALO_MIN = 0.15      # s entre disparos (agregado); o semáforo limita o paralelismo
TENTATIVAS_API = 5        # erros de rede/limite de taxa; JSON inválido é tratado no juiz


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


def carregar_vagas() -> dict[str, dict]:
    with VAGAS.open(encoding="utf-8") as fh:
        return {(v := json.loads(l))["vaga_id"]: v for l in fh if l.strip()}


def ids_calibracao() -> set[str]:
    return set(json.loads(SORTEIO.read_text(encoding="utf-8")).get("ids_calibracao", []))


def pares_do_teste(vagas: dict, curriculos: list[str], n: int) -> list[tuple[str, str]]:
    rng = random.Random(SEED_TESTE)
    producao = sorted(set(vagas) - ids_calibracao())
    v1, v2 = rng.sample(producao, 2)
    metade = n // 2
    return ([(v1, c) for c in rng.sample(sorted(curriculos), n - metade)]
            + [(v2, c) for c in rng.sample(sorted(curriculos), metade)])


async def _julgar_um(par, juiz, vagas, rubricas, textos, gate, sem, saida, contagem, mostrar):
    vaga_id, cv_id = par
    texto = textos.get(cv_id, "")
    if not texto:
        contagem["sem_texto"] += 1
        _registrar_erro(vaga_id, cv_id, "texto original do currículo não encontrado")
        return
    async with sem:
        for tentativa in range(1, TENTATIVAS_API + 1):
            try:
                await gate.espera()
                j = await juiz.ajulgar(vagas[vaga_id], rubricas[vaga_id], cv_id, texto)
                break
            except JuizError as e:
                contagem["erro_json"] += 1
                _registrar_erro(vaga_id, cv_id, str(e))
                return
            except Exception as e:  # rede, 429, 503...
                if tentativa == TENTATIVAS_API:
                    contagem["erro_api"] += 1
                    _registrar_erro(vaga_id, cv_id, f"{type(e).__name__}: {e}")
                    return
                await asyncio.sleep(10 * tentativa)
    gravar_julgamento(saida, j)
    contagem["ok"] += 1
    if j.tentativas > 1:
        contagem["com_retentativa"] += 1
    if mostrar:
        _imprimir(j, vagas[vaga_id], rubricas[vaga_id])
    elif contagem["ok"] % 200 == 0:
        print(f"  {contagem['ok']} julgados  ({time.strftime('%H:%M:%S')})", flush=True)


def _registrar_erro(vaga_id, cv_id, msg):
    with ERROS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"vaga_id": vaga_id, "curriculo_id": cv_id, "erro": msg[:500],
                             "data": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")


def _imprimir(j, vaga, rubrica):
    marca = {"atende": "✔", "parcial": "~", "nao_atende": "✘"}
    print(f"\n── {vaga['cargo'][:60]}  |  currículo {j.curriculo_id}")
    print(f"   nota do juiz: {j.nota}   (regra do checklist sugeriria {j.nota_sugerida_checklist})"
          f"   tentativas: {j.tentativas}")
    for item, julg in zip(rubrica.itens, j.itens):
        tipo = "OBR" if item.obrigatorio else "des"
        ev = f'  «{julg.evidencia}»' if julg.evidencia else ""
        print(f"   {marca[julg.atendimento]} [{tipo}] {item.descricao[:70]}{ev[:90]}")
    print(f"   justificativa: {j.justificativa}")


async def rodar(pares, juiz, vagas, rubricas, textos, saida, concorrencia, mostrar):
    contagem = {"ok": 0, "com_retentativa": 0, "erro_json": 0, "erro_api": 0, "sem_texto": 0}
    gate, sem = RateGate(INTERVALO_MIN), asyncio.Semaphore(concorrencia)
    await asyncio.gather(*(
        _julgar_um(p, juiz, vagas, rubricas, textos, gate, sem, saida, contagem, mostrar) for p in pares
    ))
    return contagem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teste", type=int, default=None, help="N pares de 2 vagas de produção")
    ap.add_argument("--calibracao", action="store_true", help="só os pares anotados pelo autor")
    ap.add_argument("--juiz", choices=["gemini", "opus", "haiku", "sonnet"], default="gemini",
                    help="gemini = modelo do .env (padrão); opus = Claude Opus 5.5 via Vertex; "
                         "haiku = Claude Haiku 5.5 sem raciocínio, API da Anthropic")
    ap.add_argument("--effort", default="medium", help="esforço do Claude (só com --juiz opus)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--qdrant-path", default=settings.qdrant_local_path)
    args = ap.parse_args()

    vagas = carregar_vagas()
    rubricas = carregar_rubricas(RUBRICAS)
    payloads = dict(CurriculoStore(path=args.qdrant_path).listar_pontos())
    indice = indexar_curriculos(DIRS_CURRICULOS)
    textos = {cid: texto_do_curriculo(p, indice) for cid, p in payloads.items()}

    if args.juiz in ("haiku", "sonnet"):
        from src.extraction.providers.claude import ClaudeExtractor

        modelo = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}[args.juiz]
        llm = ClaudeExtractor(api_key=settings.anthropic_api_key, model=modelo)
    elif args.juiz == "opus":
        from src.extraction.providers.claude_vertex import ClaudeVertexExtractor

        llm = ClaudeVertexExtractor(project=settings.gcp_project, effort=args.effort)
    else:
        llm = create_extractor(settings)
    juiz = JuizRubrica(llm)
    print(f"modelo: {juiz.modelo} ({settings.gcp_location})  |  prompt sha256: {juiz.prompt_sha256[:12]}")
    print(f"vagas: {len(vagas)}  |  currículos no banco: {len(payloads)}  "
          f"|  sem texto original: {sum(1 for t in textos.values() if not t)}")

    if args.calibracao:
        # Os pares que o autor anotou. Grava no arquivo do censo: são os mesmos pares, e a
        # rodada completa depois os pula em vez de repetir.
        saida = Path(args.saida) if args.saida else SAIDA_CENSO
        feitos = carregar_julgamentos(saida)
        with (SAIDA_DIR / "anotacoes_calibracao.csv").open(encoding="utf-8") as fh:
            todos = [(l["vaga_id"], l["curriculo_id"]) for l in csv.DictReader(fh)]
        pares = [p for p in todos if p not in feitos]
        print(f"pares de calibração: {len(todos)}  |  já julgados: {len(todos) - len(pares)}  "
              f"|  a julgar agora: {len(pares)}")
        mostrar = False
    elif args.teste:
        saida = Path(args.saida) if args.saida else SAIDA_TESTE
        pares = pares_do_teste(vagas, list(payloads), args.teste)
        assert not {v for v, _ in pares} & ids_calibracao(), "teste não pode usar vaga de calibração"
        mostrar = True
    else:
        saida = Path(args.saida) if args.saida else SAIDA_CENSO
        feitos = carregar_julgamentos(saida)
        pares = [(v, c) for v in vagas for c in sorted(payloads) if (v, c) not in feitos]
        print(f"pares no censo: {len(vagas) * len(payloads)}  |  já julgados: {len(feitos)}  "
              f"|  a julgar agora: {len(pares)}")
        mostrar = False

    global ERROS
    ERROS = saida.with_name(saida.stem + "_erros.jsonl")
    print(f"saída: {saida}  |  paralelo: {args.concurrency}\n")
    inicio = time.monotonic()
    contagem = asyncio.run(rodar(pares, juiz, vagas, rubricas, textos, saida, args.concurrency, mostrar))
    print(f"\n{contagem}  em {time.monotonic() - inicio:.0f}s")
    print(formatar_resumo())
    if contagem["erro_json"] or contagem["erro_api"] or contagem["sem_texto"]:
        print(f"falhas registradas em {ERROS} — rode de novo para tentar os pares que faltaram")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
