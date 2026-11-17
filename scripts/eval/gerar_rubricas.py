"""Fase 2 do plano de avaliação (§3.2): gera a rubrica de qualificação das vagas do estudo.

Uso:
    python scripts/eval/gerar_rubricas.py [--vagas data/eval/plano/vagas_estudo.jsonl]
                                          [--concurrency N] [--limite N] [--refazer]

    --concurrency N  gera N rubricas em paralelo (default 4). Um rate gate interno pacea o
                     disparo agregado, como em `scripts/ingest_curriculos.py`.
    --limite N       gera só as N primeiras vagas pendentes — para inspecionar o resultado
                     antes de rodar as 60.
    --refazer        regera vagas que já têm rubrica no CSV.

**Idempotente por padrão.** Vaga que já tem rubrica é pulada, porque a §3.2 passo 4 manda
gerar uma vez e reutilizar: regerar mudaria os itens (e portanto os `item_id`) contra os
quais anotações já podem ter sido feitas. `--refazer` existe para o caso de mudança
deliberada de prompt, e aí as anotações daquela vaga precisam ser refeitas.

**Revisão humana.** Este script só produz o rascunho (`origem=llm`). Nas 3 vagas de
calibração a §3.2 passo 2 exige revisão humana antes do uso — feita na interface de
anotação, que grava `origem=humano_revisado`. O script avisa quais faltam.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.evaluation.plano.config import config as cfg_padrao  # noqa: E402
from src.evaluation.plano.onet import BaseONetAusente, VocabularioONet  # noqa: E402
from src.evaluation.plano.rubrica import (  # noqa: E402
    GeradorRubrica,
    RubricaVaga,
    carregar_rubricas,
    conferir_rubricas,
    salvar_rubricas,
)
from src.evaluation.plano.selecao import VagaPool  # noqa: E402
from src.extraction.factory import create_extractor  # noqa: E402
from src.usage import formatar_resumo  # noqa: E402

VAGAS_PADRAO = Path("data/eval/plano/vagas_estudo.jsonl")
SORTEIO_PADRAO = Path("data/eval/plano/sorteio_vagas.json")
INTERVALO_MIN = 1.0  # s entre disparos; o Vertex aguenta bem mais que o free tier


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


def carregar_vagas(caminho: Path) -> list[VagaPool]:
    if not caminho.exists():
        raise SystemExit(
            f"{caminho} não existe — rode antes: python scripts/eval/sortear_vagas_estudo.py"
        )
    with caminho.open(encoding="utf-8") as fh:
        return [VagaPool(**json.loads(linha)) for linha in fh if linha.strip()]


def ids_de_calibracao(caminho: Path) -> set[str]:
    if not caminho.exists():
        return set()
    return set(json.loads(caminho.read_text(encoding="utf-8")).get("ids_calibracao", []))


def carregar_vocabulario() -> VocabularioONet | None:
    try:
        vocab = VocabularioONet.carregar()
    except BaseONetAusente as e:
        print(f"[AVISO] {e}")
        print("        As rubricas serão geradas sem o vocabulário de apoio da §3.1.\n")
        return None
    problemas = vocab.problemas()
    if problemas:
        print("[AVISO] mapa área->SOC com problemas:")
        for p in problemas:
            print(f"        - {p}")
    return vocab


async def _gerar_uma(vaga, gerador, gate, semaforo, resultados, erros):
    async with semaforo:
        for tentativa in range(2):
            try:
                await gate.espera()
                rubrica = await gerador.agerar(vaga)
                resultados[vaga.vaga_id] = rubrica
                obrig = len(rubrica.obrigatorios)
                print(f"[OK]    {vaga.vaga_id}  {len(rubrica.itens):2d} itens "
                      f"({obrig} obrigatórios)  {vaga.cargo[:50]}")
                for aviso in rubrica.avisos():
                    print(f"        aviso: {aviso}")
                return
            except Exception as e:
                if tentativa == 0:
                    print(f"[RETRY] {vaga.vaga_id}: {e}")
                    await asyncio.sleep(10)
                    continue
                erros.append((vaga.vaga_id, str(e)))
                print(f"[ERRO]  {vaga.vaga_id}: {e}")


async def gerar_todas(vagas, gerador, concorrencia) -> tuple[dict[str, RubricaVaga], list]:
    resultados: dict[str, RubricaVaga] = {}
    erros: list[tuple[str, str]] = []
    gate, semaforo = RateGate(INTERVALO_MIN), asyncio.Semaphore(concorrencia)
    await asyncio.gather(
        *(_gerar_uma(v, gerador, gate, semaforo, resultados, erros) for v in vagas)
    )
    return resultados, erros


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vagas", default=str(VAGAS_PADRAO))
    ap.add_argument("--sorteio", default=str(SORTEIO_PADRAO))
    ap.add_argument("--saida", default=None, help="CSV de saída (default: <saida_dir>/rubricas_vagas.csv)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()

    saida = Path(args.saida) if args.saida else Path(cfg_padrao.saida_dir) / "rubricas_vagas.csv"
    vagas = carregar_vagas(Path(args.vagas))
    calibracao = ids_de_calibracao(Path(args.sorteio))
    existentes = carregar_rubricas(saida)

    pendentes = vagas if args.refazer else [v for v in vagas if v.vaga_id not in existentes]
    if args.limite:
        pendentes = pendentes[: args.limite]

    print(f"vagas do estudo: {len(vagas)}  |  já com rubrica: {len(existentes)}  "
          f"|  a gerar agora: {len(pendentes)}")
    print(f"modelo: {settings.gemini_model} ({settings.gcp_location})  |  saída: {saida}\n")

    if pendentes:
        vocab = carregar_vocabulario()
        gerador = GeradorRubrica(create_extractor(settings), vocabulario=vocab)
        novas, erros = asyncio.run(gerar_todas(pendentes, gerador, args.concurrency))

        # Grava mesmo com erros parciais: rubrica gerada é chamada de API paga,
        # perder as que deram certo porque outra falhou seria desperdício.
        todas = {**existentes, **novas}
        salvar_rubricas([todas[v.vaga_id] for v in vagas if v.vaga_id in todas], saida)
        print(f"\n{len(novas)} rubrica(s) nova(s) gravada(s) em {saida}")
        if erros:
            print(f"{len(erros)} falha(s): {', '.join(v for v, _ in erros)}")
        print(formatar_resumo())
    else:
        todas = existentes
        print("nada a gerar (use --refazer para regerar)")

    conferencia = conferir_rubricas(todas, vagas, calibracao)
    print("\nCONFERÊNCIA (§3.2)")
    print(f"  rubricas: {conferencia['rubricas']}/{len(vagas)}  "
          f"|  revisadas por humano: {conferencia['revisadas']}/{len(calibracao)} exigidas")
    print(f"  itens: {conferencia['itens_total']} no total, "
          f"média {conferencia['itens_por_vaga_media']} por vaga "
          f"(min {conferencia['itens_por_vaga_min']}, max {conferencia['itens_por_vaga_max']})")
    for aviso in conferencia["avisos"][:10]:
        print(f"  [aviso] {aviso}")
    if len(conferencia["avisos"]) > 10:
        print(f"  ... e mais {len(conferencia['avisos']) - 10} aviso(s)")
    for bloqueio in conferencia["bloqueios"][:10]:
        print(f"  [BLOQUEIO] {bloqueio}")

    relatorio = Path(cfg_padrao.saida_dir) / "rubricas_conferencia.json"
    relatorio.write_text(json.dumps(conferencia, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nrelatório em {relatorio}")

    if conferencia["ok"]:
        print("RESULTADO: rubricas prontas para a Fase 3 (amostragem)")
        return 0
    print("RESULTADO: pendências acima antes de seguir")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
