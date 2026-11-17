"""
Ingestão de currículos: lê PDFs/textos de um diretório, extrai, embeda (skills+perfil)
e indexa no Qdrant (modo local em disco, sem precisar de Docker/servidor).

Uso:
    python scripts/ingest_curriculos.py <diretorio> [--recreate] [--concurrency N] [--reprocessar]

    --recreate      apaga e recria a coleção antes de indexar (⚠️ ver a regra no topo do
                    CLAUDE.md: exige autorização explícita do usuário no momento).
    --reprocessar   reextrai currículos já indexados (por padrão eles são pulados —
                    o ID do ponto vem do conteúdo, então reingerir só sobrescreveria).
    --concurrency N processa N currículos em paralelo (default: 1, sequencial — comportamento
                    original). Útil quando o gargalo é latência/retry de rede, não o rate limit
                    em si — um "rate gate" interno pacea o disparo agregado de chamadas pra ficar
                    perto do limite gratuito do Gemini (~10 req/min) mesmo com N > 1 concorrentes.

Nota: o tier gratuito do Gemini limita a ~10 requisições/minuto para geração de
conteúdo (extração) e embedding. Cada currículo gasta 2 chamadas (1 extração +
1 embedding em lote dos dois vetores skills+perfil). O script pacea as chamadas
e tenta uma vez de novo em caso de erro (rate limit ou 503 transitório) antes de
desistir de um arquivo.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.dedup import hash_conteudo
from src.embedding.factory import create_embedder
from src.extraction.curriculo_extractor import CurriculoExtractor
from src.extraction.factory import create_extractor
from src.ingestion import aingerir_curriculo, ingerir_curriculo
from src.parsers.pdf_parser import parse_pdf
from src.usage import formatar_resumo
from src.vectorstore.store import CurriculoStore


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    return path.read_text(encoding="utf-8")


class RateGate:
    """Pacea a taxa AGREGADA de disparo de chamadas (não o nº de chamadas em
    voo) — mantém o throughput perto do rate limit gratuito mesmo com várias
    tarefas concorrentes, em vez de cada tarefa dormir de forma independente."""

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


async def _processar_um(item, extractor, embedder, store, gate, semaforo, ok_erros):
    path, texto = item
    async with semaforo:
        for tentativa in range(2):
            try:
                await gate.espera()
                curriculo = await aingerir_curriculo(texto, extractor, embedder, store, arquivo_fonte=path.name)
                ok_erros["ok"] += 1
                print(f"[OK]     {path.name} -> {curriculo.nome or 'Nome não informado'}")
                return
            except Exception as e:
                if tentativa == 0:
                    print(f"[RETRY]  {path.name}: {e}")
                    await asyncio.sleep(15)
                    continue
                ok_erros["erros"].append((path.name, str(e)))
                print(f"[ERRO]   {path.name}: {e}")


async def _main_concorrente(arquivos, extractor, embedder, store, concurrency: int) -> tuple[int, list]:
    # 2 chamadas/currículo (extração + embedding em lote); pacea pra ~9 req/min agregado.
    gate = RateGate(min_interval=(60 / 9) / 2)
    semaforo = asyncio.Semaphore(concurrency)
    ok_erros = {"ok": 0, "erros": []}
    await asyncio.gather(
        *(_processar_um(p, extractor, embedder, store, gate, semaforo, ok_erros) for p in arquivos)
    )
    return ok_erros["ok"], ok_erros["erros"]


def _main_sequencial(arquivos, extractor, embedder, store) -> tuple[int, list]:
    ok, erros = 0, []
    for i, (path, texto) in enumerate(arquivos):
        if i > 0:
            time.sleep(6.5)  # fica sob o limite de ~10 req/min do free tier do Gemini
        for tentativa in range(2):
            try:
                curriculo = ingerir_curriculo(texto, extractor, embedder, store, arquivo_fonte=path.name)
                ok += 1
                print(f"[OK]     {path.name} -> {curriculo.nome or 'Nome não informado'}")
                break
            except Exception as e:
                if tentativa == 0:
                    print(f"[RETRY]  {path.name}: {e}")
                    time.sleep(20)
                    continue
                erros.append((path.name, str(e)))
                print(f"[ERRO]   {path.name}: {e}")
    return ok, erros


def _preparar(arquivos: list[Path], store: CurriculoStore, reprocessar: bool) -> list[tuple[Path, str]]:
    """Lê os textos e descarta o que não precisa ir pro LLM: cópias do mesmo
    conteúdo dentro do próprio lote e currículos já indexados.

    O dataset repete arquivos (ver `src/dedup.py`); sem esse filtro cada cópia
    gastaria 2 chamadas de API para acabar sobrescrevendo o mesmo ponto.
    """
    ja_indexados: set[str] = set()
    if not reprocessar:
        ja_indexados = {
            payload.get("conteudo_hash")
            for _, payload in store.listar_pontos()
            if payload.get("conteudo_hash")
        }

    selecionados: list[tuple[Path, str]] = []
    vistos: set[str] = set()
    repetidos = existentes = 0
    for path in arquivos:
        texto = _read_text(path)
        h = hash_conteudo(texto)
        if h in vistos:
            repetidos += 1
            continue
        if h in ja_indexados:
            existentes += 1
            continue
        vistos.add(h)
        selecionados.append((path, texto))

    if repetidos or existentes:
        print(
            f"Filtrados: {repetidos} cópia(s) dentro do lote, "
            f"{existentes} já indexado(s) (use --reprocessar para reextrair)"
        )
    return selecionados


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(
            "Uso: python scripts/ingest_curriculos.py <diretorio> "
            "[--recreate] [--concurrency N] [--reprocessar]"
        )
        sys.exit(1)

    directory = Path(args[0])
    recreate = "--recreate" in sys.argv
    reprocessar = "--reprocessar" in sys.argv
    concurrency = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 1

    encontrados = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".pdf", ".txt"))
    if not encontrados:
        print(f"Nenhum arquivo .pdf/.txt encontrado em {directory}")
        sys.exit(1)

    extractor = CurriculoExtractor(create_extractor(settings))
    embedder = create_embedder(settings)
    store = CurriculoStore(path=settings.qdrant_local_path)
    store.create_collection(vector_size=settings.embedding_dimension, recreate=recreate)

    arquivos = _preparar(encontrados, store, reprocessar)
    print(f"{len(arquivos)} de {len(encontrados)} arquivo(s) a processar")
    if not arquivos:
        print(f"Nada a fazer. Total na coleção: {store.count()}")
        return

    if concurrency > 1:
        ok, erros = asyncio.run(_main_concorrente(arquivos, extractor, embedder, store, concurrency))
    else:
        ok, erros = _main_sequencial(arquivos, extractor, embedder, store)

    print(f"\nConcluído: {ok} indexados, {len(erros)} erros. Total na coleção: {store.count()}")
    print(formatar_resumo())


if __name__ == "__main__":
    main()
