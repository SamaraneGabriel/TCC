"""
Ingestão de currículos: lê PDFs/textos de um diretório, extrai, embeda (skills+perfil)
e indexa no Qdrant (modo local em disco, sem precisar de Docker/servidor).

Uso:
    python scripts/ingest_curriculos.py <diretorio> [--recreate]

    --recreate  apaga e recria a coleção antes de indexar (útil ao reingerir dados de teste).

Nota: o tier gratuito do Gemini limita a ~10 requisições/minuto para geração de
conteúdo (extração). O script espaça as chamadas e tenta uma vez de novo em
caso de erro (rate limit ou 503 transitório) antes de desistir de um arquivo.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.embedding.factory import create_embedder
from src.extraction.curriculo_extractor import CurriculoExtractor
from src.extraction.factory import create_extractor
from src.ingestion import ingerir_curriculo
from src.parsers.pdf_parser import parse_pdf
from src.vectorstore.store import CurriculoStore


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Uso: python scripts/ingest_curriculos.py <diretorio> [--recreate]")
        sys.exit(1)

    directory = Path(args[0])
    recreate = "--recreate" in sys.argv

    arquivos = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".pdf", ".txt"))
    if not arquivos:
        print(f"Nenhum arquivo .pdf/.txt encontrado em {directory}")
        sys.exit(1)

    extractor = CurriculoExtractor(create_extractor(settings))
    embedder = create_embedder(settings)
    store = CurriculoStore(path=settings.qdrant_local_path)
    store.create_collection(vector_size=settings.embedding_dimension, recreate=recreate)

    ok, erros = 0, []
    for i, path in enumerate(arquivos):
        if i > 0:
            time.sleep(6.5)  # fica sob o limite de ~10 req/min do free tier do Gemini

        for tentativa in range(2):
            try:
                curriculo = ingerir_curriculo(_read_text(path), extractor, embedder, store)
                ok += 1
                print(f"[OK]     {path.name} -> {curriculo.nome}")
                break
            except Exception as e:
                if tentativa == 0:
                    print(f"[RETRY]  {path.name}: {e}")
                    time.sleep(20)
                    continue
                erros.append((path.name, str(e)))
                print(f"[ERRO]   {path.name}: {e}")

    print(f"\nConcluído: {ok} indexados, {len(erros)} erros. Total na coleção: {store.count()}")


if __name__ == "__main__":
    main()
