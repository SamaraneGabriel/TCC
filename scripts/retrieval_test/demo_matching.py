"""
Demo do pipeline de matching completo (vaga -> currículos rankeados), sem UI.
Útil para verificar acurácia/funcionamento antes (ou em vez) de abrir o Streamlit.

Uso:
    python scripts/retrieval_test/demo_matching.py <arquivo_texto_vaga> [top_k]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.embedding.factory import create_embedder
from src.extraction.factory import create_extractor
from src.extraction.vaga_extractor import VagaExtractor
from src.matching.ranker import rank
from src.matching.retriever import buscar_candidatos
from src.vectorstore.store import CurriculoStore


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python scripts/retrieval_test/demo_matching.py <arquivo_texto_vaga> [top_k]")
        sys.exit(1)

    texto = Path(sys.argv[1]).read_text(encoding="utf-8")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    extractor = VagaExtractor(create_extractor(settings))
    embedder = create_embedder(settings)
    store = CurriculoStore(path=settings.qdrant_local_path)

    vaga = extractor.extract(texto)
    print(f"=== Vaga extraída: {vaga.titulo} | {vaga.senioridade} | {vaga.localidade} ===")
    print(f"Skills obrigatórias: {vaga.skills_obrigatorias}")
    print(f"Skills desejadas: {vaga.skills_desejadas}\n")

    candidatos = buscar_candidatos(vaga, embedder, store, top_k=top_k)
    ranqueados = rank(vaga, candidatos)

    print(f"=== {len(ranqueados)} candidato(s) (de {store.count()} indexados) ===")
    for r in ranqueados:
        c = r.candidato
        print(
            f"#{r.rank:<2} {c.nome:<25} score={r.score_final:.3f} "
            f"(vetor_pos={r.score_vetorial_normalizado:.2f}, skills={len(r.skills_match)}/{r.skills_total_vaga}, "
            f"senioridade={'OK' if r.senioridade_match else '-'}, local={'OK' if r.localidade_match else '-'}) "
            f"| {c.senioridade_estimada}, {c.localidade}"
        )
        print(f"     skills: {', '.join(c.skills_tecnicas)}")


if __name__ == "__main__":
    main()
