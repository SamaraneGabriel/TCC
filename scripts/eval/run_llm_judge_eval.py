"""
Roda o pipeline de avaliação via LLM-as-judge: para cada vaga numa pasta,
busca candidatos na base atual, pede pro judge avaliar relevância + sugerir
uma ordem própria, calcula métricas objetivas e salva tudo em data/eval/runs/.

Uso:
    python scripts/eval/run_llm_judge_eval.py [pasta_de_vagas] [top_k]

    pasta_de_vagas  default: tests/fixtures/synthetic/vagas/
    top_k           default: 10
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.embedding.factory import create_embedder
from src.evaluation.judge import GeminiJudge
from src.evaluation.metrics import grau_medio, pct_relevantes, spearman_rank_correlation
from src.evaluation.runs import novo_run_dir, salvar_json
from src.extraction.factory import create_extractor
from src.extraction.vaga_extractor import VagaExtractor
from src.matching.ranker import rank
from src.matching.retriever import buscar_candidatos
from src.vectorstore.store import CurriculoStore


def avaliar_vaga(texto_vaga: str, vaga_extractor, embedder, store, judge, top_k: int) -> dict:
    vaga = vaga_extractor.extract(texto_vaga)

    candidatos = buscar_candidatos(vaga, embedder, store, top_k=top_k)
    ranqueados = rank(vaga, candidatos)
    ordem_sistema = [r.candidato.id for r in ranqueados]

    julgamento = judge.julgar(vaga, [r.candidato for r in ranqueados])

    graus_na_ordem = [julgamento.graus.get(id_, 0) for id_ in ordem_sistema]
    correlacao = None
    if julgamento.ordem_sugerida:
        try:
            correlacao = spearman_rank_correlation(ordem_sistema, julgamento.ordem_sugerida)
        except ValueError:
            correlacao = None  # judge nao cobriu exatamente os mesmos ids -- sinaliza, nao quebra o run

    pasta = novo_run_dir(vaga.titulo)
    salvar_json(pasta, "vaga.json", {"texto": texto_vaga, "extraido": vaga.model_dump()})
    salvar_json(
        pasta,
        "resultados.json",
        [
            {"id": r.candidato.id, "nome": r.candidato.nome, "rank": r.rank, "score_final": r.score_final}
            for r in ranqueados
        ],
    )
    salvar_json(
        pasta,
        "julgamento.json",
        {
            "graus": julgamento.graus,
            "justificativas": julgamento.justificativas,
            "ordem_sugerida": julgamento.ordem_sugerida,
        },
    )
    salvar_json(pasta, "usage.json", julgamento.usage)

    metricas = {
        "pct_relevantes": pct_relevantes(graus_na_ordem),
        "grau_medio": grau_medio(graus_na_ordem),
        "correlacao_ordem": correlacao,
        "n_candidatos": len(ranqueados),
    }
    salvar_json(pasta, "metricas.json", metricas)

    return {"vaga": vaga.titulo, "pasta": str(pasta), **metricas, "tokens": julgamento.usage["total_tokens"]}


def main() -> None:
    pasta_vagas = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/synthetic/vagas")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    arquivos = sorted(pasta_vagas.glob("*.txt"))
    if not arquivos:
        print(f"Nenhuma vaga .txt encontrada em {pasta_vagas}")
        sys.exit(1)

    vaga_extractor = VagaExtractor(create_extractor(settings))
    embedder = create_embedder(settings)
    store = CurriculoStore(path=settings.qdrant_local_path)
    judge = GeminiJudge(api_key=settings.gemini_api_key)

    resumo = []
    for i, arquivo in enumerate(arquivos):
        if i > 0:
            time.sleep(6.5)  # fica sob o limite de ~10 req/min do free tier do Gemini

        print(f"\n=== Avaliando: {arquivo.name} ===")
        texto = arquivo.read_text(encoding="utf-8")
        try:
            resultado = avaliar_vaga(texto, vaga_extractor, embedder, store, judge, top_k)
            resumo.append(resultado)
            print(
                f"  % relevantes: {resultado['pct_relevantes']:.0%} | "
                f"grau médio: {resultado['grau_medio']:.2f} | "
                f"correlação de ordem: {resultado['correlacao_ordem']}"
            )
        except Exception as e:
            print(f"  [ERRO] {e}")

    print("\n=== Resumo ===")
    for r in resumo:
        print(
            f"{r['vaga']:<40} pct_relevantes={r['pct_relevantes']:.0%}  "
            f"grau_medio={r['grau_medio']:.2f}  correlacao={r['correlacao_ordem']}  tokens={r['tokens']}"
        )


if __name__ == "__main__":
    main()
