"""
Sandbox: insercao e recuperacao com dois vetores por point (skills + perfil).

Modos:
  python sandbox/vectorstore_sandbox.py            -> in-memory (sem Docker)
  python sandbox/vectorstore_sandbox.py --docker   -> Docker em localhost:6333
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion

from src.schemas.curriculo import Certificacao, CurriculoEstruturado, Experiencia, Formacao
from src.vectorstore.store import (
    COLLECTION_NAME,
    VEC_PERFIL,
    VEC_SKILLS,
    CurriculoStore,
)

VECTOR_SIZE = 384
W = 65


# ──────────────────────────────────────────────────────────────────────────────
# CURRICULOS
# ──────────────────────────────────────────────────────────────────────────────
#
# Para cada curriculo definimos dois textos separados que serao embedados:
#
#   skills_text  -> lista de palavras-chave tecnicas
#                   (o que sera comparado com as skills da vaga)
#
#   perfil_text  -> descricao narrativa do profissional
#                   (o que sera comparado com a descricao da vaga)
#
# No pipeline real esses textos virao de templates aplicados sobre o schema.
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATOS = [
    {
        "curriculo": CurriculoEstruturado(
            nome="Maria Oliveira",
            email="maria@email.com",
            localidade="Belo Horizonte, MG",
            senioridade_estimada="pleno",
            anos_experiencia_estimados=4.5,
            area_principal="Engenharia de Software",
            skills_tecnicas=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
            modelo_trabalho_preferido="remoto",
            experiencias=[Experiencia(
                cargo="Desenvolvedora Backend", empresa="TechCorp",
                duracao_meses=24, skills_utilizadas=["Python", "FastAPI", "PostgreSQL"],
            )],
            formacoes=[Formacao(grau="graduacao", curso="Ciencia da Computacao",
                                instituicao="UFMG", ano_conclusao=2020)],
            certificacoes=[Certificacao(nome="AWS Cloud Practitioner", ano=2023)],
        ),
        # skills_text: o que essa pessoa SABe FAZER
        "skills_text": "Python FastAPI PostgreSQL Docker AWS REST API backend microsservicos",
        # perfil_text: quem essa pessoa E profissionalmente
        "perfil_text": (
            "Desenvolvedora backend pleno com 4.5 anos de experiencia em APIs REST "
            "e microsservicos usando FastAPI e PostgreSQL na TechCorp."
        ),
    },
    {
        "curriculo": CurriculoEstruturado(
            nome="Carlos Souza",
            email="carlos@email.com",
            localidade="Sao Paulo, SP",
            senioridade_estimada="senior",
            anos_experiencia_estimados=7.0,
            area_principal="Engenharia de Dados",
            skills_tecnicas=["Python", "Spark", "Kafka", "Airflow", "GCP", "BigQuery"],
            modelo_trabalho_preferido="hibrido",
        ),
        "skills_text": "Python Spark Kafka Airflow GCP BigQuery pipeline dados ETL streaming",
        "perfil_text": (
            "Engenheiro de dados senior com 7 anos construindo pipelines "
            "de dados em larga escala com Spark e Kafka no GCP."
        ),
    },
    {
        "curriculo": CurriculoEstruturado(
            nome="Ana Lima",
            email="ana@email.com",
            localidade="Curitiba, PR",
            senioridade_estimada="junior",
            anos_experiencia_estimados=1.5,
            area_principal="Desenvolvimento Frontend",
            skills_tecnicas=["JavaScript", "TypeScript", "React", "CSS", "HTML"],
            modelo_trabalho_preferido="remoto",
        ),
        "skills_text": "JavaScript TypeScript React CSS HTML frontend componentes acessibilidade",
        "perfil_text": (
            "Desenvolvedora frontend junior com 1.5 anos focada em React "
            "e acessibilidade web."
        ),
    },
    {
        "curriculo": CurriculoEstruturado(
            nome="Lucas Ferreira",
            email="lucas@email.com",
            localidade="Belo Horizonte, MG",
            senioridade_estimada="pleno",
            anos_experiencia_estimados=3.0,
            area_principal="Engenharia de Software",
            skills_tecnicas=["Python", "Django", "PostgreSQL", "Redis", "RabbitMQ"],
            modelo_trabalho_preferido="remoto",
        ),
        # Lucas conhece Python/PostgreSQL (skills parecidas com a vaga),
        # mas trabalha com Django e pagamentos (perfil diferente da vaga)
        "skills_text": "Python Django PostgreSQL Redis RabbitMQ backend pagamento API",
        "perfil_text": (
            "Desenvolvedor backend pleno com 3 anos trabalhando em sistemas "
            "de pagamento com Django e integracoes via RabbitMQ."
        ),
    },
]

# A vaga que sera usada como consulta
VAGA = {
    "titulo": "Desenvolvedor(a) Backend Python - Pleno",
    "descricao": (
        "Buscamos dev backend pleno para construir e manter APIs REST "
        "com FastAPI, integracao com bancos de dados e deploy em cloud AWS."
    ),
    # skills_text da vaga: skills obrigatorias + desejadas
    "skills_text": "Python FastAPI PostgreSQL Docker AWS REST API backend",
    # perfil_text da vaga: descricao da posicao e responsabilidades
    "perfil_text": (
        "Posicao de backend pleno para desenvolvimento de APIs REST com FastAPI "
        "e PostgreSQL, deploy em AWS, trabalho em equipe agil, 3+ anos de experiencia."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS SIMULADOS
# ──────────────────────────────────────────────────────────────────────────────

def _seed(text: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(text))


def fake_embedding(text: str) -> list[float]:
    """
    Gera embedding deterministico baseado nas palavras do texto.
    Textos com mais palavras em comum ficam mais proximos no espaco vetorial.
    No pipeline real, substitua por: model.encode(text).tolist()
    """
    words = set(text.lower().split())
    rng = random.Random(_seed(text))
    vec = [rng.gauss(0, 0.3) for _ in range(VECTOR_SIZE)]

    for word in words:
        wrng = random.Random(_seed(word))
        for _ in range(12):
            dim = wrng.randint(0, VECTOR_SIZE - 1)
            vec[dim] += wrng.gauss(2.5, 0.5)

    norm = sum(x ** 2 for x in vec) ** 0.5
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vetores ja sao unitarios


# ──────────────────────────────────────────────────────────────────────────────
# FORMATACAO
# ──────────────────────────────────────────────────────────────────────────────

def header(step: int, title: str) -> None:
    print()
    print("=" * W)
    print(f"  ETAPA {step}: {title}")
    print("=" * W)


def sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def row(label: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}  {label:<28}  {value}")


def sep() -> None:
    print("  " + "-" * (W - 2))


def vec_preview(v: list[float], n: int = 5) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v[:n]) + f", ...]  dim={len(v)}"


def score_bar(score: float, scale: float = 1.0) -> str:
    filled = max(0, int(score / scale * 24))
    return "[" + "#" * filled + "-" * (24 - filled) + f"]  {score:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# SANDBOX
# ──────────────────────────────────────────────────────────────────────────────

def run(docker: bool) -> None:

    # ── ETAPA 1: Conexao ──────────────────────────────────────────────────────
    header(1, "CONEXAO COM O QDRANT")
    if docker:
        row("Modo", "Docker  (localhost:6333)")
        row("Dashboard", "http://localhost:6333/dashboard")
        raw_client = QdrantClient(host="localhost", port=6333)
    else:
        row("Modo", "in-memory  (sem Docker necessario)")
        raw_client = QdrantClient(":memory:")
    store = CurriculoStore.__new__(CurriculoStore)
    store.client = raw_client
    print("\n  Conexao OK.")

    # ── ETAPA 2: Criar collection ─────────────────────────────────────────────
    header(2, "CRIAR COLLECTION COM DOIS VETORES NOMEADOS")
    row("Collection", COLLECTION_NAME)
    row("Vetor 1", f'"{VEC_SKILLS}"  ->  skills tecnicas  (dim={VECTOR_SIZE}, cosine)')
    row("Vetor 2", f'"{VEC_PERFIL}"  ->  perfil narrativo  (dim={VECTOR_SIZE}, cosine)')
    store.create_collection(vector_size=VECTOR_SIZE, recreate=True)
    print("\n  Collection criada.")

    # ── ETAPA 3: Gerar embeddings e inspecionar similaridades ─────────────────
    header(3, "GERAR EMBEDDINGS E INSPECIONAR SIMILARIDADES")
    print()
    print("  Cada curriculo gera DOIS vetores a partir de textos distintos.")
    print("  O vetor de skills captura o que o candidato sabe fazer.")
    print("  O vetor de perfil captura quem ele e profissionalmente.")
    print()

    vaga_skills_vec = fake_embedding(VAGA["skills_text"])
    vaga_perfil_vec = fake_embedding(VAGA["perfil_text"])

    row("Vaga — skills_text", VAGA["skills_text"])
    row("Vaga — perfil_text", VAGA["perfil_text"][:55] + "...")
    print()

    all_vecs: dict[str, tuple[list[float], list[float]]] = {}

    sub("Similaridade cosine de cada candidato com a vaga")
    print()
    print(f"  {'Candidato':<22}  {'skills sim':>11}  {'perfil sim':>11}  {'media':>9}")
    print("  " + "-" * (W - 2))
    for item in CANDIDATOS:
        c = item["curriculo"]
        sv = fake_embedding(item["skills_text"])
        pv = fake_embedding(item["perfil_text"])
        all_vecs[c.nome] = (sv, pv)
        s_sim = cosine(sv, vaga_skills_vec)
        p_sim = cosine(pv, vaga_perfil_vec)
        avg = (s_sim + p_sim) / 2
        print(f"  {c.nome:<22}  {s_sim:>+.4f}     {p_sim:>+.4f}     {avg:>+.4f}")

    print()
    print("  Observe: Lucas tem skills similares (Python/PostgreSQL) mas perfil")
    print("  diferente (pagamentos/Django vs APIs REST/FastAPI). A fusao vai")
    print("  capturar essa diferenca melhor do que um unico vetor.")

    # ── ETAPA 4: Inserir curriculos ───────────────────────────────────────────
    header(4, "INSERIR CURRICULOS (dois vetores por point)")
    inserted_ids: dict[str, str] = {}
    for item in CANDIDATOS:
        c = item["curriculo"]
        sv, pv = all_vecs[c.nome]
        cid = store.upsert(c, skills_vector=sv, perfil_vector=pv)
        inserted_ids[c.nome] = cid
        print()
        sep()
        row("Nome",              c.nome)
        row("ID",                cid[:18] + "...")
        row("Senioridade",       c.senioridade_estimada or "-")
        row("Skills",            ", ".join(c.skills_tecnicas[:5]))
        row("Modelo",            c.modelo_trabalho_preferido or "-")
        row(f'vec["{VEC_SKILLS}"]', vec_preview(sv))
        row(f'vec["{VEC_PERFIL}"]', vec_preview(pv))
    sep()
    print(f"\n  Total na collection: {store.count()} curriculo(s).")

    # ── ETAPA 5: Busca por vetor individual — skills ──────────────────────────
    header(5, f'BUSCA POR VETOR INDIVIDUAL — using="{VEC_SKILLS}"')
    print()
    print("  Rankeia somente pela proximidade das skills tecnicas.")
    print("  Candidatos com mais skills em comum com a vaga aparecem primeiro.\n")
    res_skills = store.search_by(vaga_skills_vec, using=VEC_SKILLS, limit=4)
    for rank, r in enumerate(res_skills, 1):
        print(f"  #{rank}  {r.nome:<22}  score={r.score:+.4f}  skills={r.skills_tecnicas[:3]}")

    # ── ETAPA 6: Busca por vetor individual — perfil ──────────────────────────
    header(6, f'BUSCA POR VETOR INDIVIDUAL — using="{VEC_PERFIL}"')
    print()
    print("  Rankeia somente pela proximidade do perfil narrativo.")
    print("  Candidatos com descricao de carreira similar a da vaga aparecem primeiro.\n")
    res_perfil = store.search_by(vaga_perfil_vec, using=VEC_PERFIL, limit=4)
    for rank, r in enumerate(res_perfil, 1):
        print(f"  #{rank}  {r.nome:<22}  score={r.score:+.4f}  seniority={r.senioridade_estimada}")

    # ── ETAPA 7: Busca com fusao RRF ──────────────────────────────────────────
    header(7, "BUSCA COM FUSAO RRF (Reciprocal Rank Fusion)")
    print()
    print("  Combina os dois rankings anteriores sem precisar definir pesos.")
    print("  Um candidato bom em AMBOS os vetores sobe; bom em apenas um desce.")
    print("  O score RRF e 1/(k + rank) somado, nao e cosine diretamente.\n")

    res_rrf = store.search(vaga_skills_vec, vaga_perfil_vec, limit=4, fusion=Fusion.RRF)
    max_score = res_rrf[0].score if res_rrf else 1.0
    for rank, r in enumerate(res_rrf, 1):
        bar = score_bar(r.score, scale=max_score)
        print(f"  #{rank}  {r.nome:<22}  {bar}")
    print()

    # mostrar se a ordem mudou em relacao aos individuais
    order_skills = [r.nome for r in res_skills]
    order_perfil = [r.nome for r in res_perfil]
    order_fused  = [r.nome for r in res_rrf]
    sub("Comparacao de rankings")
    print(f"  skills only:  {order_skills}")
    print(f"  perfil only:  {order_perfil}")
    print(f"  RRF fused:    {order_fused}")
    if order_fused != order_skills or order_fused != order_perfil:
        print()
        print("  -> A fusao produziu uma ordem diferente dos rankings individuais.")

    # ── ETAPA 8: Busca com fusao DBSF ────────────────────────────────────────
    header(8, "BUSCA COM FUSAO DBSF (Distribution-Based Score Fusion)")
    print()
    print("  Normaliza a distribuicao de scores de cada vetor antes de combinar.")
    print("  Util quando os dois vetores produzem escalas de score muito diferentes.\n")

    res_dbsf = store.search(vaga_skills_vec, vaga_perfil_vec, limit=4, fusion=Fusion.DBSF)
    max_dbsf = res_dbsf[0].score if res_dbsf else 1.0
    for rank, r in enumerate(res_dbsf, 1):
        bar = score_bar(r.score, scale=max_dbsf)
        print(f"  #{rank}  {r.nome:<22}  {bar}")

    # ── ETAPA 9: Fusao + filtro de metadados ─────────────────────────────────
    header(9, "FUSAO + FILTRO DE METADADOS — pleno + remoto")
    print()
    print("  O filtro e aplicado ANTES da busca vetorial em cada prefetch.")
    print("  So candidatos pleno+remoto entram no pool de candidatos.\n")

    res_filtered = store.search(
        vaga_skills_vec, vaga_perfil_vec,
        limit=4,
        senioridade="pleno",
        modelo_trabalho="remoto",
        fusion=Fusion.RRF,
    )
    print(f"  {len(res_filtered)} resultado(s) (de {store.count()} total):\n")
    for rank, r in enumerate(res_filtered, 1):
        sep()
        row(f"  #{rank} Nome",      r.nome)
        row("    Score RRF",        f"{r.score:.4f}")
        row("    Senioridade",      r.senioridade_estimada or "-")
        row("    Modelo trabalho",  r.modelo_trabalho_preferido or "-")
        row("    Skills",           ", ".join(r.skills_tecnicas[:4]))
    sep()

    # ── ETAPA 10: Delete ──────────────────────────────────────────────────────
    header(10, "DELETAR UM CURRICULO")
    alvo = "Ana Lima"
    print(f"\n  Deletando: {alvo}  (id={inserted_ids[alvo][:18]}...)")
    print(f"  Antes:  {store.count()} curriculo(s)")
    store.delete(inserted_ids[alvo])
    print(f"  Depois: {store.count()} curriculo(s)")

    res_pos = store.search(vaga_skills_vec, vaga_perfil_vec, limit=4)
    nomes = [r.nome for r in res_pos]
    assert alvo not in nomes
    print(f"  Resultados apos delete: {nomes}")
    print(f"  Confirmado: '{alvo}' removido.")

    # ── Fim ───────────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  SANDBOX CONCLUIDO — todos os passos executados com sucesso.")
    print("=" * W)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()
    run(docker=args.docker)
