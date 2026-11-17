"""Carregamento do pool de vagas reais (seção 2.1 do plano de avaliação).

Dataset (decisão do usuário em 2026-09-15): `data/vagas/kaggle/postings.csv` —
vagas reais do LinkedIn, colunas `job_title`, `company`, `job_location`,
`job_link`, `first_seen`, `search_city`, `search_country`, `job level`,
`job_type`, `job_summary`, `job_skills`. Nenhuma vaga é gerada aqui: o módulo
só lê, normaliza e amostra o que existe no arquivo.

Duas derivações são heurísticas e ficam isoladas em funções próprias, porque a
qualidade da estratificação do sorteio (seção 2.2) depende delas:

- **senioridade**: o dataset só distingue "Associate" e "Mid senior", grosso
  demais para o vocabulário do projeto ('junior'/'pleno'/'senior'). O marcador
  explícito no título ("Senior", "Staff", "Junior"...) vale mais que o campo, e
  só na ausência dele o `job level` decide.
- **área técnica**: não existe no dataset; sai de palavras-chave no título e,
  se o título for genérico ("Software Engineer"), nas skills declaradas.

Ambas guardam o valor cru (`job level`, título, skills) no JSONL do pool, então
revisar a regra depois não obriga a reprocessar nada.
"""
import csv
import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from ...dedup import hash_conteudo
from .selecao import VagaPool

# Vocabulários do projeto (ver src/extraction/prompts/vaga_prompt.txt).
MAPA_MODELO_TRABALHO = {"remote": "remoto", "hybrid": "híbrido", "onsite": "presencial"}
MAPA_JOB_LEVEL = {"associate": "pleno", "mid senior": "senior"}

PADRAO_SENIOR = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|architect|head|director|manager|iii|iv)\b", re.I
)
PADRAO_JUNIOR = re.compile(
    r"\b(junior|jr\.?|entry[- ]level|graduate|trainee|intern(ship)?|apprentice|associate)\b", re.I
)

# Ordem importa: a primeira área cujas palavras-chave casarem vence, então as
# especializações vêm antes das genéricas ("Backend" casaria com quase tudo).
AREAS_TECNICAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Embarcados/Firmware", ("embedded", "firmware", "rtos", "microcontroller", "device driver",
                             "bare metal", "fpga", "vhdl", "autosar", "can bus")),
    ("IA/ML", ("machine learning", "deep learning", "data scientist", "data science", " ai ",
               "artificial intelligence", "nlp", "computer vision", "llm", "pytorch", "tensorflow")),
    ("Dados/BI", ("data engineer", "data analyst", "business intelligence", " bi ", "etl",
                  "data warehouse", "analytics", "spark", "hadoop", "tableau", "power bi",
                  "snowflake", "databricks")),
    ("Segurança", ("security", "cybersecurity", "appsec", "infosec", "penetration",
                   "cryptograph", "vulnerability")),
    ("QA/Testes", ("qa ", "quality assurance", "test engineer", "sdet", "test automation",
                   "tester", "quality engineer")),
    ("Mobile", ("android", "ios ", "mobile", "react native", "flutter", "swift", "kotlin",
                "objective-c")),
    ("Jogos", ("game", "unity", "unreal", "gameplay")),
    ("DevOps/Infra", ("devops", "sre", "site reliability", "infrastructure", "platform engineer",
                      "cloud engineer", "kubernetes", "terraform", "ci/cd", "systems engineer")),
    ("Full-stack", ("full stack", "fullstack", "full-stack")),
    ("Frontend", ("frontend", "front-end", "front end", "ui engineer", "react", "angular",
                  "vue", "javascript", "typescript", "web developer")),
    ("Backend", ("backend", "back-end", "back end", "api", "java", "c#", ".net", "python",
                 "golang", " go ", "ruby", "php", "node", "scala", "c++", "microservices",
                 "salesforce", "sap", "erp", "database", "sql")),
    ("Gestão/Produto", ("engineering manager", "product manager", "program manager",
                        "scrum master", "product owner", "technical program")),
)

AREA_PADRAO = "Engenharia de Software (geral)"

# Cargos que a busca do dataset capturou por tangenciarem "software" mas que não
# são vagas de engenharia — julgá-los contra um corpus de currículos de TI só
# produziria ranking degenerado (nada relevante). Quase todos vêm de um mesmo
# anunciante de vagas jurídicas de patentes, repetidas cidade a cidade.
PADRAO_NAO_ENGENHARIA = re.compile(
    r"\b(patent (attorney|agent)|attorney|lawyer|paralegal|recruiter|talent acquisition|"
    r"account executive|sales (rep|representative|manager|engineer)?|marketing|nurse|"
    r"physician|teacher|professor|accountant|human resources|customer success)\b",
    re.I,
)


def modelo_trabalho(job_type: str | None) -> str | None:
    return MAPA_MODELO_TRABALHO.get((job_type or "").strip().lower())


def senioridade(cargo: str, job_level: str | None) -> str | None:
    """Marcador explícito no título tem prioridade sobre o `job level` do dataset."""
    titulo = f" {cargo or ''} "
    if PADRAO_SENIOR.search(titulo):
        return "senior"
    if PADRAO_JUNIOR.search(titulo):
        return "junior"
    return MAPA_JOB_LEVEL.get((job_level or "").strip().lower())


def area_tecnica(cargo: str, skills: str | None = None) -> str:
    """Primeiro pelo título (mais confiável), depois pelas skills declaradas."""
    for fonte in (cargo, skills):
        texto = f" {(fonte or '').lower()} ".replace("/", " / ")
        for area, palavras in AREAS_TECNICAS:
            if any(p in texto for p in palavras):
                return area
    return AREA_PADRAO


def vaga_id(link: str | None, cargo: str, empresa: str | None, indice: int) -> str:
    """ID estável entre execuções: o número do anúncio no LinkedIn quando existe,
    senão um hash curto do (título, empresa, posição no arquivo)."""
    achado = re.search(r"(\d{6,})", link or "")
    if achado:
        return f"vaga_{achado.group(1)}"
    bruto = f"{cargo}|{empresa}|{indice}".encode("utf-8")
    return f"vaga_{hashlib.sha256(bruto).hexdigest()[:12]}"


def _linha_para_vaga(linha: dict, indice: int) -> VagaPool:
    cargo = (linha.get("job_title") or "").strip()
    skills = (linha.get("job_skills") or "").strip()
    return VagaPool(
        vaga_id=vaga_id(linha.get("job_link"), cargo, linha.get("company"), indice),
        cargo=cargo,
        texto=(linha.get("job_summary") or "").strip(),
        area_tecnica=area_tecnica(cargo, skills),
        senioridade=senioridade(cargo, linha.get("job level")),
        empresa=(linha.get("company") or "").strip() or None,
        localidade=(linha.get("job_location") or "").strip() or None,
        modelo_trabalho=modelo_trabalho(linha.get("job_type")),
        skills=skills or None,
    )


def _descricao_util(vaga: VagaPool, min_chars: int) -> bool:
    """Seção 2, passo 3: descrição de requisitos não trivial. Aqui o corte é mais
    duro que "mais que uma linha" — uma vaga curta demais não dá ao LLM-juiz o
    que julgar, e entraria no estudo só para virar ruído."""
    linhas = [l for l in vaga.texto.splitlines() if l.strip()]
    return len(vaga.texto) >= min_chars and len(linhas) > 1


def carregar_postings(
    caminho: Path,
    min_chars_descricao: int = 400,
    limite: int | None = None,
    excluir_nao_engenharia: bool = True,
    deduplicar: bool = True,
) -> list[VagaPool]:
    """Lê o CSV do Kaggle e devolve as vagas utilizáveis do pool.

    Descarta, nesta ordem:

    - descrição trivial ou curta demais (seção 2, passo 3);
    - `vaga_id` repetido (o mesmo anúncio capturado por buscas de cidades diferentes);
    - cargos fora de engenharia (`PADRAO_NAO_ENGENHARIA`);
    - **anúncios com o texto idêntico a outro já aceito** — 15% do dataset é a
      mesma vaga republicada cidade a cidade (maior grupo: 37 cópias). Sortear
      duas cópias da mesma vaga quebraria a independência entre as vagas do
      estudo, que a agregação por bootstrap (seção 9) pressupõe.
    """
    csv.field_size_limit(10**9)
    vagas: list[VagaPool] = []
    ids_vistos: set[str] = set()
    textos_vistos: set[str] = set()
    with caminho.open(encoding="utf-8", errors="replace", newline="") as f:
        for indice, linha in enumerate(csv.DictReader(f)):
            vaga = _linha_para_vaga(linha, indice)
            if not vaga.cargo or vaga.vaga_id in ids_vistos:
                continue
            if not _descricao_util(vaga, min_chars_descricao):
                continue
            if excluir_nao_engenharia and PADRAO_NAO_ENGENHARIA.search(vaga.cargo):
                continue
            if deduplicar:
                h = hash_conteudo(vaga.texto)
                if h in textos_vistos:
                    continue
                textos_vistos.add(h)
            ids_vistos.add(vaga.vaga_id)
            vagas.append(vaga)
            if limite and len(vagas) >= limite:
                break
    return vagas


def amostrar_pool(vagas: list[VagaPool], n: int, seed: int) -> list[VagaPool]:
    """Recorta um pool de tamanho `n` (seção 2: "pool disponível ~5000 vagas").

    Sorteio simples e semeado: o pool é o universo do estudo, quem estratifica é
    o sorteio das vagas de estudo (`selecao.sortear_vagas_estudo`).
    """
    if n >= len(vagas):
        return list(vagas)
    return random.Random(seed).sample(vagas, n)


def cobertura_pool(vagas: Iterable[VagaPool]) -> dict:
    """Contagens por área e senioridade — o que o passo 3 da seção 2 checa depois
    no sorteio, aqui só para inspecionar o pool antes de usá-lo."""
    por_area: dict[str, int] = {}
    por_senioridade: dict[str | None, int] = {}
    total = 0
    for v in vagas:
        total += 1
        por_area[v.area_tecnica or AREA_PADRAO] = por_area.get(v.area_tecnica or AREA_PADRAO, 0) + 1
        por_senioridade[v.senioridade] = por_senioridade.get(v.senioridade, 0) + 1
    return {
        "total": total,
        "areas": dict(sorted(por_area.items(), key=lambda kv: -kv[1])),
        "senioridades": dict(sorted(por_senioridade.items(), key=lambda kv: -kv[1])),
    }


def salvar_pool(vagas: Iterable[VagaPool], caminho: Path) -> None:
    """JSONL (uma vaga por linha) — o texto das vagas tem quebras de linha e
    aspas que tornariam um CSV frágil de reler."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as f:
        for v in vagas:
            f.write(json.dumps(
                {
                    "vaga_id": v.vaga_id,
                    "cargo": v.cargo,
                    "empresa": v.empresa,
                    "localidade": v.localidade,
                    "modelo_trabalho": v.modelo_trabalho,
                    "senioridade": v.senioridade,
                    "area_tecnica": v.area_tecnica,
                    "texto": v.texto,
                    "skills": v.skills,
                },
                ensure_ascii=False,
            ) + "\n")


def carregar_pool(caminho: Path) -> list[VagaPool]:
    return list(_ler_jsonl(caminho))


def _ler_jsonl(caminho: Path) -> Iterator[VagaPool]:
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            yield VagaPool(
                vaga_id=d["vaga_id"],
                cargo=d["cargo"],
                texto=d.get("texto", ""),
                area_tecnica=d.get("area_tecnica"),
                senioridade=d.get("senioridade"),
                empresa=d.get("empresa"),
                localidade=d.get("localidade"),
                modelo_trabalho=d.get("modelo_trabalho"),
                skills=d.get("skills"),
            )
