"""
Converte uma amostra do dataset Hugging Face Mehyaar/Annotated_NER_PDF_Resumes
(um JSON por currículo, campo "text" com o texto bruto extraído de PDF) em
arquivos .txt, no formato que scripts/ingest_curriculos.py já processa.

Teste exploratório: o dataset é bem heterogêneo (RH, engenharia civil, logística,
TI, etc.) e o texto vem cru de extração de PDF (espaçamento quebrado, alguns
arquivos vazios). Para montar uma amostra útil para testar matching contra uma
vaga de TI, filtra por currículos que mencionam vocabulário técnico e pega os
N com mais ocorrências — mantém o texto real, só prioriza quais arquivos usar.

O texto bruto vem com frases quebradas em várias linhas curtas (wrap do PDF) e,
nas seções de experiência/formação, campos como "Company Name / City / State"
espalhados em blocos minúsculos separados por linha em branco. `reflow()` re-
agrupa isso em parágrafos legíveis: junta linhas que claramente são a mesma
frase/rótulo quebrada, mas respeita quebras reais (linha em branco na fonte +
o texto acumulado já parece uma unidade completa) e isola cabeçalhos de seção
conhecidos (Summary, Skills, Experience, Education...) e itens numerados
("1. Projeto X") em parágrafos próprios. Também corrige mojibake comum
(UTF-8 decodificado como cp1252) e caracteres de formatação invisíveis.

Uso:
    python scripts/convert_meyhaar_resumes.py [--limit N]

Lê:        data/hugging_face/meyhaar/ResumesJsonAnnotated/*.json
Escreve em: data/curriculos_meyhaar/curriculo_meyhaar_NNNN.txt
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

ENTRADA = Path("data/hugging_face/meyhaar/ResumesJsonAnnotated")
SAIDA = Path("data/curriculos_meyhaar")

VOCAB_TECH = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "aws", "azure",
    "gcp", "docker", "kubernetes", "machine learning", "deep learning",
    "data scien", "software engineer", "software developer", "programming",
    "database", " api ", "react", "angular", "node", "c++", "c#", ".net", "php",
    "golang", "tensorflow", "pytorch", "nlp", "natural language processing",
    "computer vision", "etl", "spark", "hadoop", "linux", " git ", "devops",
    "cloud computing", "html", "css", "backend", "front end", "frontend",
    "full stack", "data analyst", "business intelligence", "tableau",
    "power bi", "mongodb", "postgresql", "mysql", "sql server", "android",
    "ios developer", "qa engineer", "automation testing", "agile", "scrum",
    "data structures", "algorithms", "django", "flask", "rest api", "microservices",
]

HEADERS = {
    "summary", "professional summary", "career objective", "objective",
    "highlights", "key skills", "skills", "technical skills",
    "core competencies", "experience", "professional experience",
    "work experience", "work history", "employment history", "education",
    "education and training", "academic qualifications", "qualification",
    "qualifications", "certifications", "certification", "licenses",
    "interests", "hobbies", "affiliations", "activities", "projects",
    "accomplishments", "achievements", "additional information",
    "languages", "publications", "awards", "references", "profile",
    "personal details", "declaration", "training",
    "volunteer experience", "extracurricular activities", "strengths",
    "key strengths", "areas of expertise",
}

FRAGMENT_MAX_PALAVRAS = 10
LIDER_PONTUACAO = re.compile(r"^([,.;:!?)\]}\-]+)\s*(.*)$", re.S)
DECORACAO = re.compile(r"^[\s_\-=*:]+|[\s_\-=*:]+$")
ITEM_NUMERADO = re.compile(r"^\d{1,3}\s*[.)]\s*\S")


def fix_mojibake(texto: str) -> str:
    """Corrige o caso comum de UTF-8 decodificado como cp1252 (ex.: 'Â',
    'ï¼'), fazendo o caminho inverso; se o texto já estiver correto, o
    round-trip falha ou piora e é descartado."""
    try:
        reparado = texto.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto
    if reparado.count("�") > texto.count("�"):
        return texto
    return reparado


def normalizar_base(texto: str) -> str:
    texto = fix_mojibake(texto)
    texto = unicodedata.normalize("NFKC", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Cf")
    texto = texto.replace("\xa0", " ").replace("\t", " ")
    return re.sub(r" {2,}", " ", texto)


def limpar_decoracao(linha: str) -> str:
    return DECORACAO.sub("", linha)


def eh_header(linha_decorada: str) -> bool:
    return linha_decorada.lower() in HEADERS


def eh_fragmento(bloco: str) -> bool:
    palavras = bloco.split()
    return len(palavras) <= FRAGMENT_MAX_PALAVRAS and not re.search(r"[.!?]$", bloco)


def juntar(a: str, b: str) -> str:
    """Junta duas partes de texto; anexa pontuação líder de `b` direto (sem
    espaço) em vez de deixar ', State' ou '. NET' com espaço sobrando."""
    b = b.strip()
    if not b:
        return a
    if not a:
        return b
    m = LIDER_PONTUACAO.match(b)
    if m and m.group(1):
        resto = m.group(2)
        return a + m.group(1) + (" " + resto if resto else "")
    return a + " " + b


def reflow(texto: str) -> str:
    texto = normalizar_base(texto)

    # (linha, comecou_bloco_novo) — comecou_bloco_novo indica se havia uma
    # linha em branco na fonte antes desta linha (sinal de quebra real).
    itens = []
    for bruto in re.split(r"\n\s*\n", texto):
        linhas_bloco = [l.strip() for l in bruto.splitlines() if l.strip()]
        for i, l in enumerate(linhas_bloco):
            itens.append((l, i == 0))

    paragrafos = []
    buffer = ""
    for linha, novo_bloco in itens:
        decorada = limpar_decoracao(linha)
        if eh_header(decorada):
            if buffer:
                paragrafos.append(buffer)
                buffer = ""
            paragrafos.append(decorada)
            continue
        if not buffer:
            buffer = linha
            continue
        if ITEM_NUMERADO.match(linha) or (novo_bloco and not eh_fragmento(buffer)):
            # item de lista numerada, ou havia quebra real na fonte E o
            # buffer já parece uma unidade completa (frase terminada)
            paragrafos.append(buffer)
            buffer = linha
        else:
            # mesma frase quebrada em várias linhas (wrap de PDF), ou o
            # buffer ainda é só um fragmento curto (rótulo/data/local) —
            # continua juntando
            buffer = juntar(buffer, linha)
    if buffer:
        paragrafos.append(buffer)
    return "\n\n".join(paragrafos).strip() + "\n"


def score_tech(texto_lower: str) -> int:
    return sum(texto_lower.count(termo) for termo in VOCAB_TECH)


def main() -> None:
    limite = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 30
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0

    SAIDA.mkdir(parents=True, exist_ok=True)

    candidatos = []
    for path in sorted(ENTRADA.glob("*.json")):
        registro = json.loads(path.read_text(encoding="utf-8"))
        texto = (registro.get("text") or "").strip()
        if len(texto) < 200:
            continue
        candidatos.append((score_tech(texto.lower()), path, texto))

    # ordenação estável — com o mesmo VOCAB_TECH, os primeiros N já escolhidos
    # em uma rodada anterior continuam sendo os primeiros N aqui; --skip pula
    # esse trecho já convertido e continua a partir do próximo colocado.
    candidatos.sort(key=lambda t: t[0], reverse=True)
    escolhidos = candidatos[skip:skip + limite]

    total = 0
    for score, path, texto in escolhidos:
        total += 1
        ident = re.search(r"\((\d+)\)", path.stem)
        ident = ident.group(1) if ident else str(total)
        caminho = SAIDA / f"curriculo_meyhaar_{int(ident):04d}.txt"
        caminho.write_text(reflow(texto), encoding="utf-8")

    if escolhidos:
        print(f"score tech: min={escolhidos[-1][0]} max={escolhidos[0][0]}")
    print(f"{total} currículos convertidos em {SAIDA}/ (de {len(candidatos)} candidatos com texto >= 200 chars)")


if __name__ == "__main__":
    main()
