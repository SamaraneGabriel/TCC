"""
Converte o dataset JSONL de currículos (Hugging Face) em arquivos .txt,
um por currículo, no formato que scripts/ingest_curriculos.py já processa.

Uso:
    python scripts/convert_hf_resumes.py [--limit N]

Lê:        data/hugging_face/master_resumes.jsonl
Escreve em: data/curriculos/curriculo_NNNN_nome.txt
"""
import json
import re
import sys
from pathlib import Path

ENTRADA = Path("data/hugging_face/master_resumes.jsonl")
SAIDA = Path("data/curriculos")

PLACEHOLDERS = {"", "unknown", "not provided", "no", "n/a"}


def limpo(valor):
    """None para valores-placeholder ('Unknown', vazio, etc.), senão o valor como string."""
    if valor is None:
        return None
    texto = str(valor).strip()
    return None if texto.lower() in PLACEHOLDERS else texto


def slug(nome: str) -> str:
    return re.sub(r"[^a-z0-9]", "", nome.lower())[:30] or "candidato"


def skills_tecnicas(registro: dict) -> list[str]:
    """Achata todas as categorias de skills.technical.* (nomes, sem duplicar)."""
    categorias = (registro.get("skills") or {}).get("technical") or {}
    vistos, skills = set(), []
    for itens in categorias.values():
        for item in itens or []:
            nome = limpo(item.get("name"))
            if nome and nome not in vistos:
                vistos.add(nome)
                skills.append(nome)
    return skills


def idiomas(registro: dict) -> list[str]:
    resultado = []
    for item in (registro.get("skills") or {}).get("languages") or []:
        nome, nivel = limpo(item.get("name")), limpo(item.get("level"))
        if nome:
            resultado.append(f"{nome} ({nivel})" if nivel else nome)
    return resultado


def experiencias_texto(registro: dict) -> str:
    blocos = []
    for exp in registro.get("experience") or []:
        cargo = limpo(exp.get("title")) or "Cargo não informado"
        empresa = limpo(exp.get("company")) or "Empresa não informada"
        datas = exp.get("dates") or {}
        periodo = " – ".join(filter(None, [limpo(datas.get("start")), limpo(datas.get("end"))]))
        duracao = limpo(datas.get("duration"))

        cabecalho = f"{cargo} | {empresa}"
        if periodo:
            cabecalho += f" | {periodo}"
        if duracao:
            cabecalho += f" ({duracao})"

        bullets = [f"- {limpo(r)}" for r in exp.get("responsibilities") or [] if limpo(r)]
        blocos.append("\n".join([cabecalho] + bullets))
    return "\n\n".join(blocos)


def formacoes_texto(registro: dict) -> str:
    blocos = []
    for edu in registro.get("education") or []:
        grau = edu.get("degree") or {}
        inst = edu.get("institution") or {}

        linha = " em ".join(filter(None, [limpo(grau.get("level")), limpo(grau.get("field"))]))
        nome_inst = limpo(inst.get("name"))
        if nome_inst:
            linha += f" | {nome_inst}"
        ano = limpo((edu.get("dates") or {}).get("expected_graduation"))
        if ano:
            linha += f" | Concluído/previsto em {ano}"

        if linha:
            blocos.append(linha)
    return "\n".join(blocos)


def modelo_trabalho(registro: dict) -> str | None:
    loc = ((registro.get("personal_info") or {}).get("location") or {})
    pref = limpo(loc.get("remote_preference"))
    if not pref:
        return None
    mapa = {"remote": "remoto", "hybrid": "híbrido", "onsite": "presencial"}
    return mapa.get(pref.lower(), pref)


def renderizar(registro: dict) -> str:
    pi = registro.get("personal_info") or {}
    loc = pi.get("location") or {}
    nome = limpo(pi.get("name")) or "Nome não informado"

    partes = [nome]

    contato = " | ".join(filter(None, [
        f"Email: {limpo(pi.get('email'))}" if limpo(pi.get("email")) else None,
        f"Tel: {limpo(pi.get('phone'))}" if limpo(pi.get("phone")) else None,
    ]))
    if contato:
        partes.append(contato)

    cidade_pais = ", ".join(filter(None, [limpo(loc.get("city")), limpo(loc.get("country"))]))
    if cidade_pais:
        partes.append(cidade_pais)

    if resumo := limpo(pi.get("summary")):
        partes.append(f"\nRESUMO PROFISSIONAL\n{resumo}")

    if skills := skills_tecnicas(registro):
        partes.append(f"\nHABILIDADES TÉCNICAS\n{', '.join(skills)}")

    if experiencias := experiencias_texto(registro):
        partes.append(f"\nEXPERIÊNCIA PROFISSIONAL\n{experiencias}")

    if formacoes := formacoes_texto(registro):
        partes.append(f"\nFORMAÇÃO\n{formacoes}")

    if certs := limpo(registro.get("certifications")):
        partes.append(f"\nCERTIFICAÇÕES\n{certs}")

    if langs := idiomas(registro):
        partes.append(f"\nIDIOMAS\n{', '.join(langs)}")

    if modelo := modelo_trabalho(registro):
        partes.append(f"\nModelo de trabalho: {modelo}")

    return "\n".join(partes) + "\n"


def main() -> None:
    limite = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    SAIDA.mkdir(parents=True, exist_ok=True)

    total = 0
    with ENTRADA.open(encoding="utf-8") as f:
        for linha in f:
            if limite and total >= limite:
                break
            registro = json.loads(linha)
            total += 1
            nome = limpo((registro.get("personal_info") or {}).get("name")) or "candidato"
            caminho = SAIDA / f"curriculo_{total:04d}_{slug(nome)}.txt"
            caminho.write_text(renderizar(registro), encoding="utf-8")

    print(f"{total} currículos convertidos em {SAIDA}/")


if __name__ == "__main__":
    main()
