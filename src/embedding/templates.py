"""
Templates textuais para vetorização — formalizam a estrategia dual-vetor
esbocada em sandbox/vectorstore_sandbox.py e documentada em context/04_EMBEDDINGS.md.

Cada currículo gera dois textos:
  skills_text  -> palavras-chave técnicas (comparável ao skills_text de uma vaga)
  perfil_text  -> parágrafo narrativo do perfil (comparável ao perfil_text de uma vaga)

A simetria de estrutura entre os templates de currículo e vaga é o que faz o
modelo de embedding posicionar os dois documentos perto no espaço vetorial
quando os campos análogos são similares.
"""
from ..schemas.curriculo import CurriculoEstruturado
from ..schemas.vaga import VagaEstruturada


def curriculo_to_skills_text(c: CurriculoEstruturado) -> str:
    """Texto focado apenas em skills — sem narrativa, sem contexto."""
    if not c.skills_tecnicas:
        return "sem skills técnicas informadas"
    return ", ".join(c.skills_tecnicas)


def curriculo_to_perfil_text(c: CurriculoEstruturado) -> str:
    """Parágrafo narrativo do perfil profissional."""
    area = c.area_principal or "área não especificada"
    anos = (
        f"{c.anos_experiencia_estimados} anos de experiência"
        if c.anos_experiencia_estimados is not None
        else "experiência não informada"
    )
    senioridade = c.senioridade_estimada or "não informado"

    partes = [f"Profissional de {area} com {anos}, nível {senioridade}."]

    if c.formacoes:
        f = c.formacoes[0]
        partes.append(f"Formação em {f.curso} pela {f.instituicao}.")

    if c.skills_tecnicas:
        partes.append(f"Competências técnicas incluem {', '.join(c.skills_tecnicas)}.")

    if c.experiencias:
        e = c.experiencias[0]
        exp = f"Experiência recente como {e.cargo} na {e.empresa}"
        if e.descricao:
            exp += f", atuando com {e.descricao}"
        partes.append(exp + ".")

    if c.skills_comportamentais:
        partes.append(f"Perfil comportamental: {', '.join(c.skills_comportamentais)}.")

    if c.modelo_trabalho_preferido:
        partes.append(f"Preferência de trabalho {c.modelo_trabalho_preferido}.")

    return " ".join(partes)


def vaga_to_skills_text(v: VagaEstruturada) -> str:
    """Texto focado apenas em skills — obrigatórias + desejadas."""
    skills = v.skills_obrigatorias + v.skills_desejadas
    if not skills:
        return "sem skills técnicas informadas"
    return ", ".join(skills)


def vaga_to_perfil_text(v: VagaEstruturada) -> str:
    """Parágrafo narrativo da vaga — estrutura simétrica a `curriculo_to_perfil_text`."""
    area = v.area or "área não especificada"
    anos = (
        f"{v.experiencia_minima_anos} anos de experiência"
        if v.experiencia_minima_anos is not None
        else "experiência não informada"
    )
    senioridade = v.senioridade or "não informado"

    # Sem título (ver VagaEstruturada.titulo), a frase cai para a área — que é o
    # que o texto do anúncio de fato indica nesse caso.
    alvo = v.titulo or area
    partes = [f"Vaga de {alvo} para profissional de {area} com {anos}, nível {senioridade}."]

    if v.formacao_minima:
        partes.append(f"Formação mínima: {v.formacao_minima}.")

    if v.skills_obrigatorias:
        partes.append(f"Requisitos técnicos incluem {', '.join(v.skills_obrigatorias)}.")

    if v.skills_desejadas:
        partes.append(f"Diferenciais desejados: {', '.join(v.skills_desejadas)}.")

    if v.responsabilidades:
        partes.append(f"Responsabilidades envolvem {', '.join(v.responsabilidades)}.")

    if v.modelo_trabalho:
        partes.append(f"Modelo de trabalho {v.modelo_trabalho}.")

    return " ".join(partes)
