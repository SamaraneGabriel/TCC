"""
Métricas objetivas para o pipeline de avaliação via LLM-as-judge
(ver context/08_AVALIACAO.md e a Fase 7 do plano de implementação).

Funções puras — não chamam LLM nem tocam o vectorstore, operam sobre o
resultado já produzido pelo judge (graus de relevância + ordem sugerida).
"""


def spearman_rank_correlation(ordem_a: list[str], ordem_b: list[str]) -> float:
    """Correlação de Spearman entre duas ordenações do mesmo conjunto de ids
    (ex: ordem do sistema vs. ordem sugerida pelo LLM-judge).

    1.0 = ordens idênticas, -1.0 = ordens invertidas, ~0.0 = sem relação.
    Levanta ValueError se as duas ordens não contiverem exatamente os mesmos ids.
    """
    if set(ordem_a) != set(ordem_b):
        raise ValueError("as duas ordens precisam conter exatamente os mesmos ids")

    n = len(ordem_a)
    if n < 2:
        return 1.0  # sem discordancia possivel com 0 ou 1 item

    rank_a = {item: i for i, item in enumerate(ordem_a)}
    rank_b = {item: i for i, item in enumerate(ordem_b)}

    soma_d2 = sum((rank_a[item] - rank_b[item]) ** 2 for item in ordem_a)
    return 1 - (6 * soma_d2) / (n * (n**2 - 1))


def pct_relevantes(graus: list[int], limiar: int = 1) -> float:
    """Fração (0-1) de candidatos com grau de relevância >= limiar.
    Convenção de grau: 0=não relevante, 1=parcialmente relevante, 2=relevante."""
    if not graus:
        return 0.0
    return sum(1 for g in graus if g >= limiar) / len(graus)


def grau_medio(graus: list[int]) -> float:
    """Média aritmética dos graus de relevância (0/1/2) dos candidatos julgados."""
    if not graus:
        return 0.0
    return sum(graus) / len(graus)
