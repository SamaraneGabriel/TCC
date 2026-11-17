"""Amostragem estratificada por vaga (seção 4) e estruturas de dados da
anotação/reprodutibilidade (seção 6) do plano de avaliação.

Funções puras — assim como `selecao.py`, não sabem de onde os dados vieram:
recebem os resultados de busca já prontos (lista de `curriculo_id` na ordem
do ranking, IDs do banco) em vez de chamar `CurriculoStore`/Qdrant. Quem
consulta o sistema com `limit=TOP_POOL` e lista os IDs do banco é
responsabilidade de outro módulo/script (fase de orquestração, ainda não
implementada) — este arquivo nunca inventa currículos nem vagas.

A "cegueira" do anotador quanto a estrato/posição/score (seção 4.2, passo 7)
é uma propriedade da *apresentação* da planilha ao humano, não da estrutura
de dados: `ItemAmostrado` e `LinhaAnotacao` sempre guardam estrato e posição,
porque o cálculo das métricas (seção 8, fora de escopo aqui) precisa deles
depois.
"""
import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ConfigAvaliacao
from .selecao import CoberturaInsuficiente

CAMPOS_PLANILHA_ANOTACAO = [
    "vaga_id",
    "curriculo_id",
    "estrato",
    "posicao_sistema",
    "papel",
    "nota_humano_1",
    "nota_humano_2",
    "nota_llm_1",
    "nota_llm_2",
    "fonte_nota_final",
    "observacoes",
]

CAMPOS_LOG_REPRODUTIBILIDADE = [
    "vaga_id",
    "seed_estrato_b",
    "seed_estrato_c",
    "seed_embaralhamento",
    "data_coleta",
    "k_usado",
    "n_total_usado",
    "is_calibracao",
]


@dataclass(frozen=True)
class ItemAmostrado:
    """Um currículo sorteado para uma vaga, com a origem que o cálculo de
    métricas (seção 8) vai precisar depois. `posicao_sistema` é None para o
    estrato C: ele não veio do ranking do sistema, veio do complemento de
    conjunto (seção 4.2, passo 4)."""

    curriculo_id: str
    estrato: str  # "A" | "B" | "C"
    posicao_sistema: int | None


def seeds_da_vaga(vaga_id: str, seed_mestre: int) -> tuple[int, int, int]:
    """Deriva as três seeds de uma vaga (estrato B, estrato C, embaralhamento) a
    partir da seed mestre, para o log de reprodutibilidade da seção 7.2.

    Usa SHA-256 do par (seed_mestre, vaga_id, papel), **não** o `hash()` embutido
    do Python: `hash()` de str é randomizado por processo (PYTHONHASHSEED), então
    reproduzir o estudo noutro dia daria outra amostra. Também não se usa um
    contador sequencial — assim acrescentar ou remover uma vaga do estudo não
    desloca as seeds das demais.

    Retorna inteiros em [0, 2^31), que é o que `random.Random` aceita bem em
    qualquer plataforma.
    """

    def _seed(papel: str) -> int:
        material = f"{seed_mestre}:{vaga_id}:{papel}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)

    return _seed("estrato_b"), _seed("estrato_c"), _seed("embaralhamento")


def estrato_a(resultados_top_pool: list[str], k: int) -> list[ItemAmostrado]:
    """Seção 4.1/4.2 passo 2: censo completo das primeiras K posições.

    Se o sistema devolveu menos de K resultados (banco pequeno), o censo é
    só o que existe — não é erro, é um estrato A menor que o nominal."""
    return [
        ItemAmostrado(curriculo_id=cid, estrato="A", posicao_sistema=pos)
        for pos, cid in enumerate(resultados_top_pool[:k], start=1)
    ]


def estrato_b(resultados_top_pool: list[str], k: int, n_b: int, seed: int) -> list[ItemAmostrado]:
    """Seção 4.1/4.2 passo 3: amostra aleatória simples de `n_b` itens nas
    posições K+1..TOP_POOL. Levanta ValueError se o universo disponível
    (o que sobrou de `resultados_top_pool` além de K) for menor que `n_b` —
    sinal de que TOP_POOL não foi de fato consultado, ou o banco é pequeno
    demais para os parâmetros configurados."""
    candidatos = list(enumerate(resultados_top_pool[k:], start=k + 1))
    if n_b > len(candidatos):
        raise ValueError(
            f"estrato B: universo tem {len(candidatos)} posição(ões) (K+1..TOP_POOL), "
            f"menos que n_b={n_b}"
        )
    rng = random.Random(seed)
    sorteados = rng.sample(candidatos, n_b)
    return [ItemAmostrado(curriculo_id=cid, estrato="B", posicao_sistema=pos) for pos, cid in sorteados]


def estrato_c(
    todos_ids_banco: set[str], ids_top_pool: set[str], n_c: int, seed: int
) -> list[ItemAmostrado]:
    """Seção 4.1/4.2 passos 4-5: universo C = todos os IDs do banco menos os
    IDs do TOP_POOL (complemento de conjunto), depois amostra `n_c` desse
    universo. Sem posição de ranking (`posicao_sistema=None`) — C não vem do
    ranking do sistema.

    O universo é ordenado (`sorted`) antes do sorteio: sets em Python não têm
    ordem de iteração garantida entre execuções, e sem isso a mesma seed
    poderia produzir amostras diferentes em runs distintos."""
    universo = sorted(todos_ids_banco - ids_top_pool)
    if n_c > len(universo):
        raise ValueError(
            f"estrato C: universo tem {len(universo)} id(s) fora do TOP_POOL, menos que n_c={n_c}"
        )
    rng = random.Random(seed)
    sorteados = rng.sample(universo, n_c)
    return [ItemAmostrado(curriculo_id=cid, estrato="C", posicao_sistema=None) for cid in sorteados]


def juntar_estratos(
    itens_a: list[ItemAmostrado],
    itens_b: list[ItemAmostrado],
    itens_c: list[ItemAmostrado],
    seed: int,
) -> list[ItemAmostrado]:
    """Seção 4.2 passo 7: reúne os três estratos de uma vaga numa lista única
    embaralhada (seed registrada no log de reprodutibilidade, seção 6.2).

    O embaralhamento existe para a planilha exibida ao anotador humano, que
    não deve ver a que estrato/posição cada item pertence; os campos
    continuam presentes em cada `ItemAmostrado` para uso interno."""
    todos = [*itens_a, *itens_b, *itens_c]
    rng = random.Random(seed)
    rng.shuffle(todos)
    return todos


@dataclass(frozen=True)
class LinhaAnotacao:
    """Seção 6.1: uma linha por par vaga-currículo na planilha de anotação.
    As notas ficam None nesta etapa — anotar (humano ou LLM) é um passo
    posterior, fora de escopo aqui."""

    vaga_id: str
    curriculo_id: str
    estrato: str
    posicao_sistema: int | None
    papel: str  # calibracao | calibracao_double_label | producao_llm
    nota_humano_1: int | None = None
    nota_humano_2: int | None = None
    nota_llm_1: int | None = None
    nota_llm_2: int | None = None
    fonte_nota_final: str | None = None
    observacoes: str | None = None


def gerar_linhas_anotacao(vaga_id: str, itens: list[ItemAmostrado], papel: str) -> list[LinhaAnotacao]:
    """Converte os itens amostrados de uma vaga em linhas da planilha de
    anotação (seção 6.1), com `papel` fixo para o lote (ex.: "producao_llm")."""
    return [
        LinhaAnotacao(
            vaga_id=vaga_id,
            curriculo_id=item.curriculo_id,
            estrato=item.estrato,
            posicao_sistema=item.posicao_sistema,
            papel=papel,
        )
        for item in itens
    ]


def salvar_planilha_anotacao(linhas: list[LinhaAnotacao], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_PLANILHA_ANOTACAO)
        writer.writeheader()
        writer.writerows(asdict(linha) for linha in linhas)


def separar_amostra_calibracao(
    itens_por_vaga: dict[str, list[ItemAmostrado]],
    ids_vagas_calibracao: list[str] | set[str],
    cfg: ConfigAvaliacao,
    seed: int,
    vaga_censo_a: str | None = None,
) -> tuple[list[LinhaAnotacao], list[LinhaAnotacao]]:
    """Seção 3.2: separa, dentre os itens já amostrados por vaga (seção 4,
    antes do embaralhamento — precisamos do `estrato` de cada um), os grupos
    AB e C da amostra de calibração, olhando só para as vagas de calibração.

    AB mistura os estratos A e B garantindo pelo menos
    `cfg.min_substrato_calibracao` de cada um; C vem só do estrato C. Com
    `vaga_censo_a`, o estrato A dessa vaga entra inteiro no grupo AB (§4.2,
    recomendação) e conta para o mínimo do substrato A. Cada
    grupo é reduzido a `cfg.n_calibracao_por_grupo` itens (97 por padrão) por
    amostra aleatória simples sobre o que as vagas de calibração produziram.

    Levanta `CoberturaInsuficiente` se as vagas de calibração não renderem
    itens suficientes para algum substrato/grupo, em vez de completar com
    menos itens silenciosamente — nesse ponto o plano pede rever o sorteio de
    vagas de calibração ou de estudo (mesma decisão de quem roda o script,
    não deste código, seção 2)."""
    if 2 * cfg.min_substrato_calibracao > cfg.n_calibracao_por_grupo:
        raise ValueError(
            f"2 * min_substrato_calibracao={2 * cfg.min_substrato_calibracao} > "
            f"n_calibracao_por_grupo={cfg.n_calibracao_por_grupo}: configuração inconsistente"
        )

    itens_a: list[tuple[str, ItemAmostrado]] = []
    itens_b: list[tuple[str, ItemAmostrado]] = []
    itens_c: list[tuple[str, ItemAmostrado]] = []
    # sorted: a ordem de iteração de um set de strings muda a cada processo
    # (PYTHONHASHSEED), e ela define a lista que o rng embaralha — sem ordenar,
    # a mesma seed sorteia amostras diferentes em execuções diferentes.
    for vaga_id in sorted(ids_vagas_calibracao):
        for item in itens_por_vaga.get(vaga_id, []):
            if item.estrato == "A":
                itens_a.append((vaga_id, item))
            elif item.estrato == "B":
                itens_b.append((vaga_id, item))
            elif item.estrato == "C":
                itens_c.append((vaga_id, item))

    bloqueios = []
    if len(itens_a) < cfg.min_substrato_calibracao:
        bloqueios.append(
            f"estrato A: {len(itens_a)} item(ns) nas vagas de calibração, "
            f"mínimo {cfg.min_substrato_calibracao}"
        )
    if len(itens_b) < cfg.min_substrato_calibracao:
        bloqueios.append(
            f"estrato B: {len(itens_b)} item(ns) nas vagas de calibração, "
            f"mínimo {cfg.min_substrato_calibracao}"
        )
    if len(itens_a) + len(itens_b) < cfg.n_calibracao_por_grupo:
        bloqueios.append(
            f"grupo AB: {len(itens_a) + len(itens_b)} item(ns) disponíveis nas vagas de "
            f"calibração, precisa de {cfg.n_calibracao_por_grupo}"
        )
    if len(itens_c) < cfg.n_calibracao_por_grupo:
        bloqueios.append(
            f"grupo C: {len(itens_c)} item(ns) disponíveis nas vagas de calibração, "
            f"precisa de {cfg.n_calibracao_por_grupo}"
        )
    if bloqueios:
        raise CoberturaInsuficiente({"bloqueios": bloqueios})

    # §4.2, recomendação: o estrato A inteiro de uma vaga entra no grupo AB, para
    # que Precision@K e nDCG@K dessa vaga sejam medidos só por humano. Esses itens
    # contam para o mínimo do substrato A; o resto do grupo é sorteado como antes.
    censo = [(v, i) for v, i in itens_a if v == vaga_censo_a] if vaga_censo_a else []
    if vaga_censo_a and vaga_censo_a not in ids_vagas_calibracao:
        raise ValueError(f"vaga_censo_a={vaga_censo_a} não é vaga de calibração")
    if len(censo) + cfg.min_substrato_calibracao > cfg.n_calibracao_por_grupo:
        raise ValueError(
            f"censo do estrato A de {vaga_censo_a} ({len(censo)}) + mínimo de B "
            f"({cfg.min_substrato_calibracao}) > n_calibracao_por_grupo={cfg.n_calibracao_por_grupo}"
        )
    itens_a = [par for par in itens_a if par not in censo]

    rng = random.Random(seed)
    rng.shuffle(itens_a)
    rng.shuffle(itens_b)
    n_min_a = max(0, cfg.min_substrato_calibracao - len(censo))
    minimo_a = censo + itens_a[:n_min_a]
    minimo_b = itens_b[: cfg.min_substrato_calibracao]
    resto = itens_a[n_min_a:] + itens_b[cfg.min_substrato_calibracao :]
    rng.shuffle(resto)
    faltam = cfg.n_calibracao_por_grupo - len(minimo_a) - len(minimo_b)
    grupo_ab = minimo_a + minimo_b + resto[:faltam]

    rng.shuffle(itens_c)
    grupo_c = itens_c[: cfg.n_calibracao_por_grupo]

    linhas_ab = [
        LinhaAnotacao(
            vaga_id=vaga_id,
            curriculo_id=item.curriculo_id,
            estrato=item.estrato,
            posicao_sistema=item.posicao_sistema,
            papel="calibracao",
        )
        for vaga_id, item in grupo_ab
    ]
    linhas_c = [
        LinhaAnotacao(
            vaga_id=vaga_id,
            curriculo_id=item.curriculo_id,
            estrato=item.estrato,
            posicao_sistema=item.posicao_sistema,
            papel="calibracao",
        )
        for vaga_id, item in grupo_c
    ]
    return linhas_ab, linhas_c


def completar_censo_estrato_a(
    linhas_ab: list[LinhaAnotacao],
    itens_por_vaga: dict[str, list[ItemAmostrado]],
    ids_vagas_calibracao: list[str] | set[str],
) -> list[LinhaAnotacao]:
    """Acrescenta ao grupo AB os itens do estrato A das vagas de calibração que
    o sorteio deixou de fora, para que o top-K de cada uma seja anotado inteiro.

    Com o top-K completo, Precision@K e nDCG@K das vagas de calibração ficam
    medidos só por humano, sem depender do juiz. Não reamostra nada: os itens já
    sorteados continuam os mesmos, só entram os que faltavam. O grupo AB passa do
    tamanho da §4.1, que é um mínimo para a margem de erro, não um teto."""
    presentes = {(l.vaga_id, l.curriculo_id) for l in linhas_ab}
    novos = [
        LinhaAnotacao(
            vaga_id=vaga_id,
            curriculo_id=item.curriculo_id,
            estrato=item.estrato,
            posicao_sistema=item.posicao_sistema,
            papel="calibracao",
        )
        for vaga_id in sorted(ids_vagas_calibracao)
        for item in itens_por_vaga.get(vaga_id, [])
        if item.estrato == "A" and (vaga_id, item.curriculo_id) not in presentes
    ]
    return linhas_ab + novos


def escolher_vaga_censo_a(ids_vagas_calibracao: list[str] | set[str], seed: int) -> str:
    """Qual vaga de calibração terá o estrato A completo (§4.2). Sorteio
    determinístico pela seed, sobre os ids ordenados — não depende da ordem de
    entrada nem de quem roda o script."""
    return random.Random(f"{seed}:censo_a").choice(sorted(ids_vagas_calibracao))


@dataclass(frozen=True)
class LogReprodutibilidade:
    """Seção 6.2: uma linha por vaga, registrando as seeds usadas na
    amostragem estratificada (seção 4) para permitir reproduzir o sorteio."""

    vaga_id: str
    seed_estrato_b: int
    seed_estrato_c: int
    seed_embaralhamento: int
    data_coleta: str
    k_usado: int
    n_total_usado: int
    is_calibracao: bool


def salvar_log_reprodutibilidade(linhas: list[LogReprodutibilidade], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_LOG_REPRODUTIBILIDADE)
        writer.writeheader()
        writer.writerows(asdict(linha) for linha in linhas)
