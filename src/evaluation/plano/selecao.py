"""Seleção de vagas do pool (seção 2 do plano de avaliação).

Funções puras — recebem o pool já carregado como `VagaPool`, não sabem de onde
os dados vieram nem tocam LLM/vectorstore. Carregar o pool de um dataset real
(~5000 vagas) é responsabilidade de outro módulo/script, ainda pendente do
usuário escolher e disponibilizar o dataset em `data/vagas/` — este módulo não
inventa vagas (ver a regra de dataset orgânico do projeto).
"""
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigAvaliacao

CAMPOS_TABELA_CONTROLE = ["vaga_id", "cargo", "area_tecnica", "senioridade", "is_calibracao"]


@dataclass(frozen=True)
class VagaPool:
    """Uma vaga do pool disponível. Os campos opcionais depois de `senioridade`
    vêm do dataset real (ver `pool_vagas.py`) e não participam do sorteio — são
    carregados junto porque a fase de busca precisa deles (texto da vaga para o
    extrator, `modelo_trabalho`/`localidade` para os filtros do store)."""

    vaga_id: str
    cargo: str
    texto: str  # descrição/requisitos crus — usado só pra checar trivialidade
    area_tecnica: str | None = None
    senioridade: str | None = None
    empresa: str | None = None
    localidade: str | None = None
    modelo_trabalho: str | None = None
    skills: str | None = None


class CoberturaInsuficiente(Exception):
    """Levantada quando o sorteio não atinge a cobertura mínima exigida (seção
    2, passo 3) mesmo após as tentativas de resorteio — nesse ponto o plano
    pede completar manualmente, não é algo pra este código decidir sozinho."""

    def __init__(self, checagem: dict):
        self.checagem = checagem
        super().__init__("; ".join(checagem.get("bloqueios", [])) or "cobertura insuficiente")


def _requisitos_nao_trivial(texto: str) -> bool:
    """Seção 2, passo 3: "descrição de requisitos não trivial (mais que uma linha)"."""
    linhas = [l for l in texto.strip().splitlines() if l.strip()]
    return len(linhas) > 1


def _tem_metadados_suficientes(pool: list[VagaPool]) -> bool:
    """Estratificação (seção 2, passo 2) só faz sentido se área E senioridade
    estiverem preenchidas na quase totalidade do pool; poucos faltantes não
    impedem, mas um pool majoritariamente sem metadados cai pro sorteio simples."""
    if not pool:
        return False
    com_area = sum(1 for v in pool if v.area_tecnica)
    com_senioridade = sum(1 for v in pool if v.senioridade)
    return com_area / len(pool) >= 0.9 and com_senioridade / len(pool) >= 0.9


def sortear_vagas_estudo(pool: list[VagaPool], cfg: ConfigAvaliacao, seed: int) -> list[VagaPool]:
    """Seção 2, passo 2: sorteia N_VAGAS_ESTUDO vagas, estratificando por área
    técnica x senioridade quando os metadados estão disponíveis; caso
    contrário, sorteio aleatório simples (o passo 3 checa a cobertura depois)."""
    if len(pool) < cfg.n_vagas_estudo:
        raise ValueError(
            f"pool tem {len(pool)} vaga(s), menos que N_VAGAS_ESTUDO={cfg.n_vagas_estudo}"
        )
    rng = random.Random(seed)
    if _tem_metadados_suficientes(pool):
        return _sorteio_estratificado(pool, cfg.n_vagas_estudo, rng)
    return rng.sample(pool, cfg.n_vagas_estudo)


def _sorteio_estratificado(pool: list[VagaPool], n: int, rng: random.Random) -> list[VagaPool]:
    """Alocação proporcional ao tamanho de cada estrato (área, senioridade),
    fechando em exatamente `n` pelo método do maior resto."""
    estratos: dict[tuple, list[VagaPool]] = defaultdict(list)
    for v in pool:
        estratos[(v.area_tecnica, v.senioridade)].append(v)
    for grupo in estratos.values():
        rng.shuffle(grupo)

    chaves = list(estratos.keys())
    total = len(pool)
    cotas_exatas = {k: n * len(estratos[k]) / total for k in chaves}
    cotas = {k: int(cotas_exatas[k]) for k in chaves}
    faltam = n - sum(cotas.values())
    ordem_resto = sorted(chaves, key=lambda k: cotas_exatas[k] - cotas[k], reverse=True)
    for k in ordem_resto[:faltam]:
        cotas[k] += 1

    selecionadas: list[VagaPool] = []
    sobras: list[VagaPool] = []
    for k in chaves:
        grupo = estratos[k]
        cota = min(cotas[k], len(grupo))
        selecionadas.extend(grupo[:cota])
        sobras.extend(grupo[cota:])

    if len(selecionadas) < n:
        rng.shuffle(sobras)
        selecionadas.extend(sobras[: n - len(selecionadas)])
    return selecionadas[:n]


def checar_cobertura(vagas: list[VagaPool], cfg: ConfigAvaliacao) -> dict:
    """Seção 2, passo 3: ≥3 áreas distintas, ≥2 senioridades distintas,
    nenhuma vaga com descrição trivial (1 linha)."""
    areas = {v.area_tecnica for v in vagas if v.area_tecnica}
    senioridades = {v.senioridade for v in vagas if v.senioridade}
    triviais = [v.vaga_id for v in vagas if not _requisitos_nao_trivial(v.texto)]

    bloqueios = []
    if len(areas) < cfg.min_areas_tecnicas:
        bloqueios.append(
            f"{len(areas)} área(s) técnica(s) distinta(s) nas vagas sorteadas, "
            f"mínimo {cfg.min_areas_tecnicas}"
        )
    if len(senioridades) < cfg.min_senioridades:
        bloqueios.append(
            f"{len(senioridades)} nível(is) de senioridade distinto(s), mínimo {cfg.min_senioridades}"
        )
    if triviais:
        bloqueios.append(
            f"{len(triviais)} vaga(s) com descrição de requisitos trivial (≤1 linha): {triviais}"
        )
    return {
        "ok": not bloqueios,
        "bloqueios": bloqueios,
        "areas": sorted(areas),
        "senioridades": sorted(senioridades),
    }


def sortear_com_cobertura(
    pool: list[VagaPool], cfg: ConfigAvaliacao, seed: int, max_tentativas: int = 20
) -> tuple[list[VagaPool], dict, int]:
    """Repete o sorteio (seeds derivadas de `seed`) até a cobertura passar.

    Retorna (vagas_estudo, checagem, seed_usada). Esgotadas as tentativas sem
    cobrir o mínimo, levanta `CoberturaInsuficiente` — nesse ponto o plano pede
    completar manualmente, decisão de quem roda o script, não deste código.
    """
    rng_seeds = random.Random(seed)
    ultima_checagem: dict = {}
    for _ in range(max_tentativas):
        tentativa_seed = rng_seeds.randrange(2**32)
        vagas = sortear_vagas_estudo(pool, cfg, tentativa_seed)
        checagem = checar_cobertura(vagas, cfg)
        if checagem["ok"]:
            return vagas, checagem, tentativa_seed
        ultima_checagem = checagem
    raise CoberturaInsuficiente(ultima_checagem)


def selecionar_vagas_calibracao(vagas_estudo: list[VagaPool], cfg: ConfigAvaliacao, seed: int) -> list[str]:
    """Seção 2, passo 4: marca N_VAGAS_CALIBRACAO vaga_ids dentre as vagas de
    estudo já sorteadas, cobrindo pelo menos MIN_AREAS_CALIBRACAO áreas
    técnicas distintas entre si."""
    if len(vagas_estudo) < cfg.n_vagas_calibracao:
        raise ValueError(
            f"{len(vagas_estudo)} vaga(s) de estudo, menos que N_VAGAS_CALIBRACAO={cfg.n_vagas_calibracao}"
        )

    rng = random.Random(seed)
    por_area: dict[str | None, list[VagaPool]] = defaultdict(list)
    for v in vagas_estudo:
        por_area[v.area_tecnica].append(v)
    for grupo in por_area.values():
        rng.shuffle(grupo)

    areas_distintas = list(por_area.keys())
    rng.shuffle(areas_distintas)

    escolhidas: list[VagaPool] = []
    for area in areas_distintas:
        if len({v.area_tecnica for v in escolhidas}) >= cfg.min_areas_calibracao:
            break
        escolhidas.append(por_area[area].pop(0))

    restante = [v for grupo in por_area.values() for v in grupo]
    rng.shuffle(restante)
    for v in restante:
        if len(escolhidas) >= cfg.n_vagas_calibracao:
            break
        escolhidas.append(v)

    if len({v.area_tecnica for v in escolhidas}) < cfg.min_areas_calibracao:
        raise CoberturaInsuficiente(
            {
                "bloqueios": [
                    f"não foi possível cobrir {cfg.min_areas_calibracao} área(s) técnica(s) "
                    f"distinta(s) nas {cfg.n_vagas_calibracao} vaga(s) de calibração — "
                    "complete manualmente ou revise o sorteio de vagas de estudo"
                ]
            }
        )

    return [v.vaga_id for v in escolhidas[: cfg.n_vagas_calibracao]]


def gerar_tabela_controle(vagas_estudo: list[VagaPool], ids_calibracao: set[str]) -> list[dict]:
    """`vagas_estudo.csv` (seção 13) — gerada a partir do sorteio, nunca preenchida à mão."""
    return [
        {
            "vaga_id": v.vaga_id,
            "cargo": v.cargo,
            "area_tecnica": v.area_tecnica,
            "senioridade": v.senioridade,
            "is_calibracao": v.vaga_id in ids_calibracao,
        }
        for v in vagas_estudo
    ]


def salvar_tabela_controle(tabela: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_TABELA_CONTROLE)
        writer.writeheader()
        writer.writerows(tabela)
