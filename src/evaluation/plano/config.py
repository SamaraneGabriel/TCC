"""Parâmetros do plano de avaliação (seção 1) e as fórmulas de tamanho de amostra
(seções 3.1 e 4.1).

Regra do plano: *nenhuma fórmula deve ter K ou N fixados no código; entram como
config*. Por isso todo valor marcado `PARÂMETRO` vive aqui (sobrescrevível por
`.env`/variável de ambiente com o prefixo `AVAL_`), e todo tamanho de amostra é
**derivado** — `n_calibracao_por_grupo`, `n_b` e `n_c` são propriedades
calculadas, não constantes.

Com os números do texto do plano (N_TOTAL=4817, TOP_POOL=200) os derivados são:
    n_calibracao_por_grupo = 97   (e=0,10, sem fpc)
    n_b                    = 15   (e=0,25, fpc com N=175)
    n_c                    = 95   (e=0,10, fpc com N=4617)
Os defaults aqui são os do projeto real, não os do texto: `n_total` é o banco
após a deduplicação de 2026-09-16 (385) e `top_pool` foi baixado de 200 para
100 por decisão de 2026-09-22 — com 385 currículos, um pool de 200 seria 52%
do banco inteiro e o estrato C (o "resto do mundo" usado para estimar
Recall@K) ficaria com 185 itens. Com TOP_POOL=100 os derivados viram
n_b=13 e n_c=73. Mudar `K`, `N_TOTAL` ou `TOP_POOL` recalcula tudo, é esse o
ponto.
"""
from math import ceil

from pydantic_settings import BaseSettings, SettingsConfigDict


def n0_amostra(z: float, e: float, p: float = 0.5) -> float:
    """Seção 3.1: `n0 = Z^2 * p * (1 - p) / e^2`, sem correção de população finita.

    Retorna o valor contínuo — quem arredonda é `tamanho_amostra`, para que a
    fpc seja aplicada sobre o n0 exato (arredondar antes muda o resultado).
    """
    if e <= 0:
        raise ValueError("margem de erro `e` precisa ser > 0")
    return (z**2) * p * (1 - p) / (e**2)


def corrigir_populacao_finita(n0: float, n_universo: int) -> float:
    """Correção de população finita: `n = n0 / (1 + (n0 - 1) / N)`.

    Aplicada aos estratos B e C, cujos universos são pequenos o bastante para
    que amostrar sem reposição reduza a variância de forma não desprezível.
    """
    if n_universo <= 0:
        raise ValueError("universo precisa ser > 0")
    return n0 / (1 + (n0 - 1) / n_universo)


def tamanho_amostra(z: float, e: float, n_universo: int | None = None, p: float = 0.5) -> int:
    """Tamanho de amostra final: n0, fpc opcional, arredondado pra cima.

    `n_universo=None` pula a fpc (caso da amostra de calibração, seção 3.1).
    Com universo informado, o resultado nunca excede o universo — se a fórmula
    pedir mais itens do que existem, o estrato vira censo.
    """
    n = n0_amostra(z, e, p)
    if n_universo is None:
        return ceil(n)
    return min(ceil(corrigir_populacao_finita(n, n_universo)), n_universo)


class ConfigAvaliacao(BaseSettings):
    """Seção 1 do plano. Sobrescreva por `.env`/ambiente com prefixo `AVAL_`
    (ex.: `AVAL_K=10`, `AVAL_N_TOTAL=2500`) ou instanciando com kwargs."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="aval_", extra="ignore")

    # --- PARÂMETROS: conferir antes de rodar ---
    k: int = 25                      # limite de exibição do sistema ao recrutador
    n_total: int = 385               # currículos no banco (após a dedup de 2026-09-16;
                                     # o plano cita 4817, que era o dataset antes da ingestão real)
    top_pool: int = 100              # pool recuperado por vaga; define os estratos A e B
                                     # (o texto do plano diz 200, pensado para N_TOTAL=4817)
    n_vagas_estudo: int = 60         # vagas julgadas pelo LLM-juiz após calibração
    n_vagas_calibracao: int = 3      # subconjunto de onde sai a amostra de calibração

    # --- Estatística ---
    z: float = 1.96                  # nível de confiança 95%
    e_b: float = 0.25                # margem de erro do estrato B
    e_c: float = 0.10                # margem de erro do estrato C
    e_calib: float = 0.10            # margem de erro da calibração (não reduzir abaixo disso)
    bootstrap_b: int = 2000          # reamostragens para IC

    # --- Limiares de concordância ---
    kappa_limiar_llm: float = 0.60   # aplicado separadamente a kappa_AB e kappa_C
    kappa_limiar_humano: float = 0.41

    # --- Frações de anotação ---
    double_label_fracao: float = 0.15    # 2ª anotação humana independente, na calibração
    # A auditoria humana contínua (§8.5 da spec) foi removida por decisão de
    # 2026-09-22: contradizia a primeira linha da própria spec ("anotação manual
    # restrita à amostra de calibração") e custaria 1.020 julgamentos humanos
    # além dos 194. Ver a seção de desvios em context/10_PLANO_AVALIACAO.md.

    # --- Mínimos de composição (seções 2.3 e 3.2) ---
    min_substrato_calibracao: int = 30   # itens de A e de B, cada, no grupo AB
    min_areas_tecnicas: int = 3          # cobertura exigida em N_VAGAS_ESTUDO
    min_senioridades: int = 2
    min_areas_calibracao: int = 2        # cobertura exigida entre as vagas de calibração
    min_vagas_calibracao: int = 3

    # --- Reprodutibilidade e saída ---
    seed_mestre: int = 20260915      # deriva as seeds por vaga do log (seção 6.2)
    saida_dir: str = "data/eval/plano"
    limiar_relevante: int = 2        # binarização da escala 0-3 (seção 5)

    # ------------------------------------------------------------------
    # Universos dos estratos (seção 4.1)

    @property
    def n_universo_b(self) -> int:
        """Posições K+1 .. TOP_POOL."""
        return self.top_pool - self.k

    @property
    def n_universo_c(self) -> int:
        """Tudo além do TOP_POOL recuperado."""
        return self.n_total - self.top_pool

    # ------------------------------------------------------------------
    # Tamanhos de amostra derivados (seções 3.1 e 4.1)

    @property
    def n_calibracao_por_grupo(self) -> int:
        """Itens por grupo (AB e C) na calibração — sem fpc, por decisão do plano."""
        return tamanho_amostra(self.z, self.e_calib)

    @property
    def n_calibracao_total(self) -> int:
        return 2 * self.n_calibracao_por_grupo

    @property
    def n_b(self) -> int:
        return tamanho_amostra(self.z, self.e_b, self.n_universo_b)

    @property
    def n_c(self) -> int:
        return tamanho_amostra(self.z, self.e_c, self.n_universo_c)

    @property
    def n_double_label(self) -> int:
        """Seção 3.3 passo 3 — arredondado pra cima, para não ficar abaixo da fração."""
        return ceil(self.double_label_fracao * self.n_calibracao_total)

    @property
    def itens_por_vaga(self) -> int:
        """Total anotado por vaga: censo de A + amostras de B e C."""
        return self.k + self.n_b + self.n_c

    # ------------------------------------------------------------------
    # Consistência

    def problemas(self) -> list[str]:
        """Inconsistências que impedem rodar o estudo. Lista vazia = pode seguir."""
        p = []
        if self.k <= 0:
            p.append(f"K={self.k}: precisa ser > 0")
        if self.top_pool <= self.k:
            p.append(f"TOP_POOL={self.top_pool} <= K={self.k}: estrato B ficaria vazio")
        if self.n_total <= self.top_pool:
            p.append(
                f"N_TOTAL={self.n_total} <= TOP_POOL={self.top_pool}: estrato C ficaria vazio "
                "(o sistema retornaria o banco inteiro, e Recall@K não precisaria de estimativa)"
            )
        if self.n_vagas_calibracao > self.n_vagas_estudo:
            p.append(
                f"N_VAGAS_CALIBRACAO={self.n_vagas_calibracao} > N_VAGAS_ESTUDO="
                f"{self.n_vagas_estudo}: a calibração é um subconjunto do estudo"
            )
        if self.n_vagas_calibracao < self.min_vagas_calibracao:
            p.append(
                f"N_VAGAS_CALIBRACAO={self.n_vagas_calibracao} < mínimo "
                f"{self.min_vagas_calibracao} exigido pelo plano"
            )
        if self.e_calib > 0.10:
            p.append(f"E_CALIB={self.e_calib}: o plano proíbe afrouxar acima de 0,10")
        if self.limiar_relevante not in (1, 2, 3):
            p.append(f"limiar_relevante={self.limiar_relevante}: fora da escala 0-3")
        return p

    def avisos(self) -> list[str]:
        """Não bloqueiam, mas o plano pede atenção."""
        a = []
        if not 40 <= self.n_vagas_estudo <= 100:
            a.append(
                f"N_VAGAS_ESTUDO={self.n_vagas_estudo} fora da faixa 40-100 sugerida "
                "(ajustar conforme runtime/custo de API)"
            )
        if self.n_universo_b < 2 * self.min_substrato_calibracao:
            a.append(
                f"universo B={self.n_universo_b}: pequeno demais para tirar "
                f"{self.min_substrato_calibracao} itens de B na calibração sem repetir vagas"
            )
        if self.k < self.min_substrato_calibracao:
            a.append(
                f"K={self.k} < {self.min_substrato_calibracao}: o mínimo de itens do "
                "substrato A no grupo AB exige juntar mais de uma vaga de calibração"
            )
        return a


config = ConfigAvaliacao()
