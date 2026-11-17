"""Vocabulário O*NET de tecnologias, usado como apoio à categoria
`ferramenta_tecnologia` da rubrica (seção 3.1 do plano, desvio D3).

O que este módulo é e o que não é
---------------------------------
A seção 3.1 manda usar as tecnologias do perfil O*NET da ocupação mais próxima
da vaga **como vocabulário de referência, nunca como fonte do requisito em si**.
O conteúdo da rubrica sai sempre do texto real da vaga. Por isso aqui não há
nada que "gere requisitos": o módulo só carrega a taxonomia oficial e responde
"quais tecnologias o O*NET associa a esta área técnica", para o prompt de
geração da rubrica classificar itens com um vocabulário estável em vez de
inventar rótulos a cada chamada.

Base de dados
-------------
O*NET 31.0, arquivos CSV públicos `software_skills.csv` e `occupation_data.csv`
(baixados por `scripts/eval/baixar_onet.py` para `data/onet/`). Atenção ao nome:
até a versão 30 o arquivo se chamava *Technology Skills*; na 31.0 virou
*Software Skills*.

Cada linha de `software_skills.csv` é (ocupação, tecnologia concreta, classe
UNSPSC da tecnologia, flags `Hot Technology`/`In Demand`):

    O*NET-SOC Code,Title,Workplace Example,Element ID,Element Name,Hot Technology,In Demand
    15-1252.00,Software Developers,Apache Kafka,43.23.21.00,Data base user interface...,Y,N

`Workplace Example` é a tecnologia ("Apache Kafka"); `Element Name` é a classe
("Data base user interface and query software"). As duas servem à rubrica: a
primeira como termo, a segunda como categoria.

Mapeamento área → ocupação
--------------------------
`AREA_PARA_SOC` é fixo e auditável de propósito (desvio D3): deixar o LLM
escolher a ocupação por conta própria tornaria o vocabulário não determinístico
entre execuções, e a rubrica precisa ser reproduzível. As chaves são exatamente
as áreas de `pool_vagas.AREAS_TECNICAS` mais `AREA_PADRAO` — `problemas()`
verifica isso, então acrescentar uma área lá sem mapear aqui vira erro visível,
não silêncio.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

from .pool_vagas import AREA_PADRAO, AREAS_TECNICAS

ARQUIVO_TECNOLOGIAS = "software_skills.csv"
ARQUIVO_OCUPACOES = "occupation_data.csv"
DIR_PADRAO = Path("data/onet")
VERSAO_ONET = "31.0"

# Área técnica do pool -> ocupações O*NET-SOC. Mais de uma quando a área real
# não cabe numa ocupação só (Full-stack é web + software; DevOps mistura
# administração de sistemas com arquitetura).
AREA_PARA_SOC: dict[str, tuple[str, ...]] = {
    "Backend": ("15-1252.00", "15-1251.00"),               # Software Developers, Computer Programmers
    "Frontend": ("15-1254.00", "15-1255.00"),              # Web Developers, Web and Digital Interface Designers
    "Full-stack": ("15-1252.00", "15-1254.00"),
    "DevOps/Infra": ("15-1244.00", "15-1299.08", "15-1241.00"),  # Sysadmins, Systems Engineers/Architects, Network Architects
    "Embarcados/Firmware": ("17-2061.00", "15-1252.00"),   # Computer Hardware Engineers
    "IA/ML": ("15-2051.00", "15-1221.00"),                 # Data Scientists, Computer and Information Research Scientists
    "Segurança": ("15-1212.00", "15-1299.04", "15-1299.05"),  # InfoSec Analysts, Penetration Testers, InfoSec Engineers
    "Dados/BI": ("15-2051.01", "15-1243.00", "15-1242.00"),   # BI Analysts, Database Architects, DBAs
    "Mobile": ("15-1252.00",),
    "QA/Testes": ("15-1253.00",),                          # Software QA Analysts and Testers
    "Jogos": ("15-1255.01", "15-1252.00"),                 # Video Game Designers
    "Gestão/Produto": ("11-3021.00", "15-1299.09"),        # CIS Managers, IT Project Managers
    AREA_PADRAO: ("15-1252.00", "15-1299.08"),
}


class BaseONetAusente(FileNotFoundError):
    """Os CSVs do O*NET não estão em disco — rodar `scripts/eval/baixar_onet.py`."""


@dataclass(frozen=True)
class Tecnologia:
    """Uma linha de `software_skills.csv`, já tipada."""

    nome: str        # "Workplace Example": a tecnologia concreta
    categoria: str   # "Element Name": a classe UNSPSC a que ela pertence
    soc: str
    hot: bool        # marcada pelo O*NET como "Hot Technology"
    em_demanda: bool  # "In Demand"


class VocabularioONet:
    """Tecnologias do O*NET indexadas por ocupação, consultáveis por área técnica."""

    def __init__(
        self,
        tecnologias_por_soc: dict[str, list[Tecnologia]],
        titulos: dict[str, str],
        versao: str = VERSAO_ONET,
    ) -> None:
        self._por_soc = tecnologias_por_soc
        self._titulos = titulos
        self.versao = versao

    # ------------------------------------------------------------------
    # Carga

    @classmethod
    def carregar(cls, diretorio: Path | str = DIR_PADRAO) -> "VocabularioONet":
        d = Path(diretorio)
        f_tec, f_ocu = d / ARQUIVO_TECNOLOGIAS, d / ARQUIVO_OCUPACOES
        faltando = [str(f) for f in (f_tec, f_ocu) if not f.exists()]
        if faltando:
            raise BaseONetAusente(
                f"base O*NET não encontrada ({', '.join(faltando)}). "
                "Rode: python scripts/eval/baixar_onet.py"
            )

        titulos: dict[str, str] = {}
        with f_ocu.open(encoding="utf-8", newline="") as fh:
            for linha in csv.DictReader(fh):
                titulos[linha["O*NET-SOC Code"]] = linha["Title"]

        por_soc: dict[str, list[Tecnologia]] = {}
        with f_tec.open(encoding="utf-8", newline="") as fh:
            for linha in csv.DictReader(fh):
                soc = linha["O*NET-SOC Code"]
                por_soc.setdefault(soc, []).append(
                    Tecnologia(
                        nome=linha["Workplace Example"].strip(),
                        categoria=linha["Element Name"].strip(),
                        soc=soc,
                        hot=linha["Hot Technology"].strip().upper() == "Y",
                        em_demanda=linha["In Demand"].strip().upper() == "Y",
                    )
                )
        return cls(por_soc, titulos)

    # ------------------------------------------------------------------
    # Consulta

    def ocupacoes(self, area: str) -> tuple[str, ...]:
        """Códigos SOC mapeados para a área. Área desconhecida cai no padrão —
        não é erro: o pool pode ganhar uma área nova, e `problemas()` é quem
        cobra o mapeamento explícito."""
        return AREA_PARA_SOC.get(area, AREA_PARA_SOC[AREA_PADRAO])

    def titulo(self, soc: str) -> str:
        return self._titulos.get(soc, soc)

    def tecnologias(
        self, area: str, apenas_hot: bool = True, limite: int | None = None
    ) -> list[Tecnologia]:
        """Tecnologias das ocupações da área, sem repetir nome.

        `apenas_hot=True` é o default porque o perfil completo de Software
        Developers tem 430 tecnologias — vocabulário grande demais para caber
        num prompt sem afogar o texto da vaga, que é a fonte real do requisito.
        As ~150 marcadas como *Hot Technology* pelo O*NET cobrem o que aparece
        em vaga real. Ordena por (hot, em_demanda, nome) para o corte por
        `limite` ser determinístico, nunca dependente da ordem do arquivo.
        """
        vistos: dict[str, Tecnologia] = {}
        for soc in self.ocupacoes(area):
            for t in self._por_soc.get(soc, ()):
                if apenas_hot and not t.hot:
                    continue
                # Mesma tecnologia em duas ocupações: fica a primeira, mas as
                # flags são acumuladas (hot numa ocupação vale para a área).
                anterior = vistos.get(t.nome)
                if anterior is None:
                    vistos[t.nome] = t
                elif (t.hot, t.em_demanda) > (anterior.hot, anterior.em_demanda):
                    vistos[t.nome] = t
        ordenado = sorted(vistos.values(), key=lambda t: (not t.hot, not t.em_demanda, t.nome))
        return ordenado[:limite] if limite else ordenado

    def termos(self, area: str, apenas_hot: bool = True, limite: int | None = None) -> list[str]:
        """Só os nomes, prontos para entrar no prompt."""
        return [t.nome for t in self.tecnologias(area, apenas_hot, limite)]

    def categorias(self, area: str, apenas_hot: bool = False) -> list[str]:
        """Classes UNSPSC da área (ex.: "Web platform development software"),
        ordenadas. Complementam os termos: descrevem o *tipo* de ferramenta, que
        é o que a categoria `ferramenta_tecnologia` da rubrica classifica."""
        return sorted({t.categoria for t in self.tecnologias(area, apenas_hot)})

    # ------------------------------------------------------------------
    # Consistência

    def problemas(self) -> list[str]:
        """Inconsistências entre o mapa, o pool de vagas e a base baixada.
        Lista vazia = o vocabulário pode ser usado."""
        p: list[str] = []

        areas_do_pool = {nome for nome, _ in AREAS_TECNICAS} | {AREA_PADRAO}
        nao_mapeadas = sorted(areas_do_pool - AREA_PARA_SOC.keys())
        if nao_mapeadas:
            p.append(f"áreas do pool sem SOC mapeado: {', '.join(nao_mapeadas)}")
        sobrando = sorted(AREA_PARA_SOC.keys() - areas_do_pool)
        if sobrando:
            p.append(f"áreas mapeadas que não existem no pool: {', '.join(sobrando)}")

        for area, socs in sorted(AREA_PARA_SOC.items()):
            for soc in socs:
                if soc not in self._titulos:
                    p.append(f"{area}: SOC {soc} não existe em {ARQUIVO_OCUPACOES}")
                elif not self._por_soc.get(soc):
                    p.append(f"{area}: SOC {soc} ({self.titulo(soc)}) não tem tecnologia listada")
        return p

    def resumo(self) -> dict:
        """Panorama por área, para o log do estudo."""
        return {
            "versao_onet": self.versao,
            "areas": {
                area: {
                    "socs": [f"{soc} {self.titulo(soc)}" for soc in socs],
                    "tecnologias_hot": len(self.tecnologias(area, apenas_hot=True)),
                    "tecnologias_total": len(self.tecnologias(area, apenas_hot=False)),
                    "categorias": len(self.categorias(area)),
                }
                for area, socs in sorted(AREA_PARA_SOC.items())
            },
        }
