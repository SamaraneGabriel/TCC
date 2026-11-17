"""Anotação humana da amostra de calibração (§4.3 passos 2-3) — estado, ordem e
persistência. Lógica pura: nada aqui importa Streamlit, a interface em
`app_anotacao.py` só chama estas funções.

Três invariantes que o protocolo exige e que este módulo é responsável por manter:

**Cegueira.** O anotador não pode ver estrato, posição no ranking, score, nem a
nota do LLM (§4.3 passo 2 — ancoragem inflaria o kappa artificialmente). Os
campos continuam na planilha porque as métricas precisam deles, mas
`item_para_anotar()` devolve só o que pode ser exibido. A interface nunca lê a
linha crua.

**Ordem embaralhada e independente por anotador.** Cada anotador recebe a sua
própria ordem, derivada por SHA-256 de (seed_mestre, anotador) — assim o segundo
anotador do double-label não percorre os itens na mesma sequência do primeiro,
o que poderia correlacionar fadiga e ordem entre os dois e enviesar o kappa
humano-humano.

**Gravação a cada item.** Anotar 194 itens leva horas e ninguém termina numa
sentada. Cada resposta reescreve o CSV inteiro (194 linhas — barato) em vez de
acumular em memória, então fechar o navegador não perde nada.
"""
import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

from .amostragem import CAMPOS_PLANILHA_ANOTACAO
from .config import ConfigAvaliacao
from .rubrica import RubricaVaga

ANOTADORES = ("humano_1", "humano_2")

# Onde ficam os textos originais dos currículos (resolvidos por `arquivo_fonte`).
# Uma lista só para a interface de anotação e o juiz LLM: os dois precisam ler
# exatamente o mesmo texto.
DIRS_CURRICULOS = ("data/curriculos_meyhaar", "data/curriculos_kaggle", "data/conhecidos")
ATENDIMENTOS = ("atende", "parcial", "nao_atende")
NOTAS = (0, 1, 2, 3)

ROTULOS_NOTA = {
    0: "não relevante",
    1: "parcialmente relevante",
    2: "relevante",
    3: "altamente relevante",
}

CAMPOS_CHECKLIST = ["vaga_id", "curriculo_id", "item_id", "anotador", "atendimento", "data"]

Anotador = Literal["humano_1", "humano_2"]
Atendimento = Literal["atende", "parcial", "nao_atende"]


# ----------------------------------------------------------------------
# Planilha de anotação (§7.1)


def carregar_planilha(caminho: Path | str) -> list[dict]:
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def salvar_planilha(linhas: list[dict], caminho: Path | str) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_PLANILHA_ANOTACAO)
        w.writeheader()
        w.writerows({c: linha.get(c, "") for c in CAMPOS_PLANILHA_ANOTACAO} for linha in linhas)


def chave(linha: dict) -> tuple[str, str]:
    return (linha["vaga_id"], linha["curriculo_id"])


def nota_de(linha: dict, anotador: str) -> int | None:
    bruto = linha.get(f"nota_{anotador}", "")
    return int(bruto) if bruto not in (None, "") else None


def registrar_nota(
    linhas: list[dict],
    vaga_id: str,
    curriculo_id: str,
    anotador: Anotador,
    nota: int,
    observacoes: str = "",
) -> dict:
    """Grava a nota do anotador na linha correspondente e devolve a linha.

    `fonte_nota_final` é marcada como `humano` — na amostra de calibração é a
    nota humana que vale, mesmo quando o LLM também já julgou o item (§8.2).
    """
    if nota not in NOTAS:
        raise ValueError(f"nota {nota} fora da escala 0-3 (§6)")
    if anotador not in ANOTADORES:
        raise ValueError(f"anotador {anotador!r} desconhecido")

    for linha in linhas:
        if chave(linha) == (vaga_id, curriculo_id):
            linha[f"nota_{anotador}"] = str(nota)
            linha["fonte_nota_final"] = "humano"
            if observacoes:
                linha["observacoes"] = observacoes
            return linha
    raise KeyError(f"par ({vaga_id}, {curriculo_id}) não está na planilha")


# ----------------------------------------------------------------------
# Checklist de atendimento da rubrica (§7.4, desvio D4)


@dataclass(frozen=True)
class LinhaChecklist:
    vaga_id: str
    curriculo_id: str
    item_id: str
    anotador: str
    atendimento: str
    data: str


def carregar_checklist(caminho: Path | str) -> dict[tuple[str, str, str], LinhaChecklist]:
    """Indexado por (curriculo_id, item_id, anotador) — a vaga já está implícita
    no `item_id`, mas fica na linha para o CSV ser legível sozinho."""
    caminho = Path(caminho)
    if not caminho.exists():
        return {}
    registros = {}
    with caminho.open(encoding="utf-8", newline="") as fh:
        for linha in csv.DictReader(fh):
            reg = LinhaChecklist(**{c: linha[c] for c in CAMPOS_CHECKLIST})
            registros[(reg.curriculo_id, reg.item_id, reg.anotador)] = reg
    return registros


def salvar_checklist(
    registros: dict[tuple[str, str, str], LinhaChecklist], caminho: Path | str
) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_CHECKLIST)
        w.writeheader()
        w.writerows(asdict(r) for r in registros.values())


def registrar_checklist(
    registros: dict[tuple[str, str, str], LinhaChecklist],
    vaga_id: str,
    curriculo_id: str,
    anotador: Anotador,
    atendimentos: dict[str, str],
) -> None:
    """Grava (ou sobrescreve) o atendimento item a item de um par vaga-currículo."""
    agora = datetime.now().isoformat(timespec="seconds")
    for item_id, atendimento in atendimentos.items():
        if atendimento not in ATENDIMENTOS:
            raise ValueError(f"atendimento {atendimento!r} inválido para {item_id}")
        registros[(curriculo_id, item_id, anotador)] = LinhaChecklist(
            vaga_id=vaga_id,
            curriculo_id=curriculo_id,
            item_id=item_id,
            anotador=anotador,
            atendimento=atendimento,
            data=agora,
        )


def checklist_do_item(
    registros: dict[tuple[str, str, str], LinhaChecklist],
    curriculo_id: str,
    anotador: Anotador,
    rubrica: RubricaVaga,
) -> dict[str, str]:
    """Atendimentos já registrados deste par, na ordem da rubrica."""
    return {
        item.item_id: registros[(curriculo_id, item.item_id, anotador)].atendimento
        for item in rubrica.itens
        if (curriculo_id, item.item_id, anotador) in registros
    }


def nota_sugerida(rubrica: RubricaVaga, atendimentos: dict[str, str]) -> int | None:
    """Nota que a escala da §6 implica, dado o checklist preenchido.

    É **sugestão**, não imposição: a §6 diz "todos ou quase todos", e o que conta
    como "quase" depende do caso. A interface mostra a sugestão e o anotador
    confirma ou muda — o que a torna útil como rede contra desatenção sem
    substituir o julgamento, que é o que o protocolo pede.

    Devolve None enquanto faltar item por marcar.
    """
    if not rubrica.itens or len(atendimentos) < len(rubrica.itens):
        return None

    obrigatorios = rubrica.obrigatorios
    if not obrigatorios:
        return None

    def pontos(itens) -> float:
        return sum(
            {"atende": 1.0, "parcial": 0.5, "nao_atende": 0.0}[atendimentos[i.item_id]]
            for i in itens
        )

    razao_obrig = pontos(obrigatorios) / len(obrigatorios)
    desejaveis = rubrica.desejaveis
    razao_desej = pontos(desejaveis) / len(desejaveis) if desejaveis else 0.0

    if razao_obrig >= 0.999 and razao_desej > 0:
        return 3
    if razao_obrig >= 0.8:
        return 2
    if razao_obrig > 0 or razao_desej > 0:
        return 1
    return 0


# ----------------------------------------------------------------------
# Ordem de apresentação e double-label


def _seed(*partes) -> int:
    material = ":".join(str(p) for p in partes).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def ordem_do_anotador(linhas: list[dict], anotador: Anotador, seed_mestre: int) -> list[dict]:
    """Ordem de apresentação, embaralhada e própria de cada anotador.

    Ordena por (vaga_id, curriculo_id) antes de embaralhar: a ordem em que as
    linhas estão no CSV já carrega informação de estrato (o arquivo é gravado
    grupo AB primeiro, depois grupo C), e usá-la como ponto de partida deixaria
    resíduo dessa informação na sequência mostrada.
    """
    base = sorted(linhas, key=chave)
    rng = random.Random(_seed(seed_mestre, "ordem", anotador))
    rng.shuffle(base)
    return base


def selecionar_double_label(
    linhas: list[dict], cfg: ConfigAvaliacao, seed_mestre: int
) -> set[tuple[str, str]]:
    """§4.3 passo 3: sorteia `n_double_label` itens da calibração para a segunda
    anotação humana independente. Determinístico e estável — o mesmo conjunto em
    qualquer execução, para o segundo anotador poder começar noutro dia."""
    base = sorted({chave(l) for l in linhas})
    n = min(cfg.n_double_label, len(base))
    return set(random.Random(_seed(seed_mestre, "double_label")).sample(base, n))


def itens_do_anotador(
    linhas: list[dict], anotador: Anotador, cfg: ConfigAvaliacao, seed_mestre: int
) -> list[dict]:
    """O que cada anotador tem para fazer: `humano_1` anota os 194; `humano_2`,
    só os sorteados para double-label."""
    ordem = ordem_do_anotador(linhas, anotador, seed_mestre)
    if anotador == "humano_1":
        return ordem
    alvo = selecionar_double_label(linhas, cfg, seed_mestre)
    return [l for l in ordem if chave(l) in alvo]


def progresso(linhas: list[dict], anotador: Anotador, cfg: ConfigAvaliacao, seed_mestre: int) -> dict:
    alvo = itens_do_anotador(linhas, anotador, cfg, seed_mestre)
    feitos = [l for l in alvo if nota_de(l, anotador) is not None]
    return {
        "total": len(alvo),
        "feitos": len(feitos),
        "faltam": len(alvo) - len(feitos),
        "percentual": round(100 * len(feitos) / len(alvo), 1) if alvo else 0.0,
        "distribuicao": {
            nota: sum(1 for l in feitos if nota_de(l, anotador) == nota) for nota in NOTAS
        },
    }


def proximo_pendente(
    linhas: list[dict], anotador: Anotador, cfg: ConfigAvaliacao, seed_mestre: int
) -> int | None:
    """Índice, na ordem do anotador, do primeiro item ainda sem nota."""
    alvo = itens_do_anotador(linhas, anotador, cfg, seed_mestre)
    return next((i for i, l in enumerate(alvo) if nota_de(l, anotador) is None), None)


# ----------------------------------------------------------------------
# Item exibido ao anotador (o filtro de cegueira)


@dataclass(frozen=True)
class ItemParaAnotar:
    """Exatamente o que pode aparecer na tela. Sem estrato, sem posição no
    ranking, sem score, sem nota do LLM (§4.3 passo 2)."""

    vaga_id: str
    curriculo_id: str
    nota_atual: int | None
    observacoes: str


def item_para_anotar(linha: dict, anotador: Anotador) -> ItemParaAnotar:
    return ItemParaAnotar(
        vaga_id=linha["vaga_id"],
        curriculo_id=linha["curriculo_id"],
        nota_atual=nota_de(linha, anotador),
        observacoes=linha.get("observacoes") or "",
    )


# ----------------------------------------------------------------------
# Texto do currículo


def indexar_curriculos(diretorios: Iterable[Path | str]) -> dict[str, Path]:
    """Mapa nome-do-arquivo -> caminho, para resolver o `arquivo_fonte` que o
    payload do Qdrant guarda."""
    indice: dict[str, Path] = {}
    for d in diretorios:
        d = Path(d)
        if d.exists():
            for arquivo in d.glob("*.txt"):
                indice.setdefault(arquivo.name, arquivo)
    return indice


def texto_do_curriculo(payload: dict, indice: dict[str, Path]) -> str:
    """Texto original do currículo, para o anotador ler.

    O payload do Qdrant guarda os campos extraídos, não o texto bruto — mas
    guarda `arquivo_fonte`. Julgar pelo texto original, e não pela extração, é
    o certo: a extração é parte do sistema sob avaliação, e anotar sobre ela
    mediria o ranqueador contra si mesmo.
    """
    nome = payload.get("arquivo_fonte")
    caminho = indice.get(nome) if nome else None
    if caminho and caminho.exists():
        return caminho.read_text(encoding="utf-8", errors="replace")
    return ""
