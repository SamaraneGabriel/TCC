"""Detecção de currículos duplicados no índice e plano de deduplicação.

Por que existe: `data/curriculos_meyhaar/` vem de um dataset público que repete
currículos — o mesmo texto salvo em arquivos diferentes e, em alguns casos, a
mesma pessoa em versões de texto ligeiramente distintas. Como a ingestão gerava
um UUID aleatório por arquivo e não guardava a origem no payload, nada colapsava:
o mesmo candidato ocupava várias posições do ranking, enviesando Precision@K e
nDCG@K do plano de avaliação (`context/10_PLANO_AVALIACAO.md`, seção 8).

Dois sinais independentes de duplicata, unidos por union-find:

1. **arquivo-fonte**: o ponto é rastreado de volta ao `.txt` que o originou
   (hash do texto normalizado); dois pontos com o mesmo hash são o mesmo
   currículo. Como o payload não guarda a origem, a rastreabilidade é
   reconstruída por correspondência de e-mail/nome/skills contra o texto.
2. **nome+email**: pega a mesma pessoa em arquivos diferentes (versões
   distintas do currículo) e os casos em que a extração do LLM variou entre
   duas ingestões do mesmo texto.

Funções puras: recebem os pontos já lidos do store e os textos já lidos do
disco. Quem toca Qdrant é `scripts/dedup_curriculos.py`.
"""
import hashlib
import re
import unicodedata
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Pesos do casamento ponto -> arquivo-fonte. E-mail é quase identificador único;
# nome é forte mas repete; a fração de skills presentes no texto é o único sinal
# disponível para os currículos sem nome nem e-mail extraídos (~20% do corpus).
PESO_EMAIL = 5.0
PESO_NOME = 3.0
PESO_SKILLS = 4.0
PESO_LOCALIDADE = 1.0

SCORE_MINIMO = 3.0   # abaixo disso a evidência é fraca demais pra afirmar a origem
MARGEM_MINIMA = 0.5  # distância pro segundo melhor arquivo, pra não escolher no empate
MIN_CHARS_SKILL = 4  # "C", "R", "SQL" casam com qualquer texto; só contam skills longas


def normalizar(texto: str) -> str:
    """NFKC + minúsculas + colapso de espaços. Duas extrações do mesmo PDF podem
    diferir só em quebra de linha/espaçamento — sem isso elas não casariam."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", texto).lower()).strip()


def hash_conteudo(texto: str) -> str:
    """Identidade de conteúdo do currículo (sobre o texto já normalizado)."""
    return hashlib.sha256(normalizar(texto).encode("utf-8")).hexdigest()[:16]


def id_deterministico(conteudo_hash: str) -> str:
    """ID do ponto derivado do conteúdo do currículo.

    Torna a ingestão idempotente: reingerir o mesmo texto sobrescreve o mesmo
    ponto em vez de criar outro. Era a ausência disso (UUID aleatório por
    arquivo) que deixava o mesmo candidato ocupando várias posições do ranking.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"curriculo:{conteudo_hash}"))


def chave_nome(nome: str | None) -> str:
    """Nome reduzido a letras/dígitos e espaços, sem acentos.

    O LLM extrai o mesmo nome de formas diferentes conforme a diagramação do PDF
    ("Juan Antonio AGUILAR-GARIB" vs. "Juan Antonio AGUILAR GARIB"); sem isso a
    mesma pessoa vira dois grupos de duplicatas.
    """
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", (nome or "")) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z ]+", " ", sem_acento.lower())).strip()


@dataclass(frozen=True)
class PontoIndexado:
    """Um ponto do Qdrant, reduzido ao que a deduplicação usa."""

    id: str
    nome: str | None = None
    email: str | None = None
    skills: tuple[str, ...] = ()
    localidade: str | None = None
    senioridade: str | None = None


@dataclass(frozen=True)
class ArquivoFonte:
    """Um conteúdo único e todos os arquivos que o contêm (idênticos entre si)."""

    conteudo_hash: str
    arquivos: tuple[str, ...]
    texto_norm: str


@dataclass(frozen=True)
class Correspondencia:
    """Resultado do casamento de um ponto com seu arquivo de origem."""

    ponto_id: str
    conteudo_hash: str | None
    arquivos: tuple[str, ...]
    score: float
    margem: float
    confiavel: bool


@dataclass(frozen=True)
class GrupoDuplicata:
    canonico: str
    duplicatas: tuple[str, ...]
    motivos: tuple[str, ...]  # "arquivo_fonte" e/ou "nome_email"

    @property
    def tamanho(self) -> int:
        return 1 + len(self.duplicatas)


@dataclass(frozen=True)
class PlanoDedup:
    grupos: tuple[GrupoDuplicata, ...] = ()
    correspondencias: dict[str, Correspondencia] = field(default_factory=dict)
    total_pontos: int = 0

    @property
    def remover(self) -> tuple[str, ...]:
        return tuple(pid for g in self.grupos for pid in g.duplicatas)

    @property
    def total_apos(self) -> int:
        return self.total_pontos - len(self.remover)


def indexar_arquivos(caminhos: Iterable[Path]) -> dict[str, ArquivoFonte]:
    """Lê os textos-fonte e agrupa por identidade de conteúdo.

    Arquivos idênticos (depois de normalizados) viram uma entrada só — é por
    isso que empate entre cópias do mesmo texto não conta como ambiguidade.
    """
    textos: dict[str, str] = {}
    arquivos: dict[str, list[str]] = {}
    for caminho in sorted(caminhos):
        texto = normalizar(caminho.read_text(encoding="utf-8", errors="replace"))
        h = hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]
        textos.setdefault(h, texto)
        arquivos.setdefault(h, []).append(caminho.name)
    return {
        h: ArquivoFonte(conteudo_hash=h, arquivos=tuple(arquivos[h]), texto_norm=textos[h])
        for h in textos
    }


def score_correspondencia(ponto: PontoIndexado, texto_norm: str) -> float:
    """Quanto o payload do ponto "explica" o texto do arquivo. O máximo prático
    é PESO_EMAIL + PESO_NOME + PESO_SKILLS + PESO_LOCALIDADE."""
    score = 0.0
    email = (ponto.email or "").strip().lower()
    if email and email in texto_norm:
        score += PESO_EMAIL
    nome = normalizar(ponto.nome or "")
    if len(nome) > 4 and nome in texto_norm:
        score += PESO_NOME
    skills = [normalizar(s) for s in ponto.skills]
    longas = [s for s in skills if len(s) >= MIN_CHARS_SKILL]
    if longas:
        score += PESO_SKILLS * (sum(1 for s in longas if s in texto_norm) / len(longas))
    localidade = normalizar(ponto.localidade or "")
    if len(localidade) > 3 and localidade in texto_norm:
        score += PESO_LOCALIDADE
    return score


def corresponder(
    pontos: Iterable[PontoIndexado],
    arquivos: dict[str, ArquivoFonte],
    score_minimo: float = SCORE_MINIMO,
    margem_minima: float = MARGEM_MINIMA,
) -> dict[str, Correspondencia]:
    """Casa cada ponto com o arquivo-fonte mais provável.

    `confiavel=False` quando a evidência é fraca ou o segundo colocado está
    perto demais (dois currículos parecidos, mas não idênticos) — esses pontos
    não entram no sinal de arquivo-fonte, só no de nome+email.
    """
    resultado: dict[str, Correspondencia] = {}
    itens = list(arquivos.values())
    for ponto in pontos:
        if not itens:
            resultado[ponto.id] = Correspondencia(ponto.id, None, (), 0.0, 0.0, False)
            continue
        notas = sorted(
            ((score_correspondencia(ponto, a.texto_norm), a) for a in itens),
            key=lambda t: (t[0], t[1].conteudo_hash),
            reverse=True,
        )
        melhor_score, melhor = notas[0]
        segundo_score = notas[1][0] if len(notas) > 1 else 0.0
        margem = melhor_score - segundo_score
        resultado[ponto.id] = Correspondencia(
            ponto_id=ponto.id,
            conteudo_hash=melhor.conteudo_hash,
            arquivos=melhor.arquivos,
            score=round(melhor_score, 3),
            margem=round(margem, 3),
            confiavel=melhor_score >= score_minimo and margem >= margem_minima,
        )
    return resultado


class _UnionFind:
    def __init__(self, itens: Iterable[str]):
        self._pai = {i: i for i in itens}

    def find(self, x: str) -> str:
        while self._pai[x] != x:
            self._pai[x] = self._pai[self._pai[x]]
            x = self._pai[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._pai[rb] = ra

    def grupos(self) -> dict[str, list[str]]:
        saida: dict[str, list[str]] = {}
        for item in self._pai:
            saida.setdefault(self.find(item), []).append(item)
        return saida


def _riqueza(ponto: PontoIndexado) -> tuple:
    """Critério do canônico: o payload mais completo sobrevive. O id no fim
    desempata de forma determinística (mesmo corpus -> mesmo canônico)."""
    return (
        bool(ponto.nome),
        bool(ponto.email),
        len(ponto.skills),
        bool(ponto.localidade),
        bool(ponto.senioridade),
        ponto.id,
    )


def escolher_canonico(ids: Iterable[str], pontos: dict[str, PontoIndexado]) -> str:
    return max(ids, key=lambda pid: _riqueza(pontos[pid]))


def montar_plano(
    pontos: Iterable[PontoIndexado],
    arquivos: dict[str, ArquivoFonte],
    correspondencias: dict[str, Correspondencia] | None = None,
) -> PlanoDedup:
    """Une os dois sinais e devolve, por grupo, quem fica e quem sai."""
    pontos = list(pontos)
    por_id = {p.id: p for p in pontos}
    corresp = correspondencias if correspondencias is not None else corresponder(pontos, arquivos)

    uf = _UnionFind(por_id)
    motivo_por_ponto: dict[str, set[str]] = {}

    def juntar(grupo: list[str], motivo: str) -> None:
        for outro in grupo[1:]:
            uf.union(grupo[0], outro)
        for pid in grupo:
            motivo_por_ponto.setdefault(pid, set()).add(motivo)

    por_hash: dict[str, list[str]] = {}
    for pid, c in corresp.items():
        if c.confiavel and c.conteudo_hash:
            por_hash.setdefault(c.conteudo_hash, []).append(pid)
    for grupo in por_hash.values():
        if len(grupo) > 1:
            juntar(sorted(grupo), "arquivo_fonte")

    # Nome sozinho não identifica: quando o currículo é anônimo, o LLM extrai o
    # título do documento como nome ("INFORMATION TECHNOLOGY SPECIALIST"), e dois
    # candidatos distintos acabariam no mesmo grupo. Por isso este sinal exige
    # e-mail; sem e-mail, sobra o sinal de arquivo-fonte.
    por_nome_email: dict[tuple[str, str], list[str]] = {}
    for p in pontos:
        nome, email = chave_nome(p.nome), (p.email or "").strip().lower()
        if nome and email:
            por_nome_email.setdefault((nome, email), []).append(p.id)
    for grupo in por_nome_email.values():
        if len(grupo) > 1:
            juntar(sorted(grupo), "nome_email")

    grupos = []
    for membros in uf.grupos().values():
        if len(membros) < 2:
            continue
        canonico = escolher_canonico(membros, por_id)
        motivos = sorted({m for pid in membros for m in motivo_por_ponto.get(pid, ())})
        grupos.append(
            GrupoDuplicata(
                canonico=canonico,
                duplicatas=tuple(sorted(m for m in membros if m != canonico)),
                motivos=tuple(motivos),
            )
        )
    grupos.sort(key=lambda g: (-g.tamanho, g.canonico))
    return PlanoDedup(grupos=tuple(grupos), correspondencias=corresp, total_pontos=len(pontos))
