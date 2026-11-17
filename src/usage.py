"""Contabilização de tokens consumidos nas chamadas de LLM/embedding.

Acumulador em memória, por modelo, alimentado pelos providers (`extraction/providers/gemini.py`,
`embedding/providers/gemini.py`) a partir do `usage_metadata` que o SDK devolve em cada resposta.
Serve para responder "quanto esta execução gastou" sem depender do faturamento do GCP, que só
consolida horas depois.

Thread-safe: a ingestão concorrente (`--concurrency N`) roda as chamadas síncronas em thread pool.
"""
import threading
from dataclasses import dataclass


@dataclass
class UsoModelo:
    chamadas: int = 0
    prompt_tokens: int = 0
    resposta_tokens: int = 0
    total_tokens: int = 0
    caracteres_faturaveis: int = 0  # embeddings: algumas respostas só reportam caracteres


_lock = threading.Lock()
_por_modelo: dict[str, UsoModelo] = {}


def registrar(
    modelo: str,
    prompt_tokens: int = 0,
    resposta_tokens: int = 0,
    total_tokens: int = 0,
    caracteres_faturaveis: int = 0,
) -> None:
    with _lock:
        uso = _por_modelo.setdefault(modelo, UsoModelo())
        uso.chamadas += 1
        uso.prompt_tokens += prompt_tokens or 0
        uso.resposta_tokens += resposta_tokens or 0
        uso.total_tokens += total_tokens or 0
        uso.caracteres_faturaveis += caracteres_faturaveis or 0


def resumo() -> dict[str, UsoModelo]:
    with _lock:
        return {modelo: UsoModelo(**vars(uso)) for modelo, uso in _por_modelo.items()}


def formatar_resumo() -> str:
    linhas = ["Tokens consumidos (por modelo):"]
    for modelo, uso in sorted(resumo().items()):
        partes = [f"{uso.chamadas} chamada(s)"]
        if uso.total_tokens:
            partes.append(
                f"{uso.total_tokens} tokens (entrada {uso.prompt_tokens} + saída {uso.resposta_tokens})"
            )
        if uso.caracteres_faturaveis:
            partes.append(f"{uso.caracteres_faturaveis} caracteres faturáveis")
        linhas.append(f"  {modelo}: {', '.join(partes)}")
    if len(linhas) == 1:
        linhas.append("  (nenhuma chamada registrada)")
    return "\n".join(linhas)
