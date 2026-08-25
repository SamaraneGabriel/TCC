"""Prints de debug ao longo das pipelines. Desligado por padrão — ativar com DEBUG=true no .env."""
from .config import settings


def dbg(etapa: str, msg: str) -> None:
    if settings.debug:
        print(f"[DEBUG] {etapa}: {msg}")
