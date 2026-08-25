"""
Demo script — test vaga extraction locally without running the API.

Usage:
    python scripts/vaga_extraction/demo_extraction.py

Provider is read from .env (LLM_PROVIDER=gemini|openai|ollama).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.extraction.factory import create_extractor
from src.extraction.vaga_extractor import VagaExtractor

SAMPLE_TEXT = """\
Desenvolvedor(a) Backend Python - Pleno
FinTech ABC | São Paulo, SP | Híbrido

SOBRE A VAGA
Buscamos dev backend para atuar no time de pagamentos, desenvolvendo e mantendo
APIs REST de alta disponibilidade.

RESPONSABILIDADES
- Desenvolver e manter APIs REST com FastAPI
- Participar de code reviews
- Contribuir com a arquitetura de microsserviços

REQUISITOS OBRIGATÓRIOS
Python, FastAPI, PostgreSQL, Docker

DIFERENCIAIS
Redis, Kafka, AWS, Kubernetes

Experiência mínima: 3 anos
Formação: graduação em Ciência da Computação ou áreas afins

Regime: CLT
"""


def main() -> None:
    text = SAMPLE_TEXT
    print("Usando vaga de exemplo embutida no script.\n")

    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {getattr(settings, f'{settings.llm_provider}_model', '?')}")
    print("Extracting...\n")

    llm = create_extractor(settings)
    extractor = VagaExtractor(llm)
    result = extractor.extract(text)

    print("=== Extracted Vaga ===")
    print(result.model_dump_json(indent=2, exclude_none=False))


if __name__ == "__main__":
    main()
