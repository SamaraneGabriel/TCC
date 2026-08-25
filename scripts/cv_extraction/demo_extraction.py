"""
Demo script — test CV extraction locally without running the API.

Usage:
    # Use built-in sample CV text:
    python scripts/demo_extraction.py

    # Extract from a real PDF:
    python scripts/demo_extraction.py path/to/curriculo.pdf

Provider is read from .env (LLM_PROVIDER=gemini|openai|ollama).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.extraction.curriculo_extractor import CurriculoExtractor
from src.extraction.factory import create_extractor

SAMPLE_TEXT = """\
Ana Paula Costa
ana.costa@email.com | (11) 91234-5678 | São Paulo, SP

SOBRE
Engenheira de dados com 6 anos de experiência em pipelines de dados e machine learning.

STACK TÉCNICO
Python, Apache Spark, dbt, Airflow, BigQuery, PostgreSQL, Docker, Kubernetes

EXPERIÊNCIA PROFISSIONAL
Senior Data Engineer | Fintech XYZ | Mar 2021 – presente (36 meses)
- Construção de pipelines de dados com Spark e Airflow
- Modelagem dimensional com dbt no BigQuery
- Redução de 40% no custo de processamento

Data Engineer | Startup ABC | Jan 2019 – Feb 2021 (26 meses)
- ETL com Python e PostgreSQL
- Dashboards analíticos com Metabase

FORMAÇÃO
Engenharia de Computação | USP | Concluído 2018

CERTIFICAÇÕES
Google Professional Data Engineer | Google | 2022

IDIOMAS
Português (nativo), Inglês (fluente), Espanhol (básico)

Disponibilidade: 30 dias | Modelo preferido: híbrido
"""


def main() -> None:
    if len(sys.argv) > 1:
        from src.parsers.pdf_parser import parse_pdf

        pdf_path = sys.argv[1]
        print(f"Parsing PDF: {pdf_path}")
        text = parse_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from PDF.\n")
    else:
        text = SAMPLE_TEXT
        print("No PDF provided — using built-in sample CV.\n")

    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {getattr(settings, f'{settings.llm_provider}_model', '?')}")
    print("Extracting...\n")

    llm = create_extractor(settings)
    extractor = CurriculoExtractor(llm)
    result = extractor.extract(text)
    with open('extracted_cv_test.txt', 'w') as f:
        f.write(result.model_dump_json(indent=2, exclude_none=False))

    print("=== Extracted CV ===")
    print(result.model_dump_json(indent=2, exclude_none=False))


if __name__ == "__main__":
    main()
