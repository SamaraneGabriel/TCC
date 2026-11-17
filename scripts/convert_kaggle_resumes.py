"""
Converte o dataset Kaggle de currículos (snehaanbhawal/resume-dataset) em arquivos .txt,
um por currículo, no formato que scripts/ingest_curriculos.py já processa.

O texto já vem pronto (extraído de PDF), só precisa de limpeza de espaçamento: o CSV usa
sequências de espaços em vez de quebras de linha para separar seções.

Uso:
    python scripts/convert_kaggle_resumes.py [--limit N]

Lê:        data/kaggle/snehaan/Resume/Resume.csv
Escreve em: data/curriculos_kaggle/curriculo_kaggle_NNNNN_categoria.txt
"""
import re
import sys
from pathlib import Path

import pandas as pd

ENTRADA = Path("data/kaggle/snehaan/Resume/Resume.csv")
SAIDA = Path("data/curriculos_kaggle")


def limpar(texto: str) -> str:
    """Normaliza espaçamento sem inventar estrutura: colapsa espaços/tabs repetidos
    (usados no CSV tanto para separar seções quanto células de tabela e até no meio
    de frases) em um único espaço; preserva quebras de linha reais quando existem.
    """
    texto = texto.replace("\t", " ")
    texto = re.sub(r" {2,}", " ", texto)
    linhas = [l.strip() for l in texto.splitlines()]
    texto = "\n".join(linhas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip() + "\n"


def slug(categoria: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", categoria.lower()).strip("-")


def main() -> None:
    limite = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    SAIDA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENTRADA)
    if limite:
        df = df.head(limite)

    total = 0
    for _, linha in df.iterrows():
        total += 1
        caminho = SAIDA / f"curriculo_kaggle_{linha['ID']}_{slug(linha['Category'])}.txt"
        caminho.write_text(limpar(linha["Resume_str"]), encoding="utf-8")

    print(f"{total} currículos convertidos em {SAIDA}/")


if __name__ == "__main__":
    main()
