"""Baixa os arquivos da base O*NET usados como vocabulário da rubrica (desvio D3).

    python scripts/eval/baixar_onet.py [--versao 31.0] [--dir data/onet] [--forcar]

Baixa dois CSVs públicos do O*NET Resource Center (sem autenticação):

- `software_skills.csv`  — tecnologias por ocupação. **Nome mudou na versão 31.0**:
  até a 30.x o arquivo se chamava `Technology Skills`. Se um dia voltar a mudar,
  é aqui que se corrige.
- `occupation_data.csv`  — código SOC -> título da ocupação, usado para validar
  o mapa `AREA_PARA_SOC` e para nomear as ocupações no log do estudo.

Idempotente: arquivo que já existe não é rebaixado, a menos que `--forcar`. O
download é registrado em `onet_download.json` (versão, URL, bytes, data), que
vira o artefato de proveniência da taxonomia citada na monografia.
"""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.plano.onet import (  # noqa: E402
    ARQUIVO_OCUPACOES,
    ARQUIVO_TECNOLOGIAS,
    DIR_PADRAO,
    VERSAO_ONET,
    VocabularioONet,
)

BASE_URL = "https://www.onetcenter.org/dl_files/database/db_{versao}_csv/{arquivo}"
# O O*NET responde 403 a user-agent vazio/urllib padrão.
USER_AGENT = "Mozilla/5.0 (compatible; TCC-matching research script)"


def url_do_arquivo(versao: str, arquivo: str) -> str:
    return BASE_URL.format(versao=versao.replace(".", "_"), arquivo=arquivo)


def baixar(url: str, destino: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resposta:
        dados = resposta.read()
    destino.write_bytes(dados)
    return len(dados)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versao", default=VERSAO_ONET, help=f"versão da base O*NET (padrão: {VERSAO_ONET})")
    ap.add_argument("--dir", default=str(DIR_PADRAO), help=f"diretório de destino (padrão: {DIR_PADRAO})")
    ap.add_argument("--forcar", action="store_true", help="rebaixa mesmo se o arquivo já existir")
    args = ap.parse_args()

    destino = Path(args.dir)
    destino.mkdir(parents=True, exist_ok=True)

    registro = {"versao_onet": args.versao, "data": datetime.now().isoformat(timespec="seconds"), "arquivos": []}
    print(f"O*NET {args.versao} -> {destino}\n")

    for arquivo in (ARQUIVO_TECNOLOGIAS, ARQUIVO_OCUPACOES):
        caminho, url = destino / arquivo, url_do_arquivo(args.versao, arquivo)
        if caminho.exists() and not args.forcar:
            tamanho, acao = caminho.stat().st_size, "já existia"
        else:
            try:
                tamanho, acao = baixar(url, caminho), "baixado"
            except urllib.error.HTTPError as e:
                print(f"  [ERRO] {arquivo}: HTTP {e.code} em {url}")
                if e.code == 404:
                    print("         O O*NET renomeia arquivos entre versões — confira os nomes em")
                    print(f"         https://www.onetcenter.org/dictionary/{args.versao}/csv/")
                return 1
            except urllib.error.URLError as e:
                print(f"  [ERRO] {arquivo}: {e.reason}")
                return 1

        sha = hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]
        print(f"  [OK ] {arquivo:24s} {tamanho:>9,} bytes  sha256:{sha}  ({acao})")
        registro["arquivos"].append(
            {"arquivo": arquivo, "url": url, "bytes": tamanho, "sha256_16": sha, "acao": acao}
        )

    (destino / "onet_download.json").write_text(
        json.dumps(registro, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # Só faz sentido declarar sucesso se o mapa área->SOC fecha contra o que foi baixado.
    vocab = VocabularioONet.carregar(destino)
    problemas = vocab.problemas()
    print()
    if problemas:
        print("MAPA ÁREA -> SOC COM PROBLEMAS:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    resumo = vocab.resumo()
    print("MAPA ÁREA -> SOC (todas as ocupações conferidas contra a base)")
    for area, dados in resumo["areas"].items():
        print(f"  {area:32s} {dados['tecnologias_hot']:4d} hot / {dados['tecnologias_total']:4d} tecnologias")
        for soc in dados["socs"]:
            print(f"      {soc}")

    (destino / "onet_resumo.json").write_text(
        json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRegistro em {destino / 'onet_download.json'} e {destino / 'onet_resumo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
