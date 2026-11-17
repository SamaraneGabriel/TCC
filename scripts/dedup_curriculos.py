"""
Deduplicação do índice de currículos: encontra o mesmo candidato indexado mais
de uma vez, grava a origem no payload e (só com --aplicar) remove os excedentes.

Uso:
    python scripts/dedup_curriculos.py <dir_curriculos> [<dir2> ...] [--aplicar]
                                       [--relatorio data/eval/dedup_relatorio.json]

    Sem --aplicar: dry-run. Analisa, imprime o resumo e escreve o relatório;
                   não escreve nada no Qdrant.
    Com --aplicar: grava `arquivo_fonte`/`conteudo_hash` no payload dos pontos
                   que ficam e apaga os pontos duplicados, um a um.

⚠️ --aplicar apaga pontos. Pela regra do topo do CLAUDE.md, isso exige
autorização explícita do usuário no momento da execução. O script nunca apaga
nem recria a coleção, e aborta se o plano pedir para remover mais da metade
dos pontos (sinal de que o casamento com os arquivos-fonte deu errado).

Os diretórios passados devem conter os textos que originaram o índice — sem
eles o sinal de arquivo-fonte não existe e sobra só nome+email.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.dedup import PontoIndexado, chave_nome, indexar_arquivos, montar_plano
from src.vectorstore.store import CurriculoStore

LIMITE_SEGURANCA = 0.5  # fração máxima do índice que o script aceita remover


def _pontos_do_store(store: CurriculoStore) -> list[PontoIndexado]:
    return [
        PontoIndexado(
            id=pid,
            nome=payload.get("nome"),
            email=payload.get("email"),
            skills=tuple(payload.get("skills_tecnicas") or ()),
            localidade=payload.get("localidade"),
            senioridade=payload.get("senioridade_estimada"),
        )
        for pid, payload in store.listar_pontos()
    ]


def _resumo(plano, pontos_por_id) -> None:
    print(f"\npontos no índice: {plano.total_pontos}")
    print(f"grupos duplicados: {len(plano.grupos)}")
    print(f"pontos a remover:  {len(plano.remover)}")
    print(f"corpus após dedup: {plano.total_apos} currículos únicos")

    por_motivo: dict[tuple, int] = {}
    for g in plano.grupos:
        por_motivo[g.motivos] = por_motivo.get(g.motivos, 0) + 1
    print("\ngrupos por sinal:")
    for motivos, n in sorted(por_motivo.items()):
        print(f"  {' + '.join(motivos) or '(nenhum)'}: {n}")

    tamanhos: dict[int, int] = {}
    for g in plano.grupos:
        tamanhos[g.tamanho] = tamanhos.get(g.tamanho, 0) + 1
    print("tamanhos de grupo:", {k: tamanhos[k] for k in sorted(tamanhos)})

    divergentes = [
        g for g in plano.grupos
        if len({chave_nome(pontos_por_id[p].nome)
                for p in (g.canonico, *g.duplicatas)
                if chave_nome(pontos_por_id[p].nome)}) > 1
    ]
    print(f"grupos com nomes divergentes (revisar): {len(divergentes)}")
    for g in divergentes[:5]:
        print(f"  {[pontos_por_id[p].nome for p in (g.canonico, *g.duplicatas)]}")

    print("\n-- 5 maiores grupos --")
    for g in plano.grupos[:5]:
        c = pontos_por_id[g.canonico]
        arq = plano.correspondencias[g.canonico].arquivos[:2]
        print(f"  {g.tamanho}x {c.nome!r} <{c.email}> canônico={g.canonico} arq={list(arq)}")


def _salvar_relatorio(plano, pontos_por_id, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    grupos = []
    for g in plano.grupos:
        grupos.append({
            "canonico": g.canonico,
            "duplicatas": list(g.duplicatas),
            "motivos": list(g.motivos),
            "membros": [
                {
                    "id": pid,
                    "nome": pontos_por_id[pid].nome,
                    "email": pontos_por_id[pid].email,
                    "n_skills": len(pontos_por_id[pid].skills),
                    "arquivos": list(plano.correspondencias[pid].arquivos),
                    "score": plano.correspondencias[pid].score,
                    "confiavel": plano.correspondencias[pid].confiavel,
                }
                for pid in (g.canonico, *g.duplicatas)
            ],
        })
    caminho.write_text(
        json.dumps(
            {
                "total_pontos": plano.total_pontos,
                "total_apos": plano.total_apos,
                "remover": list(plano.remover),
                "grupos": grupos,
                "sem_correspondencia_confiavel": [
                    pid for pid, c in plano.correspondencias.items() if not c.confiavel
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nrelatório: {caminho}")


def _aplicar(store: CurriculoStore, plano, pontos_por_id) -> None:
    fracao = len(plano.remover) / max(plano.total_pontos, 1)
    if fracao > LIMITE_SEGURANCA:
        print(
            f"\nABORTADO: o plano removeria {fracao:.0%} do índice "
            f"(limite de segurança: {LIMITE_SEGURANCA:.0%}). Revise o relatório."
        )
        sys.exit(1)

    duplicatas_do_canonico = {g.canonico: list(g.duplicatas) for g in plano.grupos}
    remover = set(plano.remover)

    print(f"\napagando {len(remover)} ponto(s) duplicado(s)...")
    store.remover_pontos(list(remover))

    print("gravando origem no payload dos pontos que ficaram...")
    gravados = 0
    for pid in pontos_por_id:
        if pid in remover:
            continue
        c = plano.correspondencias[pid]
        store.atualizar_payload(
            pid,
            {
                "arquivo_fonte": c.arquivos[0] if (c.confiavel and c.arquivos) else None,
                "arquivos_fonte": list(c.arquivos) if c.confiavel else [],
                "conteudo_hash": c.conteudo_hash if c.confiavel else None,
                "dedup_match_score": c.score,
                "dedup_absorvidos": duplicatas_do_canonico.get(pid, []),
            },
        )
        gravados += 1
    print(f"payload atualizado em {gravados} ponto(s)")
    print(f"total na coleção agora: {store.count()}")


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    aplicar = "--aplicar" in sys.argv
    relatorio = Path(
        sys.argv[sys.argv.index("--relatorio") + 1]
        if "--relatorio" in sys.argv
        else "data/eval/dedup_relatorio.json"
    )

    caminhos = [p for d in dirs for p in sorted(d.glob("*.txt"))]
    if not caminhos:
        print(f"nenhum .txt encontrado em {[str(d) for d in dirs]}")
        sys.exit(1)
    arquivos = indexar_arquivos(caminhos)
    print(f"arquivos-fonte: {len(caminhos)} ({len(arquivos)} conteúdos únicos)")

    store = CurriculoStore(path=settings.qdrant_local_path)
    pontos = _pontos_do_store(store)
    pontos_por_id = {p.id: p for p in pontos}
    print(f"casando {len(pontos)} ponto(s) com {len(arquivos)} conteúdo(s)... (alguns minutos)")

    plano = montar_plano(pontos, arquivos)
    _resumo(plano, pontos_por_id)
    _salvar_relatorio(plano, pontos_por_id, relatorio)

    if aplicar:
        _aplicar(store, plano, pontos_por_id)
    else:
        print("\n(dry-run — nada foi escrito no Qdrant; use --aplicar para executar)")


if __name__ == "__main__":
    main()
