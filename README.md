# Matching entre currículos e vagas com recuperação semântica

Código do TCC "Sistema de Matching entre Currículos e Vagas baseado em Recuperação Semântica
utilizando Embeddings" (PUC Minas). Um LLM extrai dados estruturados de currículos e vagas,
*templates* os convertem em dois textos (perfil e competências), cada um vetorizado e
indexado no Qdrant; a busca funde os dois *rankings* por RRF e aplica um re-ranqueador.
O estudo compara essa configuração com a busca só no vetor do perfil e com o BM25.

## Instalação

```bash
pip install -e ".[dev,ui,eval]"
cp .env.example .env    # preencher GEMINI_API_KEY ou GCP_PROJECT (Vertex AI)
pytest                  # testes unitários, sem chamadas de API
```

Modelos usados no estudo: `gemini-2.5-flash-lite` (extração), `gemini-embedding-001`
(768 dimensões) e `gemini-3.1-flash-lite` (juiz). O Qdrant roda em modo embarcado, em
`.qdrant_storage/`.

## Dados

A pasta `data/` não é versionada. Os conjuntos públicos usados são:

- currículos: [Annotated NER PDF Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes);
- vagas: [Software Engineer Job Postings (LinkedIn)](https://www.kaggle.com/datasets/asaniczka/software-engineer-job-postings-linkedin).

## Sistema

```bash
python scripts/convert_meyhaar_resumes.py                  # currículos -> data/curriculos_meyhaar/*.txt
python scripts/ingest_curriculos.py data/curriculos_meyhaar # extração + embeddings + Qdrant
python scripts/dedup_curriculos.py data/curriculos_meyhaar  # deduplicação (simulação por padrão)
streamlit run app_streamlit.py                              # interface de busca
```

## Estudo de avaliação

Na ordem em que foi executado:

```bash
python scripts/eval/fase0_confirmar_parametros.py   # parâmetros e pré-requisitos
python scripts/eval/preparar_pool_vagas.py          # conjunto fixo de 5.000 vagas
python scripts/eval/sortear_vagas_estudo.py         # 60 vagas, estratificadas
python scripts/eval/baixar_onet.py                  # vocabulário O*NET 31.0
python scripts/eval/gerar_rubricas.py               # rubrica KSAO por vaga
python scripts/eval/montar_amostras.py              # amostras e vagas de calibração
streamlit run app_anotacao.py                       # anotação cega do autor
python scripts/eval/rodar_juiz.py                   # juiz LLM: 60 vagas x 372 currículos
python scripts/eval/gerar_rankings.py               # rankings de C1, C2, SC e BM25
python scripts/eval/concordancia_juiz.py            # concordância juiz x autor
python scripts/eval/conferir_ordem_configuracoes.py # ordem das configurações na calibração
python scripts/eval/comparar_configuracoes.py       # nDCG@25, Wilcoxon, bootstrap, Holm
```

Nos arquivos de dados, a configuração chamada C2 no artigo (perfil + competências, RRF)
aparece como `C3`.
