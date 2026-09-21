# Saturn: Agentic Natural Language-Driven Tabular Data Discovery

**VLDB 2027**

Saturn constructs a User Preference-based Ontology (UPO) from table evidence and natural-language query requirements. The UPO guides reusable cell transformations and table-profile synthesis. At query time, Saturn produces a UPO-aligned description and searches the table profiles using BM25, vector retrieval, or their combination.

## Pipeline

| Stage | Purpose | Implementation |
| --- | --- | --- |
| Conceptualization | Build the class hierarchy and procedural specifications through CQ-driven multi-agent collaboration. | Stage 1 |
| Population | Ground columns, normalize cells with reusable transforms, and synthesize column and table profiles. | Stages 2-4 |
| Retrieval | Build profile indexes and perform UPO-aligned query transformation and retrieval. | Stage 5 and retrieval CLI |

## Setup

The supplied Conda environment targets Linux. A CUDA-capable GPU is recommended for BGE-M3 embeddings. Cached BM25 evaluation does not require an LLM endpoint or an embedding model.

```bash
conda env create -f environment.yml
conda activate saturn
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
```

Run the commands below from the repository root. Baselines use separate environments described in [ref/README.md](ref/README.md).

## Evaluate Precomputed Profiles

The artifact bundle contains one final UPO per dataset, the table profiles, and cached query transformations for all eight benchmarks. The UPO tables contain **8 ontologies, 378 classes, and 378 properties**. Files are stored as Parquet or compressed JSON with SHA-256 checksums in [data/artifacts/manifest.json](data/artifacts/manifest.json).

Import the artifacts and build a BM25 index:

```bash
python source/cli/prepare_artifacts.py
python source/cli/run_pipeline.py --dataset adventure_works \
    --step retrieval_index --rag-type bm25 --index-key td_cd_cs
python source/cli/retrieval.py --eval --dataset adventure_works \
    --rag-type bm25 --llm local --num-queries -1
```

The importer keeps existing tables and files unless `--force` is explicitly supplied. Runtime databases and indexes are stored in `data/lake/`. Original cell-level data and model weights are downloaded separately.

For vector or hybrid evaluation, configure BGE-M3 in `source/config/embedding_models.json`, then build the corresponding indexes:

```bash
python source/cli/run_pipeline.py --dataset adventure_works \
    --step retrieval_index --rag-type hybrid --index-key td_cd_cs
python source/cli/retrieval.py --eval --dataset adventure_works \
    --rag-type hybrid --llm local --num-queries -1
```

To compare the original query with its UPO-aligned description:

```bash
python -m evaluation.runners.hyde_retrieval --dataset adventure_works \
    --retriever bm25 --compare-combined
```

## Benchmarks

| Abbreviation | Dataset identifier | Data source |
| --- | --- | --- |
| Ad | `adventure_works` | AdventureWorks, distributed with Pneuma |
| Ch | `chembl` | ChEMBL, distributed with Pneuma |
| Pb | `public_bi` | Public BI, distributed with Pneuma |
| FL | `fetaqa` | FeTaQA, Solo format |
| FM | `fetaqapn` | FeTaQA, Pneuma format |
| BD | `bird` | BIRD, distributed with Pneuma |
| Cc | `chicago` | Chicago Data Portal, distributed with Pneuma |
| Ow | `openwikitable` | Open-WikiTable, following Birdie's retrieval setup |

### Data Preparation

Download the Pneuma datasets:

```bash
bash source/cli/download_pneuma_datasets.sh --all
bash source/cli/download_pneuma_datasets.sh --check-only
```

Raw inputs are stored in `data/benchmark/raw/{dataset}/`. For FeTaQA in Solo format, provide `tables.jsonl` and the query files under `queries/`.

**OpenWikiTable (Ow).** [Open-WikiTable](https://github.com/sean0042/Open_WikiTable) is built from WikiSQL and WikiTableQuestions. Following [Birdie](https://github.com/ZJU-DAILY/BIRDIE), we use its table chunks for retrieval. The benchmark contains **54,282 tables** and **6,602 test queries**, with an average of **6.6 columns** and **8.1 rows** per table.

Place `splitted_tables.json`, `train.json`, `valid.json`, and `test.json` from Open-WikiTable under `data/benchmark/raw/openwikitable/`. The converter preserves the original splits and uses stable `owt-{chunk_id}` table IDs. Retrieval answer tables are the deduplicated union of `hard_positive_idx` and `positive_idx`. The original answer categories are also retained in query metadata.

Ow's full training split has 53,819 queries. The pipeline uses a budget of **200 training queries** for conceptualization, separately from data conversion. The query IDs associated with the bundled UPOs are recorded in the artifact manifest.

Convert and ingest a dataset:

```bash
python source/cli/unify_data.py --dataset openwikitable
python source/cli/ingest_data.py --dataset openwikitable
```

Use `--index-mode bm25` during ingestion to omit embedding computation. The batch runner uses vector indexes for CQ sampling and query-transformation context by default.

## Model Configuration

Copy `.env.example` to `.env` and set the credentials needed for your endpoints:

```bash
cp .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Gemini access through OpenRouter |
| `SATURN_LLM_BASE_URL` | OpenAI-compatible population/retrieval endpoint, default `http://localhost:8000/v1` |
| `SATURN_LLM_API_KEY` | Authentication for that endpoint |

`source/config/llm_models.json` maps `gemini` to Gemini 3 Flash Preview for conceptualization and `local` to Qwen3-Next-80B-A3B-Instruct for population and query transformation. Model names and endpoints are configurable. Serve the local model separately with an OpenAI-compatible inference server.

For vector retrieval, download `BAAI/bge-m3` to `model/bge-m3/`, or set its location in `source/config/embedding_models.json`. Credentials, downloaded models, and runtime output are excluded from Git.

## Run from Raw Data

The batch runner performs conversion, ingestion, conceptualization, population, indexing, and query transformation:

```bash
python source/cli/run_batch.py --datasets "adventure_works openwikitable" --dry-run
python source/cli/run_batch.py --datasets "adventure_works openwikitable"
```

It uses 200 queries, a target of 50 classes, and five refinement rounds. Agent CQ capacity, proposal capacity, and aggregation fan-out default to 30, 30, and 10. Use `--help` for stage-specific settings and resume controls.

Individual stages are also available:

```bash
python source/cli/run_pipeline.py --dataset adventure_works \
    --step federated_primitive_tbox --llm-purpose gemini \
    --total-queries 200 --target-classes 50 --n-iterations 5
python source/cli/run_pipeline.py --dataset adventure_works \
    --step layer2_all --llm-purpose local
```

Search with UPO-aligned query transformation:

```bash
python source/cli/retrieval.py --dataset adventure_works \
    --query "Which products have the highest sales?" --llm local --use-rag
```

For cached evaluation on an entire dataset, generate query transformations once:

```bash
python source/cli/retrieval.py --dataset adventure_works \
    --analyze-queries --num-queries -1 --llm local --use-rag --rag-type vector
```

Pipeline outputs include provider-reported input/output token counts and cache statistics. Conceptualization, indexing, and query-time usage are recorded separately. Cache hits are not counted as billed model calls.

## Experiments and Baselines

The experiment runner supports refinement-round, query-budget, and concept-budget ablations:

```bash
python source/cli/run_experiment.py iteration-ablation -d adventure_works --max-iterations 10
python source/cli/run_experiment.py query-ablation -d adventure_works --queries 50 100 200 400
python source/cli/run_experiment.py concept-ablation -d adventure_works --targets 10 25 50 75 100
```

[ref/README.md](ref/README.md) describes the Birdie, Pneuma, and Solo interfaces. Supplementary experiments and analysis are available in [additional_report.pdf](additional_report.pdf).

## Layout

- `source/cli/`: data preparation, pipeline execution, artifact loading, and retrieval.
- `source/workflows/`: conceptualization, population, profile synthesis, and retrieval.
- `source/config/`: model and pipeline settings.
- `source/evaluation/`: retrieval evaluation and baseline utilities.
- `data/artifacts/`: final UPOs, table profiles, and cached query transformations.
- `ref/`: baseline setup scripts and adapters.

## Acknowledgments

We thank the authors of [Birdie](https://github.com/ZJU-DAILY/BIRDIE), [Pneuma](https://github.com/TheDataStation/Pneuma), and [Solo](https://github.com/TheDataStation/solo) for making their code and benchmarks publicly available. Their open-source contributions supported our implementation and evaluation.
