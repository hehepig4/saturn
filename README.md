# Saturn

**Scalable Agentic Table Understanding for Retrieval in Natural Language**

Saturn is an expert-inspired, training-free pipeline for natural language (NL)-driven tabular data discovery in data lakes. The key idea is to externalize reusable expert priors — the **knowledge prior** (what semantic concepts to recognize) and the **procedural prior** (how to process and present them) — as an explicit **User Preference-based Ontology (UPO)**, enabling efficient table profiling and retrieval.

## Architecture

Saturn consists of a multi-stage pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | **UPO Conceptualization** | Multi-agent ontology construction via Competency Questions (CQs). Generates Primitive Classes (PCs) with data properties (input specification, output schema). |
| 2 | **UPO Population** | Expert-inspired column profiling with transform reuse. Generates column statistics via LLM-synthesized code, applies successive halving for transform selection. |
| 3 | **Table Annotation** | Generates Defined Classes by reading out, producing column-level semantic descriptions. |
| 4 | **Table Summarization** | Multi-view serialization of table profiles (table description + column descriptions with PC-grounded readouts). |
| 5 | **Retrieval & Indexing** | Builds FAISS vector and BM25 keyword indexes from table profiles. Supports UPO-aligned retrieval via HyDE-style query transformation. |

## Environment Setup

```bash
conda env create -f environment.yml
conda activate saturn
```

### Fonts for Visualization (Optional)

The `source/draw/` module uses matplotlib for generating paper-quality figures. To use Linux Libertine fonts, download them into `source/draw/fonts/`:

```bash
mkdir -p source/draw/fonts
# Download Linux Libertine OTF fonts from https://sourceforge.net/projects/linuxlibertine/
# Place .otf files (LinLibertine_R.otf, LinBiolinum_R.otf, etc.) into source/draw/fonts/
```

## Data Preparation

### Pre-included Example Data

The repository ships with pre-computed pipeline outputs for all 7 benchmark datasets under `data/lake/`, allowing you to inspect results and run evaluations without re-running the full pipeline:

```
data/lake/
├── lancedb/
│   ├── ontology_classes.lance/    # UPO Primitive Classes (2398 rows, 7 datasets × 7 iterations)
│   ├── ontology_metadata.lance/   # Ontology metadata (49 rows)
│   ├── ontology_properties.lance/ # Data properties per PC (338 rows)
│   └── eval_results/              # Retrieval evaluation results
│       ├── {dataset}_test_unified_analysis_*.json   # Per-query analysis (7 files)
│       └── runs/eval_log/         # Evaluation logs & summary (BM25/vector/hybrid × 7 datasets)
└── indexes/
    └── {dataset}/td_cd_cs/bm25/index/
        └── corpus.jsonl           # BM25 index corpus (7 datasets, 44 MB total)
```

The ontology tables can be read via LanceDB Python API:

```python
import lancedb

db = lancedb.connect("data/lake/lancedb")
classes = db.open_table("ontology_classes").to_pandas()
metadata = db.open_table("ontology_metadata").to_pandas()
properties = db.open_table("ontology_properties").to_pandas()
```

### Downloading Benchmark Datasets

Saturn supports the following benchmark datasets:

| Abbreviation | Dataset | Source | Description |
|-----|---------|--------|-------------|
| Ad. | `adventure_works` | Pneuma | AdventureWorks relational database |
| BD. | `bird` | Pneuma | BIRD text-to-SQL benchmark |
| Cc. | `chembl` | Pneuma | ChEMBL bioactivity database |
| Ch. | `chicago` | Pneuma | Chicago Data Portal open data |
| FM. | `fetaqapn` | Pneuma | FeTaQA (Pneuma version) |
| Pb. | `public_bi` | Pneuma | Public BI benchmark |
| FL. | `fetaqa` | Solo | FeTaQA (Solo/Birdie version) |

**Step 1: Download raw data** (Pneuma datasets):

```bash
# Download all Pneuma datasets (~30 GB)
bash cli/download_pneuma_datasets.sh --all

# Download specific datasets
bash cli/download_pneuma_datasets.sh chicago bird

# Check dataset completeness
bash cli/download_pneuma_datasets.sh --check-only
```

Raw data will be placed under `data/raw/{dataset}/`. For the `fetaqa` dataset (Solo format), prepare tables and queries manually following the Solo benchmark setup.

**Step 2: Convert to unified format**:

```bash
# Convert all datasets
bash cli/convert_benchmark.sh

# Convert a single dataset
bash cli/convert_benchmark.sh --dataset bird
```

This produces `data/benchmark/unified/{dataset}/` with standardized table JSONs and train/test query splits.

**Step 3: Ingest into LanceDB**:

```bash
# Ingest a single dataset
python -m cli.ingest_data --dataset fetaqa

# Ingest all datasets
python -m cli.ingest_data --all

# Ingest tables only (skip queries)
python -m cli.ingest_data --all --tables-only

# BM25-only mode (skip embedding computation)
python -m cli.ingest_data --dataset fetaqa --index-mode bm25
```

This converts raw tables into LanceDB storage at `data/lake/lancedb/` and builds initial indexes at `data/lake/indexes/`.

## Configuration

### LLM Configuration

Edit `source/config/llm_models.json` to configure LLM endpoints. Saturn uses a purpose-based routing system:

| Purpose | Role |
|---------|------|
| `default` | Primary LLM for general tasks |
| `local` | Local vLLM/SGLang served model, for population/retrieval |
| `gemini` | Online model for conceptualization|

For local deployment with SGLang:

```bash
python -m sglang.launch_server --model-path <model_path> --port 8000
```

Then configure the `api_config_override` in `llm_models.json`:

```json
{
  "api_config_override": {
    "base_url": "http://localhost:8000/v1",
    "api_key": "API_KEY"
  }
}
```

### Embedding Configuration

Edit `source/config/embedding_models.json`. Default: BGE-M3 (1024-dim, multilingual).

Set the `local_path` field to your model directory (relative to project root, e.g., `model/bge-m3`).

## Pipeline Execution

All commands are run from the **project root** (not `source/`).

### Full Pipeline

```bash
# Run all stages on a dataset
python -m cli.run_pipeline --dataset fetaqa --step all \
    --total-queries 100 --max-tables 50
```

### Individual Stages

```bash
# Stage 1: UPO Conceptualization (Federated Primitive TBox)
python -m cli.run_pipeline --dataset fetaqa --step federated_primitive_tbox \
    --total-queries 100 --agent-cq-capacity 30 --target-classes 50

# Stage 2: UPO Population (Column Summary)
python -m cli.run_pipeline --dataset fetaqa --step column_summary \
    --max-tables 500

# Stage 3: Table Annotation (Layer 2 Defined Classes)
python -m cli.run_pipeline --dataset fetaqa --step layer2_annotation

# Stage 4: Table Summarization
python -m cli.run_pipeline --dataset fetaqa --step summarize

# Stage 2-4 in one shot
python -m cli.run_pipeline --dataset fetaqa --step layer2_all --max-tables 500

# Stage 5: Retrieval Index Building
python -m cli.run_pipeline --dataset fetaqa --step retrieval_index \
    --rag-type hybrid --index-key td_cd_cs
```

### Key Hyperparameters (Paper Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent-cq-capacity` | 30 | $M_{cq}$: Max CQs per agent |
| `--proposal-capacity` | 30 | $M_p$: Max proposals per synthesis |
| `--global-agent-span` | 10 | $M_b$: Max branching factor of PC tree |
| `--target-classes` | 50 | $M_c$: Target number of Primitive Classes |
| `--n-iterations` | 5 | $T$: Proposal-synthesis refinement rounds |
| `--total-queries` | 100 | $\|Q\|$: Sampled queries for conceptualization |
| `--budget-cap` | 10000 | $\hat{B}$: Upper limit on successive halving budget |

### Batch Execution

```bash
# Run Stage 1 on all datasets
bash cli/run_stage1_all.sh

# Run Stage 2-4 on all datasets
bash cli/run_layer2_all.sh
```

## Evaluation

### Retrieval Evaluation

```bash
# Single query search
python -m cli.retrieval -q "Which Illinois election had the most votes?" -d fetaqa

# Batch evaluation (semantic retrieval)
python -m cli.retrieval --eval -d fetaqa -n 100

# Full dual-path evaluation (semantic + structural)
python -m cli.retrieval --eval-full -d fetaqa -n 100
```

### Ablation Experiments

```bash
# Experiment 1: Refinement Iteration Ablation
python -m cli.run_experiment iteration-ablation -d fetaqa --max-iterations 10

# Experiment 2: Query Count Ablation
python -m cli.run_experiment query-ablation -d fetaqa --queries 50 100 200 400

# Experiment 3: Concept Count Ablation
python -m cli.run_experiment concept-ablation -d fetaqa --targets 10 25 50 75 100

# Analyze results
python -m cli.run_experiment analyze -p data/lake/experiments/<exp_dir>/results.json
```

## Project Structure

```
saturn/
├── environment.yml           # Conda environment specification
├── data/
│   └── lake/                 # Pipeline outputs & example data
│       ├── lancedb/          # LanceDB vector database
│       │   ├── ontology_*.lance/   # UPO ontology tables
│       │   └── eval_results/       # HyDE documents & evaluation logs
│       └── indexes/          # Search indexes
│           └── {dataset}/td_cd_cs/bm25/index/  # BM25 corpus per dataset
├── ref/                      # Baseline reproduction scripts & patches
│   ├── birdie/               # BIRDIE (DSI-based table discovery)
│   ├── pneuma/               # Pneuma (LLM summary + hybrid retrieval)
│   └── solo/                 # Solo (dense passage retrieval)
└── source/
    ├── cli/                  # Command-line entry points
    │   ├── run_pipeline.py   # Main pipeline runner (Stages 1-5)
    │   ├── run_experiment.py # Ablation experiment runner
    │   ├── retrieval.py      # Retrieval demo & evaluation
    │   └── ingest_data.py    # Benchmark data ingestion
    ├── config/               # Configuration files
    │   ├── llm_models.json   # LLM endpoint configuration
    │   └── embedding_models.json
    ├── core/                 # Core data types and utilities
    │   ├── datatypes/        # Schema definitions (Table, Column, Query, etc.)
    │   ├── formatting/       # Serialization and formatting
    │   └── identifiers/      # ID generation
    ├── llm/                  # LLM client abstraction
    ├── store/                # Storage layer
    │   ├── lancedb/          # LanceDB vector database operations
    │   ├── ontology/         # OWL ontology (owlready2) operations
    │   └── embedding/        # Embedding model registry
    ├── workflows/            # Pipeline stage implementations
    │   ├── conceptualization/  # Stage 1: Federated UPO Conceptualization
    │   ├── population/         # Stage 2: UPO Population (column profiling)
    │   ├── indexing/           # Stages 3-4
    │   │   ├── annotation/     # Stage 3: Table Annotation (Defined Classes)
    │   │   └── nodes/          # Stage 4: Table Summarization
    │   ├── retrieval/          # Stage 5: Retrieval & Indexing
    │   └── common/             # Shared workflow utilities
    ├── evaluation/           # Evaluation framework
    │   ├── runners/          # Evaluation runners (BM25, vector, HyDE, etc.)
    │   └── baselines/        # Baseline implementations
    ├── draw/                 # Visualization utilities
    └── utils/                # General utilities
```

## Baseline Reproduction

The `ref/` directory contains scripts and patches for reproducing the three baseline systems evaluated in the paper: **BIRDIE** (DSI-based), **Pneuma** (LLM summary + hybrid retrieval), and **Solo** (dense passage retrieval). Each baseline has its own conda environment and follows a consistent setup-convert-evaluate workflow. See [ref/README.md](ref/README.md) for details.
