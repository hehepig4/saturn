# Baseline Reproduction

The adapters in this directory run Birdie, Pneuma, and Solo on the eight unified Saturn benchmarks. Each baseline uses its own environment and keeps the original implementation in a separate directory.

## Setup

Run from the Saturn repository root:

```bash
bash ref/birdie/setup.sh
bash ref/pneuma/setup.sh
bash ref/solo/setup_solo.sh
```

The setup scripts download the upstream repositories into `birdie/`, `pneuma/`, and `solo/` by default. Model weights and generated files are excluded from Git.

Prepare datasets under `data/benchmark/unified/{dataset}/` using the [main data-preparation instructions](../README.md#data-preparation). All adapters use the same table IDs and ground-truth answer tables.

## Birdie

Birdie builds hierarchical semantic IDs from title/schema and table-content embeddings, generates training queries, and trains an mT5 retriever. The interface supports both its table-specific query generator and an off-the-shelf generator. These are separate configurations.

```bash
bash ref/birdie/evaluate.sh --dataset openwikitable \
    --use-tllama-vllm --lora-max-samples 200
```

This configuration needs `kingb/Llama-3-8B-table-base` under `model/kingb/Llama-3-8B-table-base/` and a Python environment with vLLM installed. Set `BIRDIE_VLLM_PYTHON` to that environment's Python executable. `--lora-max-samples` limits the human-query examples used to train the query generator. The script trains a missing adapter and launches a dedicated local server. `CUDA_VISIBLE_DEVICES` controls the visible GPUs. Generated queries are mapped to the semantic IDs of their source tables before retriever training.

Alternatively, train the adapter with `ref/birdie/train_query_generator.sh` and serve it separately. Use `--use-tllama --vllm-base-url URL --tllama-lora PATH` to connect to that endpoint, with the adapter directory name as its served model name. Without `--use-tllama`, the script uses the off-the-shelf model specified by `--vllm-model`. The two variants have separate dataset and output directories. Use `--help` for the remaining settings.

## Pneuma

Pneuma generates schema narrations and sampled-row summaries, then supports BM25, vector, or hybrid retrieval with optional LLM reranking.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL=Qwen3-Next-80B-A3B-Instruct
bash ref/pneuma/evaluate.sh --dataset openwikitable
```

Replace `EMPTY` with the authentication required by your model endpoint. Indexing and reranking record provider-reported input/output token counts.

## Solo

Solo converts tables into passages, constructs a dense index, and trains its retriever.

```bash
bash ref/solo/train_and_evaluate.sh --dataset openwikitable
```

The converter writes queries under `query/{split}/` to match Solo's trainer and evaluator. `--passage-limit` sets a resource limit for passage generation. Exceeding this limit skips the run and does not produce a retrieval score.

## Evaluation

Baseline outputs are evaluated at the table level. Queries may have multiple ground-truth answer tables. Keep dataset splits and ID mappings fixed when comparing methods. Full baseline runs can require substantial GPU memory, training time, and model-serving capacity.
