# Retrieval Evaluation

Saturn retrieves table profiles using UPO-aligned query descriptions. Available retrievers are BM25, vector search, and their hybrid.

Prepare artifacts and indexes as described in the [project README](../../README.md). Then compare the original query and its transformed description:

```bash
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
python -m evaluation.runners.hyde_retrieval -d adventure_works \
    --retriever bm25 --compare-combined
```

`--full-compare` compares the original query, table description, column descriptions, and their combination. `--comprehensive` evaluates these representations across the supported retrievers. These options do not invoke an LLM when using cached query transformations.

For multiple datasets:

```bash
bash source/evaluation/run_retrieval_eval.sh \
    --datasets "adventure_works openwikitable" --retriever "bm25 hybrid"
```

Hit@k counts a query as successful when at least one of its ground-truth answer tables appears in the top k results. The query transformation cache retains all ground-truth tables for multi-answer evaluation.
