# Command-Line Tools

Run commands from the repository root. See the [project README](../../README.md) for setup and the full workflow.

| Entry point | Purpose |
| --- | --- |
| `prepare_artifacts.py` | Verify and import the bundled UPOs, profiles, and cached query transformations. |
| `unify_data.py` | Convert benchmark inputs to table JSON and query JSONL. |
| `ingest_data.py` | Load tables and queries into LanceDB and build initial indexes. |
| `run_batch.py` | Run conversion, ingestion, conceptualization, population, indexing, and query transformation. |
| `run_pipeline.py` | Run selected pipeline stages. |
| `retrieval.py` | Transform queries, search table profiles, or evaluate cached transformed queries. |
| `run_experiment.py` | Run refinement-round, query-budget, and concept-budget experiments. |

Each entry point accepts `--help`. Batch scripts accept dataset selections so a single benchmark can be run independently.
