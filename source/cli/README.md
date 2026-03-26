# DataLake Demos

End-to-end workflow demonstrations for the DataLake project.

## Available Demos

### 1. main.py
Main entry point demonstrating the complete data lake workflow.

**Usage:**
```bash
cd source
python demos/main.py
```

### 2. dataset_to_description.py
Complete pipeline from dataset to entity descriptions.

**Workflow:**
1. Dataset Arrival → Dataset Processing (Preprocessing)
2. Data Interaction → Entity Description
3. Output: 10 Entity Descriptions

**Usage:**
```bash
cd source
python demos/dataset_to_description.py
```

### 3. dataset_to_ontology.py
Complete pipeline for extracting comprehensive ontology concepts from datasets.

**Workflow:**
1. Dataset Processing → Load and prepare dataset
2. Central Decision Node → Decide if more exploration is needed
3. Data Interaction (if needed) → Explore dataset to gather understanding
4. Ontology Assignment → Extract and link concepts to Wikidata
5. Loop until ontology coverage is sufficient
6. Output → Comprehensive ontology mapping

**Key Features:**
- Iterative exploration-extraction loop
- LLM-driven decision on exploration sufficiency
- Accumulates ontology concepts across iterations
- Tracks coverage and diversity of concepts

**Usage:**
```bash
cd source
python demos/dataset_to_ontology.py
```

### 4. upo_generation_demo.py
User Preference Ontology (UPO) generation using INQUIRE-Rerank dataset.

**Workflow:**
1. Query Sampling → Select diverse queries from corpus
2. Story Collection → Generate user stories from queries
3. CQ Extraction → Extract competency questions
4. ODP Selection → Select relevant ontology design patterns
5. TBox Generation → Generate initial ontology structure
6. External Alignment → Map to Wikidata entities
7. Verification → Test discrimination capability
8. Refinement → Iterate until quality criteria met

**Key Features:**
- 10-node LangGraph workflow with checkpointing support
- Natural world query understanding (INQUIRE-Rerank dataset)
- Wikidata alignment for semantic enrichment
- OWL Manchester syntax export
- Pellet reasoner integration for consistency checking

**Prerequisites:**
1. INQUIRE dataset ingested into LanceDB:
   ```bash
   cd source
   python scripts/ingest_inquire_rerank.py
   ```

2. Java (OpenJDK 17) for Pellet reasoner:
   ```bash
   java -version  # Should show OpenJDK 17.x
   ```

3. Embedding models configured in `configs/embedding_config.yaml`

**Usage:**
```bash
cd source

# Basic run with defaults (20 queries, 10 samples, 3 iterations)
python demos/upo_generation_demo.py

# Custom parameters
python demos/upo_generation_demo.py --queries 10 --sample 5 --iterations 2

# Enable checkpointing for debugging
python demos/upo_generation_demo.py --checkpointer sync --durability exit

# See all options
python demos/upo_generation_demo.py --help
```

**Outputs:**
- `data/lake/upo_demo_results/inquire_upo.owl` - OWL ontology in Manchester syntax
- `data/lake/upo_demo_results/trajectory.json` - Complete workflow execution trace
- `data/lake/upo_demo_results/summary.json` - Key metrics and statistics
- `data/lake/upo_demo_results/README.md` - Results documentation

## Prerequisites

Make sure you have:
1. Activated the `saturn` conda environment
2. Set up required API keys in environment variables or `.env` file
3. Prepared your dataset configuration files in `source/config/`

## Configuration

Demos use configuration files from:
- `source/config/` - Dataset configurations
- `configs/` - Global settings (LLM, paths, etc.)
- Environment variables - API keys and overrides

See `configs/README.md` for configuration details.
