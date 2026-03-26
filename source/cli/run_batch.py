#!/usr/bin/env python3
"""
Full UPO Pipeline Runner - Per-Dataset Sequential Execution

Runs the complete pipeline for each dataset sequentially:
    0. Unify Benchmark Data (raw → unified format)
    1. UPO Stage 1 (Federated Primitive TBox)
    2. Layer 2 All (Stage 2+3+4+5)
    3. Retrieval Query Analysis (optional, for evaluation)

Key Features:
    - Each dataset completes ALL stages before moving to the next
    - Independent hyperparameter configuration per stage
    - Resume capability with stage-level skip detection
    - Comprehensive logging and result tracking

Usage:
    # Run all stages for all datasets
    python scripts/run_full_pipeline.py
    
    # Run specific datasets
    python scripts/run_full_pipeline.py --datasets "fetaqa public_bi"
    
    # Skip unify stage (data already in unified format)
    python scripts/run_full_pipeline.py --skip-unify
    
    # Custom Stage 1 parameters
    python scripts/run_full_pipeline.py --stage1-total-queries 200 --stage1-llm gemini
    
    # Dry run - show what would be executed
    python scripts/run_full_pipeline.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Path setup for standalone execution
SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import _path_setup  # noqa: F401 - Sets up source path

from loguru import logger


# ============== Configuration ==============

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "source"
DATA_DIR = PROJECT_ROOT / "data"

# All known datasets
ALL_DATASETS = [
    "adventure_works",
    "fetaqa",
    "fetaqapn",
    "public_bi",
    "chembl",
    "chicago",
    "bird",
    "debug",  # Debug mock dataset for testing
]


@dataclass
class DatasetConfig:
    """Per-dataset configuration."""
    name: str
    # Stage 0: Unify
    unify_train_queries: int = 200
    unify_translate: bool = False
    # Stage 1: Federated Primitive TBox
    stage1_total_queries: int = 200
    stage1_llm_purpose: str = "gemini"
    stage1_cq_max_concurrent: int = 128
    stage1_dp_max_concurrent: int = 16
    stage1_n_iterations: int = 5  # Number of Stage 1 iterations
    stage1_target_classes: int = 50  # Target number of classes for Stage 1
    # Layer 2 All (Stage 2+3+4+5)
    layer2_max_tables: int = -1  # -1 for all
    layer2_llm_purpose: str = "local"
    layer2_batch_size: int = 1000
    layer2_table_max_workers: int = 128
    layer2_analyze_max_workers: int = 32
    layer2_sh_max_workers: int = 2
    layer2_budget_cap: int = 10000
    # TBox iteration selection for Layer 2+
    tbox_iteration: int = -1  # -1 = latest, or specific iteration number
    # Ablation options
    disable_transform_reuse: bool = False  # If True, skip transform reuse (ablation)
    # Index configuration
    index_key: str = "td_cd_cs"  # Index fields: td, td_cd, td_cd_cs
    # Index mode during ingest: which indexes to build
    # Options: 'all' (both faiss+bm25), 'bm25' (only bm25), 'vector' (only faiss)
    index_mode: str = "all"  # Build both indexes for flexibility
    # Per-stage RAG type configuration
    stage1_rag_type: str = "vector"  # Stage 1: vector-only for DPP sampling
    query_analysis_rag_type: str = "vector"  # Query analysis: vector-only
    layer2_rag_type: str = "hybrid"  # Layer 2 (description export): hybrid (both)
    # Query Analysis
    query_analysis_enabled: bool = True
    query_analysis_rag_top_k: int = 3
    query_analysis_no_primitive_classes: bool = False  # Ablation mode
    query_analysis_parallel: int = 64
    # Experiment DB isolation
    db_path: Optional[str] = None  # Override LanceDB path (sets SATURN_DB_PATH)
    output_base_path: Optional[str] = None  # Override index output path


# Default configurations per dataset
DEFAULT_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "adventure_works": DatasetConfig(
        name="adventure_works",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "bird": DatasetConfig(
        name="bird",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "chembl": DatasetConfig(
        name="chembl",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "chicago": DatasetConfig(
        name="chicago",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "fetaqa": DatasetConfig(
        name="fetaqa",
        unify_train_queries=1000,
        stage1_total_queries=200,
    ),
    "fetaqapn": DatasetConfig(
        name="fetaqapn",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "public_bi": DatasetConfig(
        name="public_bi",
        unify_train_queries=200,
        stage1_total_queries=200,
    ),
    "debug": DatasetConfig(
        name="debug",
        unify_train_queries=3,
        stage1_total_queries=5,
    ),
}


@dataclass
class StageResult:
    """Result from a single stage execution."""
    stage: str
    dataset: str
    success: bool
    elapsed: float
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class DatasetResult:
    """Result from processing a complete dataset."""
    dataset: str
    stages: Dict[str, StageResult] = field(default_factory=dict)
    total_elapsed: float = 0.0
    success: bool = True


# ============== Stage Existence Checks ==============

def check_unified_data_exists(dataset: str) -> bool:
    """Check if unified data exists for dataset."""
    unified_dir = DATA_DIR / "benchmark" / "unified" / dataset
    table_dir = unified_dir / "table"
    test_file = unified_dir / "query" / "test.jsonl"
    return table_dir.exists() and test_file.exists()


def check_tbox_exists(dataset: str, db_path: Optional[str] = None) -> bool:
    """Check if TBox exists for dataset (Stage 1 completed).
    
    Verifies BOTH filesystem TBox directory AND LanceDB ontology exist,
    since Layer 2 loads primitive classes from LanceDB, not the filesystem.
    
    Args:
        dataset: Dataset name
        db_path: Override LanceDB path (for experiment isolation)
    """
    # Check filesystem
    tbox_dir = DATA_DIR / "lake" / "tbox"
    if not tbox_dir.exists():
        return False
    pattern = f"federated_{dataset}_*"
    matches = list(tbox_dir.glob(pattern))
    if len(matches) == 0:
        return False
    
    # Also verify LanceDB has the ontology (critical for Layer 2)
    try:
        if db_path:
            from store.store_singleton import create_store
            store = create_store(db_path)
        else:
            from store.store_singleton import get_store
            store = get_store()
        from store.ontology.ontology_table import OntologyTableManager
        mgr = OntologyTableManager(store.db)
        ont = mgr.get_version_by_iteration("federated_primitive_tbox", dataset, -1)
        if not ont:
            ont = mgr.get_latest_version("primitive_tbox", dataset)
        if not ont:
            logger.warning(f"  [stage1] TBox dir exists on disk but NO ontology in LanceDB for {dataset}")
            return False
        return True
    except Exception as e:
        logger.warning(f"  [stage1] Could not verify LanceDB ontology: {e}")
        return False


def check_retrieval_index_exists(
    dataset: str,
    index_key: str = "td_cd_cs",
    output_base_path: Optional[str] = None,
) -> bool:
    """Check if retrieval index exists for dataset (Stage 5 completed).
    
    Args:
        dataset: Dataset name
        index_key: Index configuration key (td, td_cd, td_cd_cs)
        output_base_path: Override base path for index lookup
    """
    if output_base_path:
        index_dir = Path(output_base_path) / "indexes" / dataset / index_key / "faiss"
    else:
        index_dir = DATA_DIR / "lake" / "indexes" / dataset / index_key / "faiss"
    faiss_index = index_dir / "index.faiss"
    return faiss_index.exists()


def check_query_analysis_exists(
    dataset: str,
    llm_suffix: str = "local",
    rag_type: str = "hybrid",
    no_primitive_classes: bool = False,
    db_path: Optional[str] = None,
) -> bool:
    """Check if query analysis cache exists."""
    if db_path:
        eval_dir = Path(db_path) / "eval_results"
    else:
        eval_dir = DATA_DIR / "lake" / "lancedb" / "eval_results"
    
    # New format patterns with rag_type
    pc_suffix = "_no_pc" if no_primitive_classes else ""
    patterns = [
        f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3_{rag_type}{pc_suffix}.json",
        f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag5_{rag_type}{pc_suffix}.json",
    ]
    
    # Legacy patterns (backward compatible) - only check if not using new options
    if rag_type == "hybrid" and not no_primitive_classes:
        patterns.extend([
            f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3.json",
        ])
    
    for p in patterns:
        if (eval_dir / p).exists():
            return True
    return False


def check_ingested_data_exists(dataset: str, db_path: Optional[str] = None) -> bool:
    """Check if data has been ingested to LanceDB (tables and queries).
    
    Uses filesystem check instead of LanceDB API to avoid potential bugs
    with table_names() method.
    
    Args:
        dataset: Dataset name
        db_path: Override LanceDB path (for experiment isolation, checks symlinks)
    """
    try:
        lancedb_dir = Path(db_path) if db_path else DATA_DIR / "lake" / "lancedb"
        
        # Check for tables and train queries lance directories
        tables_name = f"{dataset}_tables_entries.lance"
        train_queries_name = f"{dataset}_train_queries.lance"
        
        tables_path = lancedb_dir / tables_name
        train_queries_path = lancedb_dir / train_queries_name
        
        return tables_path.exists() and train_queries_path.exists()
    except Exception:
        return False


def run_subprocess_streaming(
    cmd: List[str],
    stage_name: str,
    cwd: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> tuple[int, str]:
    """
    Run subprocess with real-time streaming output.
    
    Args:
        cmd: Command and arguments
        stage_name: Stage name for log prefix
        cwd: Working directory
        extra_env: Extra environment variables to set
    
    Returns:
        Tuple of (return_code, error_output)
    """
    error_lines = []
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
        errors='replace',
    )
    
    # Stream output in real-time
    for line in process.stdout:
        # Print with indentation for stage output
        print(f"    [{stage_name}] {line}", end="", flush=True)
        # Capture potential error lines
        if "error" in line.lower() or "exception" in line.lower():
            error_lines.append(line.strip())
    
    process.wait()
    
    error_output = "\n".join(error_lines[-5:]) if error_lines else "Unknown error"
    return process.returncode, error_output


# ============== Stage Runners ==============

def run_unify_stage(
    config: DatasetConfig,
    dry_run: bool = False,
    force: bool = False,
) -> StageResult:
    """
    Stage 0: Convert raw data to unified format.
    """
    stage_name = "unify"
    dataset = config.name
    
    # Check if already done
    if not force and check_unified_data_exists(dataset):
        logger.info(f"  [unify] ✓ Unified data exists, skipping")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Unified data already exists"
        )
    
    cmd = [
        "python", "cli/unify_data.py",
        "--dataset", dataset,
        "--train-queries", str(config.unify_train_queries),
    ]
    if config.unify_translate:
        cmd.append("--translate")
    
    logger.info(f"  [unify] Running: {' '.join(cmd)}")
    
    if dry_run:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Dry run"
        )
    
    start = time.time()
    try:
        returncode, error_output = run_subprocess_streaming(cmd, stage_name, str(SOURCE_DIR))
        elapsed = time.time() - start
        
        if returncode != 0:
            logger.error(f"  [unify] ✗ Failed after {elapsed:.1f}s")
            return StageResult(
                stage=stage_name, dataset=dataset,
                success=False, elapsed=elapsed,
                error=error_output
            )
        
        logger.info(f"  [unify] ✓ Completed in {elapsed:.1f}s")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=elapsed
        )
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  [unify] ✗ Exception: {e}")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=False, elapsed=elapsed, error=str(e)
        )


def run_ingest_stage(
    config: DatasetConfig,
    dry_run: bool = False,
    force: bool = False,
) -> StageResult:
    """
    Stage 0.5: Ingest unified data to LanceDB.
    """
    stage_name = "ingest"
    dataset = config.name
    
    # Check if already done
    if not force and check_ingested_data_exists(dataset, db_path=config.db_path):
        logger.info(f"  [ingest] ✓ LanceDB tables exist, skipping")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="LanceDB tables already exist"
        )
    
    cmd = [
        "python", "cli/ingest_data.py",
        "--dataset", dataset,
        "--index-mode", config.index_mode,
    ]
    
    logger.info(f"  [ingest] Running: {' '.join(cmd)}")
    
    if dry_run:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Dry run"
        )
    
    start = time.time()
    try:
        returncode, error_output = run_subprocess_streaming(cmd, stage_name, str(SOURCE_DIR))
        elapsed = time.time() - start
        
        if returncode != 0:
            logger.error(f"  [ingest] ✗ Failed after {elapsed:.1f}s")
            return StageResult(
                stage=stage_name, dataset=dataset,
                success=False, elapsed=elapsed,
                error=error_output
            )
        
        logger.info(f"  [ingest] ✓ Completed in {elapsed:.1f}s")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=elapsed
        )
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  [ingest] ✗ Exception: {e}")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=False, elapsed=elapsed, error=str(e)
        )


def run_stage1(
    config: DatasetConfig,
    dry_run: bool = False,
    force: bool = False,
) -> StageResult:
    """
    Stage 1: Federated Primitive TBox generation.
    """
    stage_name = "stage1"
    dataset = config.name
    
    # Check if already done
    if not force and check_tbox_exists(dataset, db_path=config.db_path):
        logger.info(f"  [stage1] ✓ TBox exists, skipping")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="TBox already exists"
        )
    
    cmd = [
        "python", "-m", "cli.run_pipeline",
        "--dataset", dataset,
        "--step", "federated_primitive_tbox",
        "--total-queries", str(config.stage1_total_queries),
        "--llm-purpose", config.stage1_llm_purpose,
        "--cq-max-concurrent", str(config.stage1_cq_max_concurrent),
        "--dp-max-concurrent", str(config.stage1_dp_max_concurrent),
        "--n-iterations", str(config.stage1_n_iterations),
        "--target-classes", str(config.stage1_target_classes),
        "--rag-type", config.stage1_rag_type,
    ]
    
    if config.db_path:
        cmd.extend(["--db-path", config.db_path])
    
    logger.info(f"  [stage1] Running: {' '.join(cmd)}")
    
    if dry_run:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Dry run"
        )
    
    start = time.time()
    try:
        returncode, error_output = run_subprocess_streaming(cmd, stage_name, str(SOURCE_DIR))
        elapsed = time.time() - start
        
        if returncode != 0:
            logger.error(f"  [stage1] ✗ Failed after {elapsed:.1f}s")
            return StageResult(
                stage=stage_name, dataset=dataset,
                success=False, elapsed=elapsed,
                error=error_output
            )
        
        logger.info(f"  [stage1] ✓ Completed in {elapsed:.1f}s")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=elapsed
        )
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  [stage1] ✗ Exception: {e}")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=False, elapsed=elapsed, error=str(e)
        )


def run_layer2_all(
    config: DatasetConfig,
    dry_run: bool = False,
    force: bool = False,
    fresh_start: bool = False,
    stats_dir: Optional[str] = None,
) -> StageResult:
    """
    Layer 2 All: Stage 2+3+4+5 (Column Summary, Layer2, Export, Index).
    """
    stage_name = "layer2_all"
    dataset = config.name
    
    # Check if already done (Stage 5 is the final indicator)
    if not force and check_retrieval_index_exists(
        dataset, config.index_key, output_base_path=config.output_base_path,
    ):
        logger.info(f"  [layer2_all] ✓ Retrieval index exists, skipping")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Retrieval index already exists"
        )
    
    cmd = [
        "python", "-m", "cli.run_pipeline",
        "--dataset", dataset,
        "--step", "layer2_all",
        "--max-tables", str(config.layer2_max_tables),
        "--llm-purpose", config.layer2_llm_purpose,
        "--batch-size", str(config.layer2_batch_size),
        "--table-max-workers", str(config.layer2_table_max_workers),
        "--analyze-max-workers", str(config.layer2_analyze_max_workers),
        "--sh-max-workers", str(config.layer2_sh_max_workers),
        "--budget-cap", str(config.layer2_budget_cap),
        "--tbox-iteration", str(config.tbox_iteration),
        "--index-key", config.index_key,
        "--rag-type", config.layer2_rag_type,
    ]
    
    if config.db_path:
        cmd.extend(["--db-path", config.db_path])
    
    if config.output_base_path:
        cmd.extend(["--output-base-path", config.output_base_path])
    
    if fresh_start:
        cmd.append("--fresh")
    
    if config.disable_transform_reuse:
        cmd.append("--disable-reuse")
    
    if stats_dir:
        cmd.extend(["--stats-dir", stats_dir])
    
    logger.info(f"  [layer2_all] Running: {' '.join(cmd)}")
    
    if dry_run:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Dry run"
        )
    
    start = time.time()
    try:
        returncode, error_output = run_subprocess_streaming(cmd, stage_name, str(SOURCE_DIR))
        elapsed = time.time() - start
        
        if returncode != 0:
            logger.error(f"  [layer2_all] ✗ Failed after {elapsed:.1f}s")
            return StageResult(
                stage=stage_name, dataset=dataset,
                success=False, elapsed=elapsed,
                error=error_output
            )
        
        logger.info(f"  [layer2_all] ✓ Completed in {elapsed:.1f}s")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=elapsed
        )
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  [layer2_all] ✗ Exception: {e}")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=False, elapsed=elapsed, error=str(e)
        )


def run_query_analysis(
    config: DatasetConfig,
    dry_run: bool = False,
    force: bool = False,
) -> StageResult:
    """
    Query Analysis: Generate RAG-enhanced query analysis cache.
    """
    stage_name = "query_analysis"
    dataset = config.name
    
    if not config.query_analysis_enabled:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Query analysis disabled"
        )
    
    # Check if already done (with new options)
    if not force and check_query_analysis_exists(
        dataset, 
        config.layer2_llm_purpose,
        rag_type=config.query_analysis_rag_type,
        no_primitive_classes=config.query_analysis_no_primitive_classes,
        db_path=config.db_path,
    ):
        logger.info(f"  [query_analysis] ✓ Analysis cache exists, skipping")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Query analysis cache already exists"
        )
    
    cmd = [
        "python", "-m", "cli.retrieval",
        "--dataset", dataset,
        "--analyze-queries",
        "--use-rag",
        "-n", "-1",  # Process all queries, not just 100
        "--index-key", config.index_key,
        "--rag-top-k", str(config.query_analysis_rag_top_k),
        "--rag-type", config.query_analysis_rag_type,
        "--parallel", str(config.query_analysis_parallel),
        "--llm", config.layer2_llm_purpose,
    ]
    
    # Add --no-primitive-classes if in ablation mode
    if config.query_analysis_no_primitive_classes:
        cmd.append("--no-primitive-classes")
    
    # Pass index-base-path for experiment isolation
    if config.output_base_path:
        cmd.extend(["--index-base-path", config.output_base_path])
    
    # Pass SATURN_DB_PATH via env for experiment isolation
    extra_env = None
    if config.db_path:
        extra_env = {"SATURN_DB_PATH": config.db_path}
    
    logger.info(f"  [query_analysis] Running: {' '.join(cmd)}")
    
    if dry_run:
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=0, skipped=True,
            skip_reason="Dry run"
        )
    
    start = time.time()
    try:
        # Reduce logging verbosity for query analysis
        os.environ["LOGURU_LEVEL"] = "INFO"
        returncode, error_output = run_subprocess_streaming(cmd, stage_name, str(SOURCE_DIR), extra_env=extra_env)
        elapsed = time.time() - start
        
        if returncode != 0:
            logger.error(f"  [query_analysis] ✗ Failed after {elapsed:.1f}s")
            return StageResult(
                stage=stage_name, dataset=dataset,
                success=False, elapsed=elapsed,
                error=error_output
            )
        
        logger.info(f"  [query_analysis] ✓ Completed in {elapsed:.1f}s")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=True, elapsed=elapsed
        )
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  [query_analysis] ✗ Exception: {e}")
        return StageResult(
            stage=stage_name, dataset=dataset,
            success=False, elapsed=elapsed, error=str(e)
        )


# ============== Main Pipeline Runner ==============

def run_dataset_pipeline(
    config: DatasetConfig,
    skip_unify: bool = False,
    skip_ingest: bool = False,
    skip_stage1: bool = False,
    skip_layer2: bool = False,
    skip_query_analysis: bool = False,
    dry_run: bool = False,
    force: bool = False,
    fresh_start: bool = False,
    stats_dir: Optional[str] = None,
) -> DatasetResult:
    """
    Run complete pipeline for a single dataset.
    
    Pipeline stages:
      0.   Unify:     Convert raw data to unified format
      0.5. Ingest:    Ingest unified data into LanceDB
      1.   Stage 1:   Federated Primitive TBox generation
      2.   Layer 2:   Stage 2+3+4+5 (Column Summary, Layer2, Export, Index)
      3.   Query Analysis: RAG-enhanced query analysis cache (optional)
    """
    result = DatasetResult(dataset=config.name)
    total_start = time.time()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {config.name}")
    logger.info(f"{'='*60}")
    
    # Stage 0: Unify
    if not skip_unify:
        stage_result = run_unify_stage(config, dry_run=dry_run, force=force)
        result.stages["unify"] = stage_result
        if not stage_result.success:
            result.success = False
            result.total_elapsed = time.time() - total_start
            return result
    
    # Stage 0.5: Ingest (unified data -> LanceDB)
    if not skip_ingest:
        stage_result = run_ingest_stage(config, dry_run=dry_run, force=force)
        result.stages["ingest"] = stage_result
        if not stage_result.success:
            result.success = False
            result.total_elapsed = time.time() - total_start
            return result
    
    # Stage 1: Federated Primitive TBox
    if not skip_stage1:
        stage_result = run_stage1(config, dry_run=dry_run, force=force)
        result.stages["stage1"] = stage_result
        if not stage_result.success:
            result.success = False
            result.total_elapsed = time.time() - total_start
            return result
    
    # Layer 2 All (Stage 2+3+4+5)
    if not skip_layer2:
        stage_result = run_layer2_all(
            config, dry_run=dry_run, force=force, fresh_start=fresh_start,
            stats_dir=stats_dir,
        )
        result.stages["layer2_all"] = stage_result
        if not stage_result.success:
            result.success = False
            result.total_elapsed = time.time() - total_start
            return result
    
    # Query Analysis
    if not skip_query_analysis:
        stage_result = run_query_analysis(config, dry_run=dry_run, force=force)
        result.stages["query_analysis"] = stage_result
        # Query analysis failure is non-critical
    
    result.total_elapsed = time.time() - total_start
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full UPO Pipeline Runner - Per-Dataset Sequential Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages (per dataset):
  0.   Unify:         Convert raw data to unified format
  0.5. Ingest:        Ingest unified data into LanceDB
  1.   Stage 1:       Federated Primitive TBox generation
  2.   Layer 2 All:   Stage 2+3+4+5 (Column Summary, Layer2, Export, Index)
  3.   Query Analysis: RAG-enhanced query analysis cache (optional)

Examples:
  # Run all datasets with default config
  python scripts/run_full_pipeline.py
  
  # Run specific datasets
  python scripts/run_full_pipeline.py --datasets "fetaqa public_bi"
  
  # Skip unify and ingest stages (data already in LanceDB)
  python scripts/run_full_pipeline.py --skip-unify --skip-ingest
  
  # Custom Stage 1 parameters for all datasets
  python scripts/run_full_pipeline.py --stage1-total-queries 200 --stage1-llm gemini
  
  # Dry run
  python scripts/run_full_pipeline.py --dry-run
  
  # Force re-run (ignore existence checks)
  python scripts/run_full_pipeline.py --force --datasets fetaqa
        """
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Space-separated dataset list (default: all {len(ALL_DATASETS)})"
    )
    
    # Stage skipping
    parser.add_argument("--skip-unify", action="store_true", help="Skip unify stage")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest stage (unified -> LanceDB)")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 (TBox)")
    parser.add_argument("--skip-layer2", action="store_true", help="Skip Layer 2 All")
    parser.add_argument("--skip-query-analysis", action="store_true", help="Skip query analysis")
    
    # Stage 0: Unify parameters
    parser.add_argument("--unify-train-queries", type=int, default=None,
                        help="Train queries for unify (override per-dataset default)")
    parser.add_argument("--unify-translate", action="store_true",
                        help="Enable translation in unify stage")
    
    # Stage 1 parameters
    parser.add_argument("--stage1-total-queries", type=int, default=None,
                        help="Total queries for Stage 1 (override per-dataset default)")
    parser.add_argument("--stage1-llm", type=str, default=None,
                        help="LLM purpose for Stage 1 (default: gemini)")
    parser.add_argument("--stage1-cq-concurrent", type=int, default=None,
                        help="CQ max concurrent for Stage 1")
    parser.add_argument("--stage1-dp-concurrent", type=int, default=None,
                        help="DP max concurrent for Stage 1")
    parser.add_argument("--stage1-n-iterations", type=int, default=None,
                        help="Number of iterations for Stage 1 (default: 5)")
    parser.add_argument("--stage1-target-classes", type=int, default=None,
                        help="Target number of classes for Stage 1 (default: 50)")
    
    # Layer 2 parameters
    parser.add_argument("--layer2-max-tables", type=int, default=None,
                        help="Max tables for Layer 2 (-1 for all)")
    parser.add_argument("--layer2-llm", type=str, default=None,
                        help="LLM purpose for Layer 2 (default: local)")
    parser.add_argument("--layer2-batch-size", type=int, default=None,
                        help="Batch size for Layer 2")
    parser.add_argument("--layer2-fresh", action="store_true",
                        help="Fresh start for Layer 2 (clear transform contracts)")
    parser.add_argument("--disable-reuse", action="store_true",
                        help="Disable transform reuse in Stage 2 (ablation mode)")
    
    # TBox iteration selection
    parser.add_argument("--tbox-iteration", type=int, default=None,
                        help="TBox iteration to use for Stage 2+ (-1 = latest, default: -1)")
    
    # Index configuration
    parser.add_argument("--index-key", type=str, default=None,
                        choices=["td", "td_cd", "td_cd_cs"],
                        help="Index fields: td (table_desc), td_cd (+column_desc), td_cd_cs (full, default)")
    
    # Retrieval configuration
    parser.add_argument("--rag-type", type=str, default=None,
                        choices=["bm25", "vector", "hybrid"],
                        help="Retrieval type for unified search across all stages (default: hybrid)")
    
    # Query analysis parameters
    parser.add_argument("--enable-query-analysis", action="store_true",
                        help="Enable query analysis stage")
    parser.add_argument("--query-analysis-rag-top-k", type=int, default=None,
                        help="RAG top-k for query analysis")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run (ignore existence checks)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file path")
    
    # Experiment DB isolation
    parser.add_argument("--db-path", type=str, default=None,
                        help="Override LanceDB path for all datasets (experiment isolation)")
    parser.add_argument("--output-base-path", type=str, default=None,
                        help="Override base path for Stage 5 index output")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{time:HH:mm:ss}</level> | <level>{level: <8}</level> | <level>{message}</level>",
        level=args.log_level,
        colorize=True,
    )
    if args.log_file:
        logger.add(args.log_file, level="DEBUG")
    
    # Parse datasets
    if args.datasets:
        datasets = args.datasets.split()
        for d in datasets:
            if d not in ALL_DATASETS:
                logger.error(f"Unknown dataset: {d}")
                logger.info(f"Available: {ALL_DATASETS}")
                sys.exit(1)
    else:
        datasets = ALL_DATASETS
    
    # Build configs
    configs = []
    for d in datasets:
        config = DEFAULT_DATASET_CONFIGS.get(d, DatasetConfig(name=d))
        
        # Apply overrides
        if args.unify_train_queries is not None:
            config.unify_train_queries = args.unify_train_queries
        if args.unify_translate:
            config.unify_translate = True
        if args.stage1_total_queries is not None:
            config.stage1_total_queries = args.stage1_total_queries
        if args.stage1_llm is not None:
            config.stage1_llm_purpose = args.stage1_llm
        if args.stage1_cq_concurrent is not None:
            config.stage1_cq_max_concurrent = args.stage1_cq_concurrent
        if args.stage1_dp_concurrent is not None:
            config.stage1_dp_max_concurrent = args.stage1_dp_concurrent
        if args.stage1_n_iterations is not None:
            config.stage1_n_iterations = args.stage1_n_iterations
        if args.stage1_target_classes is not None:
            config.stage1_target_classes = args.stage1_target_classes
        if args.layer2_max_tables is not None:
            config.layer2_max_tables = args.layer2_max_tables
        if args.layer2_llm is not None:
            config.layer2_llm_purpose = args.layer2_llm
        if args.layer2_batch_size is not None:
            config.layer2_batch_size = args.layer2_batch_size
        if args.tbox_iteration is not None:
            config.tbox_iteration = args.tbox_iteration
        if args.index_key is not None:
            config.index_key = args.index_key
        if args.rag_type is not None:
            config.rag_type = args.rag_type
        if args.enable_query_analysis:
            config.query_analysis_enabled = True
        if args.query_analysis_rag_top_k is not None:
            config.query_analysis_rag_top_k = args.query_analysis_rag_top_k
        if args.disable_reuse:
            config.disable_transform_reuse = True
        if args.db_path is not None:
            # Resolve to absolute path for subprocess compatibility (cwd may differ)
            p = Path(args.db_path)
            config.db_path = str(p if p.is_absolute() else PROJECT_ROOT / p)
        if args.output_base_path is not None:
            p = Path(args.output_base_path)
            config.output_base_path = str(p if p.is_absolute() else PROJECT_ROOT / p)
        
        configs.append(config)
    
    # Print header
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " Full UPO Pipeline - Per-Dataset Sequential ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Datasets: {[c.name for c in configs]}")
    logger.info(f"  Skip Unify: {args.skip_unify}")
    logger.info(f"  Skip Ingest: {args.skip_ingest}")
    logger.info(f"  Skip Stage1: {args.skip_stage1}")
    logger.info(f"  Skip Layer2: {args.skip_layer2}")
    logger.info(f"  Skip Query Analysis: {args.skip_query_analysis}")
    logger.info(f"  Dry Run: {args.dry_run}")
    logger.info(f"  Force: {args.force}")
    if args.db_path:
        logger.info(f"  DB Path: {args.db_path}")
    if args.output_base_path:
        logger.info(f"  Output Base Path: {args.output_base_path}")
    logger.info("")
    
    # Auto-setup symlinks when --db-path is provided (experiment isolation)
    if args.db_path:
        db_path = Path(args.db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        main_db = DATA_DIR / "lake" / "lancedb"
        for config in configs:
            ds = config.name
            for suffix in ["tables_entries", "train_queries", "test_queries"]:
                src = main_db / f"{ds}_{suffix}.lance"
                dst = db_path / f"{ds}_{suffix}.lance"
                if src.exists() and not dst.exists():
                    os.symlink(src, dst)
                    logger.debug(f"  Symlinked: {ds}_{suffix}.lance")
        # NOTE: Do NOT symlink cache — CQ cache depends on Stage 1 LLM model,
        # so each experiment variant needs its own independent cache directory.
        logger.info(f"  Experiment DB symlinks ready: {db_path}")

    # Auto-symlink raw indexes when --output-base-path is provided
    if args.output_base_path:
        main_indexes = DATA_DIR / "lake" / "indexes"
        for config in configs:
            ds = config.name
            raw_src = main_indexes / ds / "raw"
            if raw_src.exists():
                raw_dst = Path(args.output_base_path) / "indexes" / ds / "raw"
                raw_dst.parent.mkdir(parents=True, exist_ok=True)
                if not raw_dst.exists():
                    os.symlink(raw_src, raw_dst)
                    logger.debug(f"  Symlinked raw index: {ds}/raw/")
        logger.info(f"  Raw index symlinks ready")
    
    # Create experiment directory for LLM stats
    reuse_tag = "no_reuse" if args.disable_reuse else "with_reuse"
    exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "logs" / "experiments" / f"ablation_{reuse_tag}_{exp_ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    exp_config = {
        "timestamp": exp_ts,
        "datasets": [c.name for c in configs],
        "disable_transform_reuse": args.disable_reuse,
        "skip_unify": args.skip_unify,
        "skip_ingest": args.skip_ingest,
        "skip_stage1": args.skip_stage1,
        "force": args.force,
        "layer2_fresh": args.layer2_fresh,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    logger.info(f"  Experiment dir: {exp_dir}")
    
    # Run pipeline for each dataset
    results: List[DatasetResult] = []
    total_start = time.time()
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\n[{i}/{len(configs)}] Processing: {config.name}")
        
        result = run_dataset_pipeline(
            config,
            skip_unify=args.skip_unify,
            skip_ingest=args.skip_ingest,
            skip_stage1=args.skip_stage1,
            skip_layer2=args.skip_layer2,
            skip_query_analysis=args.skip_query_analysis,
            dry_run=args.dry_run,
            force=args.force,
            fresh_start=args.layer2_fresh,
            stats_dir=str(exp_dir),
        )
        results.append(result)
        
        if not result.success:
            logger.error(f"✗ {config.name} failed, stopping pipeline")
            break
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count
    
    logger.info(f"Total Time: {total_elapsed:.1f}s")
    logger.info(f"Datasets: {success_count} success, {failed_count} failed")
    logger.info("")
    
    for r in results:
        status = "✓" if r.success else "✗"
        logger.info(f"  {status} {r.dataset}: {r.total_elapsed:.1f}s")
        for stage_name, stage_result in r.stages.items():
            if stage_result.skipped:
                logger.info(f"      ○ {stage_name}: skipped ({stage_result.skip_reason})")
            elif stage_result.success:
                logger.info(f"      ✓ {stage_name}: {stage_result.elapsed:.1f}s")
            else:
                logger.info(f"      ✗ {stage_name}: FAILED")
    
    # Exit with error if any failed
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
