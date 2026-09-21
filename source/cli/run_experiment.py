#!/usr/bin/env python3
"""
TBox Ablation Experiment Runner

Runs ablation experiments for evaluating TBox generation quality with isolated LanceDB:
- Experiment 1: Iteration Ablation - Single run to max iterations, extract per-iteration metrics
- Experiment 2: Query Count Ablation - Tests impact of query count on Stage 1
- Experiment 3: Concept Count Ablation - Tests impact of target class count

Each experiment creates an isolated LanceDB directory with symlinks to shared table/query data.

Usage:
    python -m cli.run_experiment iteration-ablation -d fetaqa --max-iterations 10
    python -m cli.run_experiment query-ablation -d fetaqa --queries 50 100 200 400
    python -m cli.run_experiment concept-ablation -d fetaqa --targets 10 25 50 75 100
    python -m cli.run_experiment analyze -p data/lake/experiments/xxx/results.json
    python -m cli.run_experiment list
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import lake_data_path, ensure_project_cwd, get_db_path

ensure_project_cwd()


# ============================================================
# LanceDB Isolation Utilities
# ============================================================

def _get_shared_lance_tables(dataset_name: str) -> List[str]:
    """Get the shared lance table names (tables_entries, queries) for a dataset."""
    return [
        f"{dataset_name}_tables_entries.lance",
        f"{dataset_name}_train_queries.lance",
        f"{dataset_name}_test_queries.lance",
    ]


def setup_experiment_db(experiment_name: str, dataset_name: str, variant_name: str = "") -> Path:
    """
    Create an isolated LanceDB directory for an experiment variant.
    Symlinks shared table/query data from the main DB.
    
    When variant_name is provided, creates per-variant isolation:
      data/lake/experiments/{experiment_name}/{variant_name}/lancedb/
    Otherwise:
      data/lake/experiments/{experiment_name}/lancedb/

    Returns:
        Path to experiment LanceDB directory
    """
    main_db = get_db_path()
    if variant_name:
        exp_db = lake_data_path("experiments", experiment_name, variant_name, "lancedb")
    else:
        exp_db = lake_data_path("experiments", experiment_name, "lancedb")
    exp_db.mkdir(parents=True, exist_ok=True)

    # Create symlinks for shared data
    for table_dir in _get_shared_lance_tables(dataset_name):
        src = main_db / table_dir
        dst = exp_db / table_dir
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
            logger.debug(f"  Symlinked: {table_dir}")

    logger.info(f"Experiment DB ready: {exp_db}")
    return exp_db


def switch_to_experiment_db(exp_db_path: Path) -> None:
    """
    Reset the global StoreManager singleton and point it at exp_db_path.
    All subsequent get_store() calls will use this DB.
    """
    from store.store_singleton import reset_store, get_store
    reset_store()
    get_store(str(exp_db_path))


def restore_main_db() -> None:
    """Restore the global StoreManager to the default DB path."""
    from store.store_singleton import reset_store, get_store
    reset_store()
    get_store(str(get_db_path()))


# ============================================================
# TBox Quality Metrics
# ============================================================

def compute_tbox_metrics(review_history: List[Dict]) -> Dict[str, Any]:
    """
    Compute TBox quality metrics from review history.

    Per-iteration:
      - n_classes, n_agents, avg_coverage_ratio, high_coverage_class_ratio
      - convergence_delta (vs previous iteration)

    Final:
      - final_n_classes, final_avg_coverage, final_high_coverage_ratio
      - total_actions, avg_convergence_delta
    """
    if not review_history:
        return {"error": "No review history available"}

    metrics: Dict[str, Any] = {
        "n_iterations": len(review_history),
        "per_iteration": [],
        "final_metrics": {},
    }

    prev_avg_coverage = None

    for entry in review_history:
        voting_summary = entry.get("voting_summary", [])
        n_classes = entry.get("n_classes", len(voting_summary))
        n_agents = entry.get("n_voting_agents", 0)
        iteration = entry.get("iteration", 0)

        # Compute coverage from raw_votes if voting_summary missing
        if not voting_summary:
            raw_votes = entry.get("raw_votes", {})
            if raw_votes:
                all_classes: set = set()
                for votes in raw_votes.values():
                    all_classes.update(votes.keys())

                coverage_ratios = []
                for cls_name in all_classes:
                    positive = sum(
                        1 for votes in raw_votes.values() if votes.get(cls_name, 0) == 1
                    )
                    coverage_ratios.append(positive / len(raw_votes) if raw_votes else 0)

                avg_coverage = (
                    sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0
                )
                high_coverage_count = sum(1 for r in coverage_ratios if r > 0.5)
                high_coverage_ratio = (
                    high_coverage_count / len(coverage_ratios) if coverage_ratios else 0
                )
                n_classes = len(all_classes)
            else:
                avg_coverage = 0.0
                high_coverage_ratio = 0.0
        else:
            coverage_ratios = [v.get("coverage_ratio", 0) for v in voting_summary]
            avg_coverage = (
                sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0
            )
            high_coverage_count = sum(1 for r in coverage_ratios if r > 0.5)
            high_coverage_ratio = (
                high_coverage_count / len(coverage_ratios) if coverage_ratios else 0
            )

        convergence_delta = (
            abs(avg_coverage - prev_avg_coverage) if prev_avg_coverage is not None else None
        )
        prev_avg_coverage = avg_coverage

        metrics["per_iteration"].append({
            "iteration": iteration,
            "n_classes": n_classes,
            "n_agents": n_agents,
            "avg_coverage_ratio": round(avg_coverage, 4),
            "high_coverage_class_ratio": round(high_coverage_ratio, 4),
            "convergence_delta": round(convergence_delta, 4) if convergence_delta is not None else None,
            "n_actions": entry.get("n_actions", 0),
        })

    if metrics["per_iteration"]:
        final = metrics["per_iteration"][-1]
        metrics["final_metrics"] = {
            "final_n_classes": final["n_classes"],
            "final_avg_coverage": final["avg_coverage_ratio"],
            "final_high_coverage_ratio": final["high_coverage_class_ratio"],
            "total_actions": sum(m["n_actions"] for m in metrics["per_iteration"]),
        }
        deltas = [m["convergence_delta"] for m in metrics["per_iteration"] if m["convergence_delta"] is not None]
        if deltas:
            metrics["final_metrics"]["avg_convergence_delta"] = round(
                sum(deltas) / len(deltas), 4
            )

    return metrics


# ============================================================
# Single-Run Helper
# ============================================================

def _run_single_variant(
    dataset_name: str,
    n_queries: int,
    n_iterations: int,
    n_clusters: int = 0,
    target_classes: int = 50,
    llm_purpose: str = "local",
    experiment_name: str = "",
    variant_name: str = "",
    rag_type: str = "vector",
) -> Dict[str, Any]:
    """
    Run one federated TBox variant inside an isolated experiment DB.

    Steps:
      1. Create isolated lancedb with symlinks
      2. Switch global StoreManager to isolated DB
      3. Run federated TBox workflow
      4. Collect metrics, LLM stats
      5. Save results to experiment directory
      6. Restore main DB
    """
    from workflows.conceptualization import run_federated_tbox
    from llm.statistics import get_usage_stats, reset_usage_stats
    from llm.invoke_with_stats import clear_llm_response_cache
    from workflows.retrieval.unified_similarity import clear_similarity_cache

    # Setup isolated DB (per-variant for multi-variant experiments)
    exp_db = setup_experiment_db(experiment_name, dataset_name, variant_name)
    switch_to_experiment_db(exp_db)

    # Prepare results directory
    results_dir = lake_data_path("experiments", experiment_name, variant_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running variant '{variant_name}': queries={n_queries}, "
                f"iterations={n_iterations}, target_classes={target_classes}")

    # Clear ALL process-level caches to prevent cross-contamination between variants.
    # Without this, similarity matrix / embedding / LLM response caches from
    # a prior variant leak into the current one causing incorrect clustering.
    reset_usage_stats()
    clear_similarity_cache()
    clear_llm_response_cache()
    start_time = time.time()

    try:
        result_state = run_federated_tbox(
            dataset_name=dataset_name,
            table_store_name=f"{dataset_name}_tables_entries",
            query_store_name=f"{dataset_name}_train_queries",
            n_clusters=n_clusters,
            n_iterations=n_iterations,
            target_classes=target_classes,
            llm_purpose=llm_purpose,
            max_queries=n_queries,
            rag_type=rag_type,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Variant '{variant_name}' failed: {e}")
        restore_main_db()
        return {
            "variant_name": variant_name,
            "success": False,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
            "llm_stats": _collect_llm_stats(),
        }

    elapsed = time.time() - start_time
    result = _build_result(
        result_state, variant_name, elapsed,
        n_queries, n_iterations, n_clusters, target_classes,
        dataset_name, results_dir,
    )

    # Save variant result
    variant_result_path = results_dir / "variant_result.json"
    with open(variant_result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Restore main DB
    restore_main_db()
    return result


def _build_result(
    result_state, variant_name, elapsed,
    n_queries, n_iterations, n_clusters, target_classes,
    dataset_name, results_dir,
) -> Dict[str, Any]:
    """Extract results from a completed workflow state."""
    result: Dict[str, Any] = {
        "variant_name": variant_name,
        "success": result_state.export_success,
        "elapsed_seconds": round(elapsed, 2),
        "config": {
            "dataset": dataset_name,
            "n_queries": n_queries,
            "n_iterations": n_iterations,
            "n_clusters": n_clusters,
            "target_classes": target_classes,
        },
    }

    if result_state.export_success:
        review_history = getattr(result_state, "review_log", []) or []
        result["tbox_metrics"] = compute_tbox_metrics(review_history)

        result["statistics"] = {
            "n_clusters_actual": len(result_state.cluster_assignments or {}),
            "n_cqs": len(result_state.competency_questions or []),
            "n_classes": len((result_state.current_tbox or {}).get("classes", [])),
            "n_data_properties": len((result_state.current_tbox or {}).get("data_properties", [])),
            "n_iterations_run": len(result_state.synthesis_log or []),
        }

        result["paths"] = {
            "owl_path": result_state.owl_path,
            "report_path": result_state.report_path,
        }

        # Save review history for detailed analysis
        if review_history:
            rh_path = str(results_dir / "review_history.json")
            with open(rh_path, "w") as f:
                json.dump(review_history, f, indent=2, default=str)
            result["paths"]["review_history"] = rh_path
    else:
        result["error"] = result_state.export_error or "Unknown error"

    result["llm_stats"] = _collect_llm_stats()
    return result


def _collect_llm_stats() -> Dict[str, Any]:
    """Collect current LLM usage statistics."""
    from llm.statistics import get_usage_stats
    stats = get_usage_stats()
    return {
        "total_requests": stats.get("total_requests", 0),
        "total_input_tokens": stats.get("total_input_tokens", 0),
        "total_output_tokens": stats.get("total_output_tokens", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "by_model": stats.get("by_model", {}),
        "by_caller": stats.get("by_caller", {}),
    }


def _aggregate_stage_llm_stats(variant_result: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate per-stage LLM stats into a variant-level total."""
    total = {
        "total_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
    }
    for stage_key in ("stage2", "stage3"):
        stage = variant_result.get(stage_key, {})
        stage_llm = stage.get("llm_stats", {})
        total["total_requests"] += stage_llm.get("total_requests", 0)
        total["total_input_tokens"] += stage_llm.get("total_input_tokens", 0)
        total["total_output_tokens"] += stage_llm.get("total_output_tokens", 0)
        total["total_tokens"] += stage_llm.get("total_tokens", 0)
    return total


# ============================================================
# Experiment 1: Iteration Ablation (single run to max)
# ============================================================

def run_iteration_ablation(
    dataset_name: str,
    max_iterations: int = 10,
    n_queries: int = 200,
    n_clusters: int = 0,
    target_classes: int = 50,
    llm_purpose: str = "local",
    experiment_name: Optional[str] = None,
    rag_type: str = "vector",
) -> Dict[str, Any]:
    """
    Iteration Ablation: Run ONCE with max_iterations, extract per-iteration metrics.
    No need to repeat for each iteration count -- review_log captures everything.
    """
    if not experiment_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"iter_ablation_{dataset_name}_{ts}"

    logger.info("=" * 70)
    logger.info("Experiment: Iteration Ablation (single run)")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Max iterations: {max_iterations}")
    logger.info(f"  Queries: {n_queries}")
    logger.info(f"  Experiment: {experiment_name}")

    variant_name = f"iter{max_iterations}"
    variant_result = _run_single_variant(
        dataset_name=dataset_name,
        n_queries=n_queries,
        n_iterations=max_iterations,
        n_clusters=n_clusters,
        target_classes=target_classes,
        llm_purpose=llm_purpose,
        experiment_name=experiment_name,
        variant_name=variant_name,
        rag_type=rag_type,
    )

    results = {
        "experiment_type": "iteration_ablation",
        "experiment_name": experiment_name,
        "config": {
            "dataset": dataset_name,
            "max_iterations": max_iterations,
            "n_queries": n_queries,
            "n_clusters": n_clusters,
            "llm_purpose": llm_purpose,
        },
        "variant": variant_result,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
    }

    # Save results
    exp_dir = lake_data_path("experiments", experiment_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Print per-iteration table
    _print_iteration_table(variant_result)

    return results


# ============================================================
# Experiment 2: Query Count Ablation
# ============================================================

def run_query_ablation(
    dataset_name: str,
    query_counts: List[int],
    n_iterations: int = 5,
    n_clusters: int = 0,
    target_classes: int = 50,
    llm_purpose: str = "local",
    experiment_name: Optional[str] = None,
    rag_type: str = "vector",
) -> Dict[str, Any]:
    """Query ablation: vary the number of training queries."""
    if not experiment_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"query_ablation_{dataset_name}_{ts}"

    logger.info("=" * 70)
    logger.info("Experiment: Query Count Ablation")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Query counts: {query_counts}")
    logger.info(f"  Fixed iterations: {n_iterations}")
    logger.info(f"  Experiment: {experiment_name}")

    results = {
        "experiment_type": "query_ablation",
        "experiment_name": experiment_name,
        "config": {
            "dataset": dataset_name,
            "query_counts": query_counts,
            "n_iterations": n_iterations,
            "n_clusters": n_clusters,
            "llm_purpose": llm_purpose,
        },
        "variants": [],
        "start_time": datetime.now().isoformat(),
    }

    for n_queries in query_counts:
        variant_name = f"q{n_queries}"
        logger.info(f"\n--- Variant: {variant_name} ---")
        variant_result = _run_single_variant(
            dataset_name=dataset_name,
            n_queries=n_queries,
            n_iterations=n_iterations,
            n_clusters=n_clusters,
            target_classes=target_classes,
            llm_purpose=llm_purpose,
            experiment_name=experiment_name,
            variant_name=variant_name,
            rag_type=rag_type,
        )
        results["variants"].append(variant_result)
        _log_variant_status(variant_result)

    results["end_time"] = datetime.now().isoformat()
    _save_and_print(results, experiment_name)
    return results


# ============================================================
# Experiment 3: Concept Count Ablation
# ============================================================

def run_concept_ablation(
    dataset_name: str,
    target_classes_list: List[int],
    n_queries: int = 200,
    n_iterations: int = 5,
    n_clusters: int = 0,
    llm_purpose: str = "local",
    experiment_name: Optional[str] = None,
    rag_type: str = "vector",
) -> Dict[str, Any]:
    """Concept count ablation: vary target number of classes."""
    if not experiment_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"concept_ablation_{dataset_name}_{ts}"

    logger.info("=" * 70)
    logger.info("Experiment: Concept Count Ablation")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Target classes: {target_classes_list}")
    logger.info(f"  Fixed queries: {n_queries}, iterations: {n_iterations}")
    logger.info(f"  Experiment: {experiment_name}")

    results = {
        "experiment_type": "concept_ablation",
        "experiment_name": experiment_name,
        "config": {
            "dataset": dataset_name,
            "target_classes_list": target_classes_list,
            "n_queries": n_queries,
            "n_iterations": n_iterations,
            "n_clusters": n_clusters,
            "llm_purpose": llm_purpose,
        },
        "variants": [],
        "start_time": datetime.now().isoformat(),
    }

    for tc in target_classes_list:
        variant_name = f"tc{tc}"
        logger.info(f"\n--- Variant: {variant_name} ---")
        variant_result = _run_single_variant(
            dataset_name=dataset_name,
            n_queries=n_queries,
            n_iterations=n_iterations,
            n_clusters=n_clusters,
            target_classes=tc,
            llm_purpose=llm_purpose,
            experiment_name=experiment_name,
            variant_name=variant_name,
            rag_type=rag_type,
        )
        results["variants"].append(variant_result)
        _log_variant_status(variant_result)

    results["end_time"] = datetime.now().isoformat()
    _save_and_print(results, experiment_name)
    return results


# ============================================================
# Output Helpers
# ============================================================

def _log_variant_status(variant: Dict) -> None:
    status = "OK" if variant.get("success") else "FAIL"
    elapsed = variant.get("elapsed_seconds", 0)
    name = variant.get("variant_name", "?")
    llm = variant.get("llm_stats", {})
    logger.info(
        f"  [{status}] {name}: {elapsed:.1f}s | "
        f"LLM calls={llm.get('total_requests', 0)}, "
        f"prompt_tok={llm.get('total_input_tokens', 0)}, "
        f"output_tok={llm.get('total_output_tokens', 0)}"
    )


def _print_iteration_table(variant: Dict) -> None:
    """Print per-iteration metrics from a single iteration-ablation run."""
    tbox_metrics = variant.get("tbox_metrics", {})
    iters = tbox_metrics.get("per_iteration", [])
    if not iters:
        print("No per-iteration data available.")
        return

    print("\n" + "=" * 90)
    print("Per-Iteration Metrics")
    print("=" * 90)
    header = (
        f"{'Iter':<6} {'Classes':<10} {'Agents':<8} {'AvgCov':<10} "
        f"{'HighCov':<10} {'Delta':<10} {'Actions':<8}"
    )
    print(header)
    print("-" * 90)

    for m in iters:
        delta_str = f"{m['convergence_delta']:.4f}" if m["convergence_delta"] is not None else "-"
        print(
            f"{m['iteration']:<6} {m['n_classes']:<10} {m['n_agents']:<8} "
            f"{m['avg_coverage_ratio']:<10.4f} {m['high_coverage_class_ratio']:<10.4f} "
            f"{delta_str:<10} {m['n_actions']:<8}"
        )

    print("-" * 90)
    final = tbox_metrics.get("final_metrics", {})
    if final:
        print(f"Final: {final.get('final_n_classes', '?')} classes, "
              f"coverage={final.get('final_avg_coverage', 0):.4f}, "
              f"high_cov={final.get('final_high_coverage_ratio', 0):.4f}")

    # Print LLM statistics summary
    llm = variant.get("llm_stats", {})
    if llm:
        print()
        print("LLM Statistics")
        print("-" * 90)
        print(f"  Total requests: {llm.get('total_requests', 0)}")
        print(f"  Input tokens:   {llm.get('total_input_tokens', 0):,}")
        print(f"  Output tokens:  {llm.get('total_output_tokens', 0):,}")
        print(f"  Total tokens:   {llm.get('total_tokens', 0):,}")
        by_caller = llm.get("by_caller", {})
        if by_caller:
            print(f"  By caller:")
            for caller, cstats in sorted(by_caller.items(), key=lambda x: -x[1].get("requests", 0)):
                print(f"    {caller:<30} reqs={cstats.get('requests', 0):<6} "
                      f"tokens={cstats.get('total_tokens', 0):,}")
    print()


def _save_and_print(results: Dict, experiment_name: str) -> None:
    """Save experiment results and print comparison table."""
    exp_dir = lake_data_path("experiments", experiment_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")
    _print_comparison_table(results)


def _print_comparison_table(results: Dict) -> None:
    """Print comparison table across multi-variant experiments."""
    variants = results.get("variants", [])
    if not variants:
        return

    print("\n" + "=" * 115)
    print(f"Experiment: {results.get('experiment_name', '?')}")
    print(f"Type: {results.get('experiment_type', '?')}")
    print("=" * 115)
    header = (
        f"{'Variant':<12} {'OK':<4} {'Time(s)':<10} {'Classes':<8} "
        f"{'AvgCov':<10} {'HighCov':<10} {'LLM#':<8} "
        f"{'PromptTok':<12} {'OutputTok':<12}"
    )
    print(header)
    print("-" * 115)

    for v in variants:
        status = "Y" if v.get("success") else "N"
        stats = v.get("statistics", {})
        fm = v.get("tbox_metrics", {}).get("final_metrics", {})
        llm = v.get("llm_stats", {})
        print(
            f"{v.get('variant_name', '?'):<12} {status:<4} "
            f"{v.get('elapsed_seconds', 0):<10.1f} "
            f"{stats.get('n_classes', fm.get('final_n_classes', 0)):<8} "
            f"{fm.get('final_avg_coverage', 0):<10.4f} "
            f"{fm.get('final_high_coverage_ratio', 0):<10.4f} "
            f"{llm.get('total_requests', 0):<8} "
            f"{llm.get('total_input_tokens', 0):<12} "
            f"{llm.get('total_output_tokens', 0):<12}"
        )

    print("-" * 115)
    print()


# ============================================================
# Stage 2-4 Iteration Ablation (Cross-Iteration Comparison)
# ============================================================

def _get_ontology_lance_tables() -> List[str]:
    """Get the ontology lance table names (for symlink)."""
    return [
        "ontology_classes.lance",
        "ontology_properties.lance",
        "ontology_metadata.lance",
    ]


def setup_stage2_variant_db(
    experiment_name: str,
    dataset_name: str,
    variant_name: str,
    source_ontology_db: Path,
) -> Path:
    """
    Create an isolated LanceDB directory for a Stage 2 variant.
    
    Symlinks:
      - Shared tables (tables_entries, queries) from main DB
      - Ontology tables from source_ontology_db (the iter10/lancedb from Stage 1)
    
    Returns:
        Path to the variant LanceDB directory
    """
    main_db = get_db_path()
    exp_db = lake_data_path("experiments", experiment_name, variant_name, "lancedb")
    exp_db.mkdir(parents=True, exist_ok=True)
    
    # Symlink shared data tables from main DB
    for table_dir in _get_shared_lance_tables(dataset_name):
        src = main_db / table_dir
        dst = exp_db / table_dir
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
            logger.debug(f"  Symlinked (main): {table_dir}")
    
    # Symlink ontology tables from source_ontology_db
    for ont_table in _get_ontology_lance_tables():
        src = source_ontology_db / ont_table
        dst = exp_db / ont_table
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
            logger.debug(f"  Symlinked (ontology): {ont_table}")
    
    # Also symlink cache directory if exists
    cache_src = source_ontology_db / "cache"
    cache_dst = exp_db / "cache"
    if cache_src.exists() and not cache_dst.exists():
        os.symlink(cache_src, cache_dst)
        logger.debug(f"  Symlinked: cache")
    
    logger.info(f"Stage 2 variant DB ready: {exp_db}")
    return exp_db


def run_stage2_iteration_ablation(
    experiment_path: str,
    iterations: Optional[List[int]] = None,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
    skip_stage5: bool = False,
    skip_hyde: bool = True,  # Eval is separate
    max_tables: Optional[int] = None,
    batch_size: int = 1000,
    llm_purpose: str = "local",
    base_index_key: str = "td_cd_cs",
    hyde_parallel: int = 64,
    hyde_rag_top_k: int = 3,
    hyde_rag_type: str = "vector",
    disable_transform_reuse: bool = False,
) -> Dict[str, Any]:
    """
    Run Stage 2-5 for each iteration in an existing Iteration Ablation experiment.
    
    Each iteration uses a different TBox snapshot (iter_1, iter_2, ..., iter_N).
    
    Args:
        experiment_path: Path to the iteration-ablation results.json
        iterations: List of iterations to run (default: all except iter_0)
        skip_stage3: Skip Stage 3 (Layer 2 TBox)
        skip_stage4: Skip Stage 4 (Table Summarization)
        skip_stage5: Skip Stage 5 (Retrieval Index)
        skip_hyde: Skip HyDE Query Analysis (eval is separate)
        max_tables: Max tables to process per variant
        batch_size: Batch size for Stage 2
        llm_purpose: LLM purpose key
        base_index_key: Base index key (e.g. 'td_cd_cs')
        hyde_parallel: Parallel workers for HyDE query analysis
        hyde_rag_top_k: RAG top-k for HyDE analysis
        hyde_rag_type: RAG type for HyDE ('bm25', 'vector', 'hybrid')
    
    Returns:
        Results dict with per-iteration metrics
    """
    from cli.run_pipeline import run_column_summary, run_layer2_annotation, run_summarization, run_retrieval_index
    from llm.statistics import reset_usage_stats
    import subprocess
    
    # Load original experiment results
    if not os.path.exists(experiment_path):
        return {"error": f"Not found: {experiment_path}"}
    
    with open(experiment_path) as f:
        orig_results = json.load(f)
    
    if orig_results.get("experiment_type") != "iteration_ablation":
        return {"error": f"Expected iteration_ablation experiment, got: {orig_results.get('experiment_type')}"}
    
    experiment_name = orig_results.get("experiment_name", "")
    config = orig_results.get("config", {})
    dataset_name = config.get("dataset", "")
    max_iterations = config.get("max_iterations", 10)
    
    # Find the source ontology DB (iter10/lancedb)
    exp_base = Path(experiment_path).parent
    variant_name = f"iter{max_iterations}"
    source_ontology_db = exp_base / variant_name / "lancedb"
    
    if not source_ontology_db.exists():
        return {"error": f"Source ontology DB not found: {source_ontology_db}"}
    
    # Determine iterations to run
    if iterations is None:
        # Default: iter_1 to iter_N (skip iter_0 which is just seeded classes)
        iterations = list(range(1, max_iterations + 1))
    
    logger.info("=" * 70)
    logger.info("Stage 2-5 + HyDE Iteration Ablation")
    logger.info("=" * 70)
    logger.info(f"  Base Experiment: {experiment_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Source Ontology DB: {source_ontology_db}")
    logger.info(f"  Skip Stage 3: {skip_stage3}")
    logger.info(f"  Skip Stage 4: {skip_stage4}")
    logger.info(f"  Skip Stage 5: {skip_stage5}")
    logger.info(f"  Skip HyDE: {skip_hyde}")
    if not skip_stage5:
        logger.info(f"  Base Index Key: {base_index_key}")
    if not skip_hyde:
        logger.info(f"  HyDE Config: parallel={hyde_parallel}, rag_top_k={hyde_rag_top_k}, rag_type={hyde_rag_type}")
    
    results = {
        "experiment_type": "stage2_iteration_ablation",
        "experiment_name": experiment_name,
        "config": {
            "dataset": dataset_name,
            "iterations": iterations,
            "source_experiment": experiment_path,
            "skip_stage3": skip_stage3,
            "skip_stage4": skip_stage4,
            "skip_stage5": skip_stage5,
            "skip_hyde": skip_hyde,
            "base_index_key": base_index_key,
        },
        "variants": [],
    }
    
    for iter_n in iterations:
        variant_name = f"stage2_iter{iter_n}"
        logger.info("-" * 50)
        logger.info(f"Running variant: {variant_name} (tbox_iteration={iter_n})")
        logger.info("-" * 50)
        
        start_time = time.time()
        variant_result = {
            "variant_name": variant_name,
            "tbox_iteration": iter_n,
            "success": False,
        }
        
        try:
            # Setup isolated DB
            exp_db = setup_stage2_variant_db(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                variant_name=variant_name,
                source_ontology_db=source_ontology_db,
            )
            
            # Switch to experiment DB
            switch_to_experiment_db(exp_db)
            
            # Reset global LLM stats before starting this variant
            reset_usage_stats()
            
            # Stage 2: Column Summary
            logger.info(f"[Stage 2] Column Summary (tbox_iteration={iter_n})")
            s2_result = run_column_summary(
                dataset_name=dataset_name,
                max_tables=max_tables,
                batch_size=batch_size,
                tbox_iteration=iter_n,
                llm_purpose=llm_purpose,
                disable_transform_reuse=disable_transform_reuse,
            )
            s2_llm_stats = _collect_llm_stats()
            variant_result["stage2"] = {
                "success": s2_result.get("success", True),
                "tables_processed": s2_result.get("tables_processed", 0),
                "columns_analyzed": s2_result.get("columns_analyzed", 0),
                "total_llm_calls": s2_result.get("total_llm_calls", 0),
                "code_reuse_count": s2_result.get("code_reuse_count", 0),
                "elapsed": s2_result.get("elapsed", 0),
                "llm_stats": s2_llm_stats,
                "llm_stats_timeline": s2_result.get("llm_stats_timeline", []),
            }
            
            # Stage 3: Layer 2 TBox (optional)
            if not skip_stage3:
                reset_usage_stats()
                logger.info(f"[Stage 3] Layer 2 TBox")
                s3_result = run_layer2_annotation(
                    dataset_name=dataset_name,
                    max_tables=max_tables,
                    batch_size=batch_size,
                )
                s3_llm_stats = _collect_llm_stats()
                variant_result["stage3"] = {
                    "success": s3_result.get("success", False),
                    "num_column_classes": s3_result.get("num_column_classes", 0),
                    "num_table_classes": s3_result.get("num_table_classes", 0),
                    "elapsed": s3_result.get("elapsed", 0),
                    "llm_stats": s3_llm_stats,
                }
            
            # Stage 4: Table Summarization (optional)
            if not skip_stage4:
                logger.info(f"[Stage 4] Table Summarization")
                s4_result = run_summarization(
                    dataset_name=dataset_name,
                    output_format="both",
                )
                variant_result["stage4"] = {
                    "success": s4_result.get("success", False),
                    "total_tables": s4_result.get("total_tables", 0),
                    "elapsed": s4_result.get("elapsed", 0),
                }
            
            # Stage 5: Retrieval Index (optional)
            if not skip_stage5:
                # Output to variant directory: {variant_path}/indexes/{index_key}/
                variant_path = exp_db.parent  # exp_db is .../stage2_iter{n}/lancedb
                logger.info(f"[Stage 5] Retrieval Index (output to {variant_path}/indexes/)")
                s5_result = run_retrieval_index(
                    dataset_name=dataset_name,
                    index_key=base_index_key,  # Use base key, not suffixed
                    enable_faiss=True,
                    enable_bm25=True,
                    output_base_path=variant_path,
                )
                variant_result["stage5"] = {
                    "success": s5_result.get("success", False),
                    "total_tables": s5_result.get("total_tables", 0),
                    "index_key": base_index_key,
                    "faiss_path": s5_result.get("faiss_path"),
                    "bm25_path": s5_result.get("bm25_path"),
                    "elapsed": s5_result.get("elapsed", 0),
                }
            
            # HyDE Query Analysis (optional)
            if not skip_hyde:
                # Output to variant directory: {variant_path}/eval_results/
                variant_path = exp_db.parent  # exp_db is .../stage2_iter{n}/lancedb
                hyde_output_dir = variant_path / "eval_results"
                hyde_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[HyDE] Query Analysis (output to {hyde_output_dir})")
                hyde_start = time.time()
                
                hyde_cmd = [
                    "python", "-m", "cli.retrieval",
                    "--dataset", dataset_name,
                    "--analyze-queries",
                    "--use-rag",
                    "-n", "-1",
                    "--index-key", base_index_key,  # Use base key, indexes in variant dir
                    "--rag-top-k", str(hyde_rag_top_k),
                    "--rag-type", hyde_rag_type,
                    "--parallel", str(hyde_parallel),
                    "--llm", llm_purpose,
                    "--output-dir", str(hyde_output_dir),
                    "--index-base-path", str(variant_path),  # Use variant-specific indexes
                ]
                
                # Run subprocess
                import sys
                source_dir = Path(__file__).parent.parent
                try:
                    # Stream output directly to console for real-time logging
                    proc = subprocess.run(
                        hyde_cmd,
                        cwd=str(source_dir),
                    )
                    hyde_elapsed = time.time() - hyde_start
                    hyde_success = proc.returncode == 0
                    if not hyde_success:
                        logger.warning(f"  HyDE failed with return code {proc.returncode}")
                    variant_result["hyde"] = {
                        "success": hyde_success,
                        "output_dir": str(hyde_output_dir),
                        "elapsed": hyde_elapsed,
                        "error": f"Exit code {proc.returncode}" if not hyde_success else None,
                    }
                except Exception as hyde_e:
                    variant_result["hyde"] = {
                        "success": False,
                        "error": str(hyde_e),
                    }
            
            variant_result["success"] = True
            
        except Exception as e:
            logger.error(f"Variant {variant_name} failed: {e}")
            variant_result["error"] = str(e)
        finally:
            restore_main_db()
        
        elapsed = time.time() - start_time
        variant_result["elapsed_seconds"] = round(elapsed, 2)
        # Aggregate per-stage LLM stats into variant-level total
        variant_result["llm_stats"] = _aggregate_stage_llm_stats(variant_result)
        results["variants"].append(variant_result)
        
        logger.info(f"✓ Variant {variant_name} completed in {elapsed:.1f}s")
    
    # Save results
    results_path = exp_base / "results_stage2_full.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nStage 2-5 + HyDE results saved to: {results_path}")
    
    # Print summary table
    _print_stage2_comparison_table(results)
    
    return results


def _print_stage2_comparison_table(results: Dict) -> None:
    """Print Stage 2-4 comparison across iterations."""
    variants = results.get("variants", [])
    if not variants:
        return
    
    print("\n" + "=" * 100)
    print(f"Stage 2-4 Iteration Ablation Results")
    print(f"Experiment: {results.get('experiment_name', '?')}")
    print("=" * 100)
    
    header = (
        f"{'Variant':<15} {'Iter':<5} {'OK':<4} {'Time(s)':<10} "
        f"{'Tables':<8} {'Columns':<10} {'LLM#':<8} {'Reuse':<8} {'S2 Tok':<12}"
    )
    print(header)
    print("-" * 115)
    
    for v in variants:
        status = "Y" if v.get("success") else "N"
        s2 = v.get("stage2", {})
        llm = v.get("llm_stats", {})
        s2_llm = s2.get("llm_stats", {})
        print(
            f"{v.get('variant_name', '?'):<15} "
            f"{v.get('tbox_iteration', 0):<5} "
            f"{status:<4} "
            f"{v.get('elapsed_seconds', 0):<10.1f} "
            f"{s2.get('tables_processed', 0):<8} "
            f"{s2.get('columns_analyzed', 0):<10} "
            f"{llm.get('total_requests', 0):<8} "
            f"{s2.get('code_reuse_count', 0):<8} "
            f"{s2_llm.get('total_tokens', 0):<12}"
        )
    
    print("-" * 115)
    print()


def run_stage2_general_ablation(
    experiment_path: str,
    variants: Optional[List[str]] = None,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
    skip_stage5: bool = False,
    skip_hyde: bool = True,  # Eval is separate
    max_tables: Optional[int] = None,
    batch_size: int = 1000,
    llm_purpose: str = "local",
    base_index_key: str = "td_cd_cs",
    hyde_parallel: int = 64,
    hyde_rag_top_k: int = 3,
    hyde_rag_type: str = "vector",
    disable_transform_reuse: bool = False,
) -> Dict[str, Any]:
    """
    Run Stage 2-5 for general ablation experiments (query_ablation, concept_ablation).
    
    Each variant has its own lancedb with ontology. Use the latest iteration (-1).
    
    Args:
        experiment_path: Path to the ablation results.json
        variants: List of variant names to run (e.g., ['q50', 'q100']). None = all.
        Other args: Same as run_stage2_iteration_ablation
    """
    from cli.run_pipeline import run_column_summary, run_layer2_annotation, run_summarization, run_retrieval_index
    from llm.statistics import reset_usage_stats
    import subprocess
    
    if not os.path.exists(experiment_path):
        return {"error": f"Not found: {experiment_path}"}
    
    with open(experiment_path) as f:
        orig_results = json.load(f)
    
    experiment_type = orig_results.get("experiment_type", "")
    if experiment_type not in ("query_ablation", "concept_ablation"):
        return {"error": f"Expected query_ablation or concept_ablation, got: {experiment_type}"}
    
    experiment_name = orig_results.get("experiment_name", "")
    config = orig_results.get("config", {})
    dataset_name = config.get("dataset", "")
    
    exp_base = Path(experiment_path).parent
    
    # Get available variants from results
    orig_variants = orig_results.get("variants", [])
    available_variants = [v.get("variant_name") for v in orig_variants]
    
    if variants is None:
        variants = available_variants
    
    logger.info("=" * 70)
    logger.info(f"Stage 2-5 + HyDE: {experiment_type}")
    logger.info("=" * 70)
    logger.info(f"  Base Experiment: {experiment_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Variants: {variants}")
    logger.info(f"  Skip Stage 3: {skip_stage3}")
    logger.info(f"  Skip Stage 4: {skip_stage4}")
    logger.info(f"  Skip Stage 5: {skip_stage5}")
    logger.info(f"  Skip HyDE: {skip_hyde}")
    
    results = {
        "experiment_type": f"stage2_{experiment_type}",
        "experiment_name": experiment_name,
        "config": {
            "dataset": dataset_name,
            "variants": variants,
            "source_experiment": experiment_path,
            "skip_stage3": skip_stage3,
            "skip_stage4": skip_stage4,
            "skip_stage5": skip_stage5,
            "skip_hyde": skip_hyde,
            "base_index_key": base_index_key,
        },
        "variants": [],
    }
    
    for var_name in variants:
        stage2_variant_name = f"stage2_{var_name}"
        source_ontology_db = exp_base / var_name / "lancedb"
        
        if not source_ontology_db.exists():
            logger.warning(f"  Skipping {var_name}: lancedb not found")
            continue
        
        logger.info("-" * 50)
        logger.info(f"Running variant: {stage2_variant_name}")
        logger.info("-" * 50)
        
        start_time = time.time()
        variant_result = {
            "variant_name": stage2_variant_name,
            "source_variant": var_name,
            "tbox_iteration": -1,  # Use latest
            "success": False,
        }
        
        try:
            # Setup isolated DB (use source variant's ontology)
            exp_db = setup_stage2_variant_db(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                variant_name=stage2_variant_name,
                source_ontology_db=source_ontology_db,
            )
            
            switch_to_experiment_db(exp_db)
            
            # Reset global LLM stats before starting this variant
            reset_usage_stats()
            
            # Stage 2: Column Summary (use latest iteration: -1)
            logger.info(f"[Stage 2] Column Summary (tbox_iteration=-1)")
            s2_result = run_column_summary(
                dataset_name=dataset_name,
                max_tables=max_tables,
                batch_size=batch_size,
                tbox_iteration=-1,  # Latest iteration
                llm_purpose=llm_purpose,
                disable_transform_reuse=disable_transform_reuse,
            )
            s2_llm_stats = _collect_llm_stats()
            variant_result["stage2"] = {
                "success": s2_result.get("success", True),
                "tables_processed": s2_result.get("tables_processed", 0),
                "columns_analyzed": s2_result.get("columns_analyzed", 0),
                "total_llm_calls": s2_result.get("total_llm_calls", 0),
                "code_reuse_count": s2_result.get("code_reuse_count", 0),
                "elapsed": s2_result.get("elapsed", 0),
                "llm_stats": s2_llm_stats,
                "llm_stats_timeline": s2_result.get("llm_stats_timeline", []),
            }
            
            # Stage 3: Layer 2 TBox (optional)
            if not skip_stage3:
                reset_usage_stats()
                logger.info(f"[Stage 3] Layer 2 TBox")
                s3_result = run_layer2_annotation(
                    dataset_name=dataset_name,
                    max_tables=max_tables,
                    batch_size=batch_size,
                )
                s3_llm_stats = _collect_llm_stats()
                variant_result["stage3"] = {
                    "success": s3_result.get("success", False),
                    "num_column_classes": s3_result.get("num_column_classes", 0),
                    "num_table_classes": s3_result.get("num_table_classes", 0),
                    "elapsed": s3_result.get("elapsed", 0),
                    "llm_stats": s3_llm_stats,
                }
            
            # Stage 4: Table Summarization (optional)
            if not skip_stage4:
                logger.info(f"[Stage 4] Table Summarization")
                s4_result = run_summarization(
                    dataset_name=dataset_name,
                    output_format="both",
                )
                variant_result["stage4"] = {
                    "success": s4_result.get("success", False),
                    "total_tables": s4_result.get("total_tables", 0),
                    "elapsed": s4_result.get("elapsed", 0),
                }
            
            # Stage 5: Retrieval Index (optional)
            if not skip_stage5:
                variant_path = exp_db.parent
                logger.info(f"[Stage 5] Retrieval Index (output to {variant_path}/indexes/)")
                s5_result = run_retrieval_index(
                    dataset_name=dataset_name,
                    index_key=base_index_key,
                    enable_faiss=True,
                    enable_bm25=True,
                    output_base_path=variant_path,
                )
                variant_result["stage5"] = {
                    "success": s5_result.get("success", False),
                    "total_tables": s5_result.get("total_tables", 0),
                    "index_key": base_index_key,
                    "faiss_path": s5_result.get("faiss_path"),
                    "bm25_path": s5_result.get("bm25_path"),
                    "elapsed": s5_result.get("elapsed", 0),
                }
            
            # HyDE Query Analysis (optional)
            if not skip_hyde:
                variant_path = exp_db.parent
                hyde_output_dir = variant_path / "eval_results"
                hyde_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[HyDE] Query Analysis (output to {hyde_output_dir})")
                hyde_start = time.time()
                
                hyde_cmd = [
                    "python", "-m", "cli.retrieval",
                    "--dataset", dataset_name,
                    "--analyze-queries",
                    "--use-rag",
                    "-n", "-1",
                    "--index-key", base_index_key,
                    "--rag-top-k", str(hyde_rag_top_k),
                    "--rag-type", hyde_rag_type,
                    "--parallel", str(hyde_parallel),
                    "--llm", llm_purpose,
                    "--output-dir", str(hyde_output_dir),
                    "--index-base-path", str(variant_path),
                ]
                
                source_dir = Path(__file__).parent.parent
                try:
                    # Stream output directly to console for real-time logging
                    proc = subprocess.run(
                        hyde_cmd,
                        cwd=str(source_dir),
                    )
                    hyde_elapsed = time.time() - hyde_start
                    hyde_success = proc.returncode == 0
                    if not hyde_success:
                        logger.warning(f"  HyDE failed with return code {proc.returncode}")
                    variant_result["hyde"] = {
                        "success": hyde_success,
                        "output_dir": str(hyde_output_dir),
                        "elapsed": hyde_elapsed,
                        "error": f"Exit code {proc.returncode}" if not hyde_success else None,
                    }
                except Exception as hyde_e:
                    variant_result["hyde"] = {
                        "success": False,
                        "error": str(hyde_e),
                    }
            
            variant_result["success"] = True
            
        except Exception as e:
            logger.error(f"Variant {stage2_variant_name} failed: {e}")
            variant_result["error"] = str(e)
        finally:
            restore_main_db()
        
        elapsed = time.time() - start_time
        variant_result["elapsed_seconds"] = round(elapsed, 2)
        # Aggregate per-stage LLM stats into variant-level total
        variant_result["llm_stats"] = _aggregate_stage_llm_stats(variant_result)
        results["variants"].append(variant_result)
        
        logger.info(f"✓ Variant {stage2_variant_name} completed in {elapsed:.1f}s")
    
    # Save results
    results_path = exp_base / "results_stage2_full.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nStage 2-5 + HyDE results saved to: {results_path}")
    
    _print_stage2_comparison_table(results)
    
    return results


# ============================================================
# Analysis & Listing
# ============================================================

def analyze_experiment(experiment_path: str) -> Dict[str, Any]:
    """Analyze an existing experiment's results.json."""
    if not os.path.exists(experiment_path):
        return {"error": f"Not found: {experiment_path}"}

    with open(experiment_path) as f:
        results = json.load(f)

    exp_type = results.get("experiment_type", "")

    if exp_type == "iteration_ablation":
        variant = results.get("variant", {})
        _print_iteration_table(variant)
        return results

    # Multi-variant experiments
    _print_comparison_table(results)
    return results


# ============================================================
# LLM Model Comparison Ablation
# ============================================================

EXPERIMENT_BASE = "llm_model_comparison"

def _get_current_model_names() -> Dict[str, str]:
    """Read current model_name values from llm_models.json."""
    config_path = Path(__file__).parent.parent / "config" / "llm_models.json"
    with open(config_path) as f:
        config = json.load(f)
    models = config.get("models", {})
    mapping = config.get("purpose_mapping", {})
    result = {}
    for purpose, model_key in mapping.items():
        model_cfg = models.get(model_key, {})
        result[purpose] = model_cfg.get("model_name", "unknown")
    return result


def _setup_llm_ablation_stage1_db(
    variant_name: str,
    datasets: List[str],
) -> Path:
    """
    Create isolated LanceDB for a Stage 1 variant.
    Symlinks shared data (tables_entries, queries) for all datasets.
    """
    main_db = get_db_path()
    exp_db = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "lancedb")
    exp_db.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        for table_dir in _get_shared_lance_tables(ds):
            src = main_db / table_dir
            dst = exp_db / table_dir
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

    logger.info(f"Stage 1 variant DB ready: {exp_db}")
    return exp_db


def link_existing_stage1(
    variant_name: str,
    datasets: List[str],
) -> Dict[str, Any]:
    """
    Link existing ontology tables from main DB to a Stage 1 variant directory.
    Used when Stage 1 has already been run with the current model.
    """
    main_db = get_db_path()
    exp_db = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "lancedb")
    exp_db.mkdir(parents=True, exist_ok=True)

    # Symlink shared data tables
    for ds in datasets:
        for table_dir in _get_shared_lance_tables(ds):
            src = main_db / table_dir
            dst = exp_db / table_dir
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

    # Copy (not symlink) ontology tables so they are independent snapshots
    import shutil
    for ont_table in _get_ontology_lance_tables():
        src = main_db / ont_table
        dst = exp_db / ont_table
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            logger.info(f"  Copied: {ont_table}")

    # Also copy cache if exists
    cache_src = main_db / "cache"
    cache_dst = exp_db / "cache"
    if cache_src.exists() and not cache_dst.exists():
        os.symlink(cache_src, cache_dst)

    # Record metadata
    model_names = _get_current_model_names()
    meta = {
        "variant_name": variant_name,
        "phase": "stage1",
        "source": "linked_from_main_db",
        "gemini_model_name": model_names.get("gemini", "unknown"),
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
    }
    meta_path = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "result.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✓ Linked existing Stage 1 ontology to {variant_name}")
    logger.info(f"  Gemini model: {model_names.get('gemini')}")
    return meta


def run_llm_ablation_stage1(
    variant_name: str,
    datasets: List[str],
    n_iterations: int = 2,
    total_queries: int = 200,
    target_classes: int = 50,
    n_clusters: int = 0,
    rag_type: str = "hybrid",
) -> Dict[str, Any]:
    """
    Run Stage 1 (Federated TBox) for all datasets in an isolated experiment DB.
    The gemini_model.model_name in llm_models.json must be set BEFORE calling this.
    """
    from workflows.conceptualization import run_federated_tbox
    from llm.statistics import get_usage_stats, reset_usage_stats
    from llm.invoke_with_stats import clear_llm_response_cache
    from workflows.retrieval.unified_similarity import clear_similarity_cache

    model_names = _get_current_model_names()
    gemini_model = model_names.get("gemini", "unknown")

    logger.info("=" * 70)
    logger.info("LLM Model Ablation - Stage 1 (Federated TBox)")
    logger.info("=" * 70)
    logger.info(f"  Variant: {variant_name}")
    logger.info(f"  Gemini Model: {gemini_model}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Iterations: {n_iterations}")
    logger.info(f"  Total Queries: {total_queries}")

    exp_db = _setup_llm_ablation_stage1_db(variant_name, datasets)
    switch_to_experiment_db(exp_db)

    results = {
        "variant_name": variant_name,
        "phase": "stage1",
        "gemini_model_name": gemini_model,
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
        "per_dataset": {},
    }

    for ds in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Stage 1: {ds}")
        logger.info(f"{'='*50}")

        reset_usage_stats()
        clear_llm_response_cache()
        clear_similarity_cache()
        start_time = time.time()

        try:
            result_state = run_federated_tbox(
                dataset_name=ds,
                table_store_name=f"{ds}_tables_entries",
                query_store_name=f"{ds}_train_queries",
                n_clusters=n_clusters,
                n_iterations=n_iterations,
                target_classes=target_classes,
                llm_purpose="gemini",
                max_queries=total_queries,
                rag_type=rag_type,
            )
            elapsed = time.time() - start_time
            llm_stats = _collect_llm_stats()
            final_tbox = result_state.current_tbox or {}

            results["per_dataset"][ds] = {
                "success": result_state.export_success,
                "elapsed_seconds": round(elapsed, 2),
                "n_classes": len(final_tbox.get("classes", [])),
                "n_data_properties": len(final_tbox.get("data_properties", [])),
                "llm_stats": llm_stats,
            }
            logger.info(f"  ✓ {ds} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            results["per_dataset"][ds] = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            }
            logger.error(f"  ✗ {ds} failed: {e}")

    restore_main_db()

    # Save results
    result_path = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "result.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nStage 1 results saved to: {result_path}")

    return results


def run_llm_ablation_stage2(
    variant_name: str,
    stage1_variant: str,
    datasets: List[str],
    max_tables: Optional[int] = None,
    batch_size: int = 1000,
    budget_cap: int = 1000,
) -> Dict[str, Any]:
    """
    Run Stage 2-5 for all datasets using TBox from a Stage 1 variant.
    The local_vllm_model.model_name in llm_models.json must be set BEFORE calling this.
    """
    from cli.run_pipeline import (
        run_column_summary, run_layer2_annotation,
        run_summarization, run_retrieval_index,
    )
    from llm.statistics import reset_usage_stats

    model_names = _get_current_model_names()
    local_model = model_names.get("local", "unknown")
    main_db = get_db_path()

    # Locate Stage 1 ontology DB
    stage1_db = lake_data_path("experiments", EXPERIMENT_BASE, stage1_variant, "lancedb")
    if not stage1_db.exists():
        return {"error": f"Stage 1 DB not found: {stage1_db}"}

    # Load Stage 1 metadata for gemini model name
    stage1_meta_path = lake_data_path("experiments", EXPERIMENT_BASE, stage1_variant, "result.json")
    gemini_model = "unknown"
    if stage1_meta_path.exists():
        with open(stage1_meta_path) as f:
            stage1_meta = json.load(f)
        gemini_model = stage1_meta.get("gemini_model_name", "unknown")

    logger.info("=" * 70)
    logger.info("LLM Model Ablation - Stage 2-5")
    logger.info("=" * 70)
    logger.info(f"  Variant: {variant_name}")
    logger.info(f"  Stage 1 Source: {stage1_variant} (gemini={gemini_model})")
    logger.info(f"  Local Model: {local_model}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Budget Cap: {budget_cap}")

    stats_dir = str(lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "stats"))
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "variant_name": variant_name,
        "phase": "stage2",
        "stage1_variant": stage1_variant,
        "gemini_model_name": gemini_model,
        "local_model_name": local_model,
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
        "per_dataset": {},
    }

    for ds in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Stage 2-5: {ds}")
        logger.info(f"{'='*50}")

        # Setup isolated DB with ontology from stage1
        exp_db = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "lancedb")
        exp_db.mkdir(parents=True, exist_ok=True)

        # Symlink shared data tables
        for table_dir in _get_shared_lance_tables(ds):
            src = main_db / table_dir
            dst = exp_db / table_dir
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

        # Symlink ontology tables from stage1 variant
        for ont_table in _get_ontology_lance_tables():
            src = stage1_db / ont_table
            dst = exp_db / ont_table
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

        # Symlink cache
        cache_src = stage1_db / "cache"
        cache_dst = exp_db / "cache"
        if cache_src.exists() and not cache_dst.exists():
            os.symlink(cache_src, cache_dst)

        switch_to_experiment_db(exp_db)
        reset_usage_stats()
        ds_result: Dict[str, Any] = {"success": False}
        start_time = time.time()

        try:
            # Stage 2: Column Summary
            logger.info(f"[Stage 2] Column Summary")
            s2 = run_column_summary(
                dataset_name=ds,
                max_tables=max_tables,
                batch_size=batch_size,
                budget_cap=budget_cap,
                llm_purpose="local",
                fresh_start=True,
                stats_dir=stats_dir,
            )
            s2_stats = _collect_llm_stats()
            ds_result["stage2"] = {
                "success": s2.get("success", False),
                "tables_processed": s2.get("tables_processed", 0),
                "columns_analyzed": s2.get("columns_analyzed", 0),
                "elapsed": s2.get("elapsed", 0),
                "llm_stats": s2_stats,
            }

            # Stage 3: Layer 2 Annotation
            reset_usage_stats()
            logger.info(f"[Stage 3] Layer 2 Annotation")
            s3 = run_layer2_annotation(
                dataset_name=ds,
                max_tables=max_tables,
                batch_size=batch_size,
            )
            s3_stats = _collect_llm_stats()
            ds_result["stage3"] = {
                "success": s3.get("success", False),
                "elapsed": s3.get("elapsed", 0),
                "llm_stats": s3_stats,
            }

            # Stage 4: Summarization
            logger.info(f"[Stage 4] Summarization")
            s4 = run_summarization(dataset_name=ds, output_format="both")
            ds_result["stage4"] = {
                "success": s4.get("success", False),
                "elapsed": s4.get("elapsed", 0),
            }

            # Stage 5: Retrieval Index
            variant_path = exp_db.parent
            logger.info(f"[Stage 5] Retrieval Index")
            s5 = run_retrieval_index(
                dataset_name=ds,
                index_key="td_cd_cs",
                enable_faiss=True,
                enable_bm25=True,
                output_base_path=variant_path,
            )
            ds_result["stage5"] = {
                "success": s5.get("success", False),
                "elapsed": s5.get("elapsed", 0),
            }

            ds_result["success"] = True

        except Exception as e:
            logger.error(f"  ✗ {ds} failed: {e}")
            ds_result["error"] = str(e)
        finally:
            restore_main_db()

        elapsed = time.time() - start_time
        ds_result["elapsed_seconds"] = round(elapsed, 2)
        results["per_dataset"][ds] = ds_result
        logger.info(f"  {'✓' if ds_result['success'] else '✗'} {ds} completed in {elapsed:.1f}s")

    # Save results
    result_path = lake_data_path("experiments", EXPERIMENT_BASE, variant_name, "result.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nStage 2-5 results saved to: {result_path}")

    return results


def list_experiments(base_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """List all experiments."""
    if not base_dir:
        base_dir = str(lake_data_path("experiments"))

    experiments = []
    if os.path.exists(base_dir):
        for name in sorted(os.listdir(base_dir)):
            exp_dir = os.path.join(base_dir, name)
            if not os.path.isdir(exp_dir):
                continue
            results_path = os.path.join(exp_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    variants = data.get("variants", [])
                    n_v = len(variants) if variants else (1 if "variant" in data else 0)
                    experiments.append({
                        "name": name,
                        "type": data.get("experiment_type", "?"),
                        "n_variants": n_v,
                        "path": results_path,
                    })
                except Exception:
                    pass

    return experiments


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TBox Ablation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- iteration-ablation ---
    p_iter = sub.add_parser("iteration-ablation", help="Single run to max iterations")
    p_iter.add_argument("--dataset", "-d", required=True, help="Dataset name")
    p_iter.add_argument("--max-iterations", "-i", type=int, default=10, help="Max iterations (default: 10)")
    p_iter.add_argument("--queries", "-q", type=int, default=200, help="Fixed query count (default: 200)")
    p_iter.add_argument("--clusters", "-c", type=int, default=0, help="Clusters (0=auto)")
    p_iter.add_argument("--target-classes", type=int, default=50, help="Target classes (default: 50)")
    p_iter.add_argument("--llm-purpose", default="gemini", help="LLM purpose key (Stage 1 uses gemini)")
    p_iter.add_argument("--rag-type", default="vector", help="RAG type for hard negative sampling (default: vector)")
    p_iter.add_argument("--name", default=None, help="Experiment name")

    # --- query-ablation ---
    p_query = sub.add_parser("query-ablation", help="Vary training query count")
    p_query.add_argument("--dataset", "-d", required=True, help="Dataset name")
    p_query.add_argument("--queries", "-q", type=int, nargs="+", default=[50, 100, 200, 400], help="Query counts")
    p_query.add_argument("--iterations", "-i", type=int, default=5, help="Fixed iterations (default: 5)")
    p_query.add_argument("--clusters", "-c", type=int, default=0, help="Clusters (0=auto)")
    p_query.add_argument("--target-classes", type=int, default=50, help="Target classes (default: 50)")
    p_query.add_argument("--llm-purpose", default="gemini", help="LLM purpose key (Stage 1 uses gemini)")
    p_query.add_argument("--rag-type", default="vector", help="RAG type for hard negative sampling (default: vector)")
    p_query.add_argument("--name", default=None, help="Experiment name")

    # --- concept-ablation ---
    p_concept = sub.add_parser("concept-ablation", help="Vary target class count")
    p_concept.add_argument("--dataset", "-d", required=True, help="Dataset name")
    p_concept.add_argument("--targets", "-t", type=int, nargs="+", default=[10, 25, 50, 75, 100], help="Target class counts")
    p_concept.add_argument("--queries", "-q", type=int, default=200, help="Fixed queries (default: 200)")
    p_concept.add_argument("--iterations", "-i", type=int, default=5, help="Fixed iterations (default: 5)")
    p_concept.add_argument("--clusters", "-c", type=int, default=0, help="Clusters (0=auto)")
    p_concept.add_argument("--llm-purpose", default="gemini", help="LLM purpose key (Stage 1 uses gemini)")
    p_concept.add_argument("--rag-type", default="vector", help="RAG type for hard negative sampling (default: vector)")
    p_concept.add_argument("--name", default=None, help="Experiment name")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Analyze experiment results")
    p_analyze.add_argument("--path", "-p", help="Path to results.json")
    p_analyze.add_argument("--name", "-n", help="Experiment name")

    # --- stage2-iteration-ablation ---
    p_s2iter = sub.add_parser("stage2-iteration-ablation", help="Run Stage 2-5 for each iteration of an existing iteration-ablation experiment")
    p_s2iter.add_argument("--path", "-p", required=True, help="Path to iteration-ablation results.json")
    p_s2iter.add_argument("--iterations", "-i", type=int, nargs="+", default=None, help="Specific iterations to run (default: all 1..N)")
    p_s2iter.add_argument("--max-tables", "-m", type=int, default=None, help="Max tables per variant")
    p_s2iter.add_argument("--batch-size", "-b", type=int, default=1000, help="Batch size (default: 1000)")
    p_s2iter.add_argument("--llm-purpose", default="local", help="LLM purpose key")
    p_s2iter.add_argument("--disable-reuse", action="store_true", help="Disable transform reuse (ablation)")

    # --- stage2-general-ablation (for query/concept ablation) ---
    p_s2gen = sub.add_parser("stage2-general-ablation", help="Run Stage 2-5 for query/concept ablation experiments")
    p_s2gen.add_argument("--path", "-p", required=True, help="Path to query/concept ablation results.json")
    p_s2gen.add_argument("--variants", "-v", nargs="+", default=None, help="Specific variants to run (e.g., q50 q100). Default: all")
    p_s2gen.add_argument("--max-tables", "-m", type=int, default=None, help="Max tables per variant")
    p_s2gen.add_argument("--batch-size", "-b", type=int, default=1000, help="Batch size (default: 1000)")
    p_s2gen.add_argument("--llm-purpose", default="local", help="LLM purpose key")
    p_s2gen.add_argument("--disable-reuse", action="store_true", help="Disable transform reuse (ablation)")

    # --- list ---
    sub.add_parser("list", help="List all experiments")

    # --- llm-model-ablation ---
    p_llm = sub.add_parser("llm-model-ablation", help="LLM model comparison ablation")
    p_llm.add_argument("--phase", required=True, choices=["stage1", "stage2", "link-stage1"],
                        help="Phase: stage1 (run TBox), stage2 (run ABox+eval), link-stage1 (link existing TBox from main DB)")
    p_llm.add_argument("--variant-name", required=True,
                        help="Variant name (e.g. stage1_gemini3flash, combo_A)")
    p_llm.add_argument("--stage1-variant", default=None,
                        help="Stage 1 variant to reuse TBox from (required for --phase stage2)")
    p_llm.add_argument("--datasets", required=True,
                        help="Space-separated dataset names (e.g. 'adventure_works bird chembl')")
    p_llm.add_argument("--max-tables", type=int, default=None, help="Max tables per dataset (Stage 2)")
    p_llm.add_argument("--batch-size", type=int, default=1000, help="Batch size (default: 1000)")
    p_llm.add_argument("--budget-cap", type=int, default=1000, help="SH budget cap (default: 1000)")
    p_llm.add_argument("--n-iterations", type=int, default=2, help="Stage 1 iterations (default: 2)")
    p_llm.add_argument("--total-queries", type=int, default=200, help="Stage 1 queries (default: 200)")
    p_llm.add_argument("--target-classes", type=int, default=50, help="Stage 1 target classes (default: 50)")
    p_llm.add_argument("--rag-type", default="hybrid", help="Stage 1 RAG type (default: hybrid)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    # Configure loguru
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{time:HH:mm:ss}</level> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    if args.command == "iteration-ablation":
        run_iteration_ablation(
            dataset_name=args.dataset,
            max_iterations=args.max_iterations,
            n_queries=args.queries,
            n_clusters=args.clusters,
            target_classes=args.target_classes,
            llm_purpose=args.llm_purpose,
            experiment_name=args.name,
            rag_type=args.rag_type,
        )

    elif args.command == "query-ablation":
        run_query_ablation(
            dataset_name=args.dataset,
            query_counts=args.queries,
            n_iterations=args.iterations,
            n_clusters=args.clusters,
            target_classes=args.target_classes,
            llm_purpose=args.llm_purpose,
            experiment_name=args.name,
            rag_type=args.rag_type,
        )

    elif args.command == "concept-ablation":
        run_concept_ablation(
            dataset_name=args.dataset,
            target_classes_list=args.targets,
            n_queries=args.queries,
            n_iterations=args.iterations,
            n_clusters=args.clusters,
            llm_purpose=args.llm_purpose,
            experiment_name=args.name,
            rag_type=args.rag_type,
        )

    elif args.command == "stage2-iteration-ablation":
        run_stage2_iteration_ablation(
            experiment_path=args.path,
            iterations=args.iterations,
            max_tables=args.max_tables,
            batch_size=args.batch_size,
            llm_purpose=args.llm_purpose,
            disable_transform_reuse=args.disable_reuse,
        )

    elif args.command == "stage2-general-ablation":
        run_stage2_general_ablation(
            experiment_path=args.path,
            variants=args.variants,
            max_tables=args.max_tables,
            batch_size=args.batch_size,
            llm_purpose=args.llm_purpose,
            disable_transform_reuse=args.disable_reuse,
        )

    elif args.command == "analyze":
        path = args.path
        if not path and args.name:
            path = str(lake_data_path("experiments", args.name, "results.json"))
        if not path:
            print("Error: --path or --name required")
            return 1
        analysis = analyze_experiment(path)
        if isinstance(analysis, dict) and "error" in analysis:
            print(f"Error: {analysis['error']}")
            return 1

    elif args.command == "list":
        experiments = list_experiments()
        if not experiments:
            print("No experiments found.")
        else:
            print(f"\n{'Name':<50} {'Type':<25} {'Variants':<10}")
            print("-" * 85)
            for exp in experiments:
                print(f"  {exp['name']:<48} {exp['type']:<25} {exp['n_variants']:<10}")
            print()

    elif args.command == "llm-model-ablation":
        datasets = args.datasets.split()
        if args.phase == "link-stage1":
            link_existing_stage1(
                variant_name=args.variant_name,
                datasets=datasets,
            )
        elif args.phase == "stage1":
            run_llm_ablation_stage1(
                variant_name=args.variant_name,
                datasets=datasets,
                n_iterations=args.n_iterations,
                total_queries=args.total_queries,
                target_classes=args.target_classes,
                rag_type=args.rag_type,
            )
        elif args.phase == "stage2":
            if not args.stage1_variant:
                print("Error: --stage1-variant required for --phase stage2")
                return 1
            run_llm_ablation_stage2(
                variant_name=args.variant_name,
                stage1_variant=args.stage1_variant,
                datasets=datasets,
                max_tables=args.max_tables,
                batch_size=args.batch_size,
                budget_cap=args.budget_cap,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
