#!/usr/bin/env python3
"""
Batch BM25 Analysis Runner & Summary Exporter

Runs bm25_analysis.py in full mode across all datasets and aggregates results
into a summary table showing:
  - Mean primitive class contribution ratio
  - Ablation-remove performance delta (recall@1, recall@5, recall@10, mrr)
  - Ablation-keep performance delta (same metrics)

Usage:
    python -m evaluation.run_bm25_batch                      # Run all + summarize
    python -m evaluation.run_bm25_batch --summarize-only     # Only read existing JSONs
    python -m evaluation.run_bm25_batch --datasets chembl fetaqa  # Specific datasets
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import _path_setup  # noqa: F401
from core.paths import get_db_path


ALL_DATASETS = [
    "adventure_works",
    "chembl",
    "fetaqa",
    "fetaqapn",
    "public_bi",
    "bird",
    "chicago",
]

EVAL_DIR = get_db_path() / "eval_results"


def run_bm25_analysis(dataset: str, num_queries: int = -1) -> Optional[Path]:
    """
    Run bm25_analysis.py --mode full for a single dataset.

    Returns:
        Path to output JSON, or None on failure.
    """
    output_name = f"{dataset}_bm25_full.json"
    output_path = EVAL_DIR / output_name

    cmd = [
        sys.executable, "-m", "evaluation.runners.bm25_analysis",
        "-d", dataset,
        "--mode", "full",
        "-n", str(num_queries),
        "--show-cases", "0",
        "--output", output_name,
    ]

    print(f"\n{'='*70}")
    print(f"  Running BM25 analysis: {dataset}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=False,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  ✗ FAILED: {dataset} (exit code {result.returncode})")
            return None
        print(f"  ✓ Completed: {dataset}")
        return output_path
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT: {dataset}")
        return None
    except Exception as e:
        print(f"  ✗ ERROR: {dataset}: {e}")
        return None


def load_results(dataset: str) -> Optional[Dict[str, Any]]:
    """Load a dataset's BM25 analysis JSON results."""
    output_path = EVAL_DIR / f"{dataset}_bm25_full.json"
    if not output_path.exists():
        return None
    with open(output_path) as f:
        return json.load(f)


def summarize_results(datasets: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load and summarize results from all datasets.

    Returns:
        Dict[dataset_name -> summary_dict]
    """
    summaries = {}

    for ds in datasets:
        data = load_results(ds)
        if data is None:
            print(f"  ⚠ No results for {ds}")
            continue

        contrib = data.get("contribution", {})
        ablation_remove = data.get("ablation_remove", {})
        ablation_keep = data.get("ablation_keep", {})

        stats = contrib.get("stats", {})
        total_q = contrib.get("total_queries", 0)
        q_with_gt = contrib.get("queries_with_gt_in_index", 0)

        summaries[ds] = {
            "total_queries": total_q,
            "queries_with_gt": q_with_gt,
            "mean_prim_ratio": stats.get("mean_primitive_ratio", None),
            "median_prim_ratio": stats.get("median_primitive_ratio", None),
            # Ablation remove
            "remove_orig_r1": _safe_get(ablation_remove, "original_metrics", "recall@1"),
            "remove_orig_mrr": _safe_get(ablation_remove, "original_metrics", "mrr"),
            "remove_delta_r1": _safe_get(ablation_remove, "delta", "recall@1"),
            "remove_delta_r5": _safe_get(ablation_remove, "delta", "recall@5"),
            "remove_delta_r10": _safe_get(ablation_remove, "delta", "recall@10"),
            "remove_delta_mrr": _safe_get(ablation_remove, "delta", "mrr"),
            # Ablation keep
            "keep_orig_r1": _safe_get(ablation_keep, "original_metrics", "recall@1"),
            "keep_delta_r1": _safe_get(ablation_keep, "delta", "recall@1"),
            "keep_delta_r5": _safe_get(ablation_keep, "delta", "recall@5"),
            "keep_delta_r10": _safe_get(ablation_keep, "delta", "recall@10"),
            "keep_delta_mrr": _safe_get(ablation_keep, "delta", "mrr"),
        }

    return summaries


def _safe_get(data: Dict, *keys) -> Optional[float]:
    """Safely traverse nested dict keys."""
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
        if data is None:
            return None
    return data


def _fmt(val, fmt_str=".1%", fallback="—"):
    """Format a value as percentage, or fallback string."""
    if val is None:
        return fallback
    return f"{val:{fmt_str}}"


def _fmt_delta(val, fmt_str=".1%", fallback="—"):
    """Format delta with sign and pp unit."""
    if val is None:
        return fallback
    pp = val * 100
    sign = "+" if pp > 0 else ""
    return f"{sign}{pp:.1f}pp"


def print_summary_table(summaries: Dict[str, Dict[str, Any]]):
    """Print a formatted summary table."""
    if not summaries:
        print("No results to display.")
        return

    print(f"\n{'='*110}")
    print("  BM25 Primitive Class Contribution & Ablation Summary")
    print(f"{'='*110}")

    # Header
    print(f"{'Dataset':<18} {'#Q':>4} {'Mean%':>7} {'Med%':>7} "
          f"│ {'Remove':^30} │ {'Keep-only':^30}")
    print(f"{'':18} {'':>4} {'Ratio':>7} {'Ratio':>7} "
          f"│ {'ΔR@1':>8} {'ΔR@5':>8} {'ΔR@10':>8} {'ΔMRR':>8} "
          f"│ {'ΔR@1':>8} {'ΔR@5':>8} {'ΔR@10':>8} {'ΔMRR':>8}")
    print(f"{'─'*18}─{'─'*4}─{'─'*7}─{'─'*7}─"
          f"┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─"
          f"┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}")

    for ds, s in summaries.items():
        print(
            f"{ds:<18} {s['queries_with_gt']:>4} "
            f"{_fmt(s['mean_prim_ratio']):>7} {_fmt(s['median_prim_ratio']):>7} "
            f"│ {_fmt_delta(s['remove_delta_r1']):>8} {_fmt_delta(s['remove_delta_r5']):>8} "
            f"{_fmt_delta(s['remove_delta_r10']):>8} {_fmt_delta(s['remove_delta_mrr']):>8} "
            f"│ {_fmt_delta(s['keep_delta_r1']):>8} {_fmt_delta(s['keep_delta_r5']):>8} "
            f"{_fmt_delta(s['keep_delta_r10']):>8} {_fmt_delta(s['keep_delta_mrr']):>8}"
        )

    print(f"{'─'*110}")

    # Averages
    n = len(summaries)
    def avg_key(k):
        vals = [s[k] for s in summaries.values() if s[k] is not None]
        return sum(vals) / len(vals) if vals else None

    avg_mean_ratio = avg_key("mean_prim_ratio")
    avg_remove_r1 = avg_key("remove_delta_r1")
    avg_remove_r5 = avg_key("remove_delta_r5")
    avg_remove_r10 = avg_key("remove_delta_r10")
    avg_remove_mrr = avg_key("remove_delta_mrr")
    avg_keep_r1 = avg_key("keep_delta_r1")
    avg_keep_r5 = avg_key("keep_delta_r5")
    avg_keep_r10 = avg_key("keep_delta_r10")
    avg_keep_mrr = avg_key("keep_delta_mrr")

    print(
        f"{'AVERAGE':<18} {'':>4} "
        f"{_fmt(avg_mean_ratio):>7} {'':>7} "
        f"│ {_fmt_delta(avg_remove_r1):>8} {_fmt_delta(avg_remove_r5):>8} "
        f"{_fmt_delta(avg_remove_r10):>8} {_fmt_delta(avg_remove_mrr):>8} "
        f"│ {_fmt_delta(avg_keep_r1):>8} {_fmt_delta(avg_keep_r5):>8} "
        f"{_fmt_delta(avg_keep_r10):>8} {_fmt_delta(avg_keep_mrr):>8}"
    )
    print(f"{'='*110}")


def export_csv(summaries: Dict[str, Dict[str, Any]], output_path: Path):
    """Export summary to CSV."""
    import csv

    headers = [
        "dataset", "num_queries", "mean_prim_ratio", "median_prim_ratio",
        "remove_delta_r1", "remove_delta_r5", "remove_delta_r10", "remove_delta_mrr",
        "keep_delta_r1", "keep_delta_r5", "keep_delta_r10", "keep_delta_mrr",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for ds, s in summaries.items():
            row = {"dataset": ds}
            row["num_queries"] = s["queries_with_gt"]
            for k in headers[2:]:
                row[k] = s.get(k)
            writer.writerow(row)

    print(f"\n💾 CSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch BM25 analysis runner & summary exporter"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets to analyze (default: all {len(ALL_DATASETS)})"
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Only summarize existing results (skip running analysis)"
    )
    parser.add_argument(
        "-n", "--num-queries", type=int, default=-1,
        help="Number of queries per dataset (-1 for all)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export summary to CSV file"
    )
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS

    if not args.summarize_only:
        print(f"\n🔬 Running BM25 analysis for {len(datasets)} datasets...")
        for ds in datasets:
            run_bm25_analysis(ds, num_queries=args.num_queries)

    # Summarize
    print(f"\n📊 Summarizing results...")
    summaries = summarize_results(datasets)
    print_summary_table(summaries)

    # CSV export
    if args.csv:
        csv_path = EVAL_DIR / args.csv
        export_csv(summaries, csv_path)
    else:
        # Default CSV export
        csv_path = EVAL_DIR / "bm25_primitive_class_summary.csv"
        export_csv(summaries, csv_path)


if __name__ == "__main__":
    main()
