#!/usr/bin/env python3
"""
Plot cumulative LLM call / token curves from experiment stats.

Reads LLM stats JSON files produced by run_pipeline.py and plots:
- X-axis: cumulative processed tables
- Y-axis: cumulative LLM requests / tokens
- One line per LLM caller (purpose)

Supports side-by-side comparison of w/o reuse vs w/ reuse.

Usage:
    # Plot a single experiment directory
    python draw/plot_llm_stats_curve.py --exp-dir logs/experiments/ablation_no_reuse_20260315_120000

    # Compare two experiment directories (w/o reuse vs w/ reuse)
    python draw/plot_llm_stats_curve.py \
        --no-reuse-dir logs/experiments/ablation_no_reuse_20260315_120000 \
        --with-reuse-dir logs/experiments/ablation_with_reuse_20260315_130000

    # Plot specific datasets
    python draw/plot_llm_stats_curve.py --exp-dir ... --datasets chembl chicago
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_SOURCE_DIR = _SCRIPT_DIR.parent
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = _SOURCE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "visualizations"

# Consistent color palette for callers
CALLER_COLORS = {
    "classify_columns_node": "#2196F3",   # blue
    "analyze_columns_node": "#F44336",     # red
    "expand_virtual_columns_node": "#4CAF50",  # green
    "unknown": "#9E9E9E",                 # grey
}
FALLBACK_COLORS = ["#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]


def load_stats_files(exp_dir: str, datasets: Optional[list[str]] = None) -> dict:
    """Load all llm_stats JSON files from an experiment directory.

    Returns:
        {dataset_name: stats_data_dict}
    """
    exp_path = Path(exp_dir)
    result = {}
    for f in sorted(exp_path.glob("*_llm_stats_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        ds = data.get("dataset_name", f.stem.split("_llm_stats")[0])
        if datasets and ds not in datasets:
            continue
        result[ds] = data
    return result


def _get_color(caller: str, idx: int) -> str:
    """Get consistent color for a caller."""
    if caller in CALLER_COLORS:
        return CALLER_COLORS[caller]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def _short_caller(name: str) -> str:
    """Shorten node names for legend."""
    return name.replace("_node", "").replace("_columns", "_col")


def plot_single_experiment(
    stats_by_dataset: dict,
    metric: str = "requests",
    output_path: Optional[str] = None,
    title_prefix: str = "",
) -> None:
    """Plot cumulative LLM metric curves for each dataset in a single experiment."""
    datasets = sorted(stats_by_dataset.keys())
    n = len(datasets)
    if n == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax_idx, ds in enumerate(datasets):
        ax = axes[0][ax_idx]
        data = stats_by_dataset[ds]
        timeline = data.get("timeline", [])
        if not timeline:
            ax.set_title(f"{ds} (no data)")
            continue

        # Collect all callers
        all_callers = set()
        for snap in timeline:
            all_callers.update(snap.get("cumulative_by_caller", {}).keys())
        all_callers = sorted(all_callers)

        # Plot each caller
        extra_idx = 0
        for caller in all_callers:
            x_vals = []
            y_vals = []
            for snap in timeline:
                tables = snap.get("total_tables", 0)
                caller_stats = snap.get("cumulative_by_caller", {}).get(caller, {})
                val = caller_stats.get(metric, caller_stats.get(f"total_{metric}", 0))
                x_vals.append(tables)
                y_vals.append(val)

            color = _get_color(caller, extra_idx)
            if caller not in CALLER_COLORS:
                extra_idx += 1
            ax.plot(x_vals, y_vals, label=_short_caller(caller), color=color, linewidth=1.5)

        # Also plot total
        x_total = [snap.get("total_tables", 0) for snap in timeline]
        key = f"total_{metric}" if metric != "requests" else "total_requests"
        y_total = [snap.get("cumulative_totals", {}).get(key, 0) for snap in timeline]
        ax.plot(x_total, y_total, label="total", color="black", linewidth=2, linestyle="--")

        ax.set_xlabel("Processed Tables")
        ax.set_ylabel(f"Cumulative {metric.replace('_', ' ').title()}")
        ax.set_title(f"{title_prefix}{ds}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison(
    no_reuse_stats: dict,
    with_reuse_stats: dict,
    metric: str = "requests",
    output_path: Optional[str] = None,
) -> None:
    """Plot side-by-side comparison of w/o reuse vs w/ reuse for each dataset."""
    all_datasets = sorted(set(no_reuse_stats.keys()) | set(with_reuse_stats.keys()))
    n = len(all_datasets)
    if n == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), squeeze=False)

    for variant_idx, (label, stats) in enumerate([
        ("w/o Reuse", no_reuse_stats),
        ("w/ Reuse", with_reuse_stats),
    ]):
        for ds_idx, ds in enumerate(all_datasets):
            ax = axes[variant_idx][ds_idx]
            data = stats.get(ds, {})
            timeline = data.get("timeline", [])

            if not timeline:
                ax.set_title(f"{label}: {ds} (no data)")
                continue

            all_callers = set()
            for snap in timeline:
                all_callers.update(snap.get("cumulative_by_caller", {}).keys())
            all_callers = sorted(all_callers)

            extra_idx = 0
            for caller in all_callers:
                x_vals, y_vals = [], []
                for snap in timeline:
                    tables = snap.get("total_tables", 0)
                    caller_stats = snap.get("cumulative_by_caller", {}).get(caller, {})
                    val = caller_stats.get(metric, caller_stats.get(f"total_{metric}", 0))
                    x_vals.append(tables)
                    y_vals.append(val)

                color = _get_color(caller, extra_idx)
                if caller not in CALLER_COLORS:
                    extra_idx += 1
                ax.plot(x_vals, y_vals, label=_short_caller(caller), color=color, linewidth=1.5)

            x_total = [snap.get("total_tables", 0) for snap in timeline]
            key = f"total_{metric}" if metric != "requests" else "total_requests"
            y_total = [snap.get("cumulative_totals", {}).get(key, 0) for snap in timeline]
            ax.plot(x_total, y_total, label="total", color="black", linewidth=2, linestyle="--")

            reuse_count = data.get("code_reuse_count", 0)
            title = f"{label}: {ds}"
            if reuse_count:
                title += f" (reuse={reuse_count})"
            ax.set_title(title)
            ax.set_xlabel("Processed Tables")
            ax.set_ylabel(f"Cumulative {metric.replace('_', ' ').title()}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot LLM stats curves from experiment data")
    parser.add_argument("--exp-dir", type=str, help="Single experiment directory to plot")
    parser.add_argument("--no-reuse-dir", type=str, help="Experiment dir for w/o reuse variant")
    parser.add_argument("--with-reuse-dir", type=str, help="Experiment dir for w/ reuse variant")
    parser.add_argument("--datasets", type=str, default=None, help="Space-separated dataset filter")
    parser.add_argument("--metric", type=str, default="requests",
                        choices=["requests", "total_tokens", "input_tokens", "output_tokens"],
                        help="Metric to plot (default: requests)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    ds_filter = args.datasets.split() if args.datasets else None

    if args.no_reuse_dir and args.with_reuse_dir:
        nr = load_stats_files(args.no_reuse_dir, ds_filter)
        wr = load_stats_files(args.with_reuse_dir, ds_filter)
        out = args.output or str(OUTPUT_DIR / f"llm_stats_comparison_{args.metric}.png")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_comparison(nr, wr, metric=args.metric, output_path=out)
    elif args.exp_dir:
        stats = load_stats_files(args.exp_dir, ds_filter)
        out = args.output or str(OUTPUT_DIR / f"llm_stats_single_{args.metric}.png")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_single_experiment(stats, metric=args.metric, output_path=out)
    else:
        parser.error("Provide --exp-dir or both --no-reuse-dir and --with-reuse-dir")


if __name__ == "__main__":
    main()
