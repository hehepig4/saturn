#!/usr/bin/env python3
"""
Visualize Evaluation Results

Generate visualizations from aggregated evaluation results.

Usage:
    # Generate all default visualizations
    python -m evaluation.visualize_results \
        --data data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet \
        --output-dir data/lake/lancedb/eval_results/visualizations
    
    # Generate specific plots
    python -m evaluation.visualize_results \
        --data aggregated.parquet \
        --output-dir viz/ \
        --plots hyde_comparison retriever_comparison hit_curve
    
    # Custom filtering
    python -m evaluation.visualize_results \
        --data aggregated.parquet \
        --output-dir viz/ \
        --filter-dataset fetaqa public_bi \
        --filter-method hyde
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: Required packages not found")
    print("Install with: pip install pandas matplotlib seaborn")
    sys.exit(1)

from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_hyde_comparison(df: pd.DataFrame, output_path: Path):
    """Generate HyDE mode comparison bar charts
    
    Args:
        df: DataFrame with HyDE evaluation results
        output_path: Output file path
    """
    logger.info("Generating HyDE comparison plot...")
    
    # Filter HyDE results
    hyde_df = df[df['hyde_mode'] == 'combined'].copy()
    
    if hyde_df.empty:
        logger.warning("No HyDE results found, skipping plot")
        return
    
    # Aggregate by dataset and HyDE mode
    agg_df = hyde_df.groupby(['dataset', 'hyde_mode']).agg({
        'hit_at_1': 'mean',
        'hit_at_10': 'mean',
        'mrr': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ('hit_at_1', 'Hit@1'),
        ('hit_at_10', 'Hit@10'),
        ('mrr', 'MRR')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        # Pivot for grouped bar chart
        pivot = agg_df.pivot(index='dataset', columns='hyde_mode', values=metric)
        
        # Plot
        pivot.plot(kind='bar', ax=ax, rot=45, width=0.8)
        ax.set_title(f'{title} by HyDE Mode', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.legend(title='HyDE Mode', title_fontsize=10, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis as percentage for hit metrics
        if metric.startswith('hit_'):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.suptitle('HyDE Mode Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved HyDE comparison plot to: {output_path}")
    plt.close()


def plot_retriever_comparison(df: pd.DataFrame, output_path: Path):
    """Generate retriever type comparison heatmap
    
    Args:
        df: DataFrame with evaluation results
        output_path: Output file path
    """
    logger.info("Generating retriever comparison plot...")
    
    # Filter results with retriever_type
    retriever_df = df[df['retriever_type'].notna()].copy()
    
    if retriever_df.empty:
        logger.warning("No retriever type data found, skipping plot")
        return
    
    # Aggregate by dataset and retriever type
    agg_df = retriever_df.groupby(['dataset', 'retriever_type']).agg({
        'hit_at_1': 'mean',
        'hit_at_10': 'mean',
        'mrr': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = [
        ('hit_at_1', 'Hit@1', '.0%'),
        ('hit_at_10', 'Hit@10', '.0%'),
        ('mrr', 'MRR', '.3f')
    ]
    
    for ax, (metric, title, fmt) in zip(axes, metrics):
        # Pivot for heatmap
        pivot = agg_df.pivot(index='dataset', columns='retriever_type', values=metric)
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap='YlGnBu', ax=ax,
                   cbar_kws={'label': title}, vmin=0, vmax=pivot.max().max())
        ax.set_title(f'{title} by Retriever Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Retriever Type', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
    
    plt.suptitle('Retriever Type Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved retriever comparison plot to: {output_path}")
    plt.close()


def plot_hit_curve(df: pd.DataFrame, output_path: Path):
    """Generate Hit@K curves
    
    Args:
        df: DataFrame with evaluation results
        output_path: Output file path
    """
    logger.info("Generating Hit@K curve plot...")
    
    hit_cols = ['hit_at_1', 'hit_at_5', 'hit_at_10', 'hit_at_50', 'hit_at_100']
    k_values = [1, 5, 10, 50, 100]
    
    # Check if all hit columns exist
    missing_cols = [col for col in hit_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}, skipping plot")
        return
    
    # Group by method and configuration
    group_cols = ['dataset', 'method']
    
    # Add optional grouping columns
    for col in ['retriever_type', 'hyde_mode']:
        if col in df.columns and df[col].notna().any():
            group_cols.append(col)
    
    grouped = df.groupby(group_cols)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Plot each group
    for name, group in grouped:
        # Create label from group name
        if isinstance(name, tuple):
            label = "-".join([str(x) if pd.notna(x) else "" for x in name])
        else:
            label = str(name)
        
        # Compute mean hits
        hits = [group[col].mean() for col in hit_cols]
        
        # Plot
        plt.plot(k_values, hits, marker='o', label=label, linewidth=2, markersize=6)
    
    plt.xlabel('K', fontsize=14, fontweight='bold')
    plt.ylabel('Hit@K', fontsize=14, fontweight='bold')
    plt.title('Hit@K Curves', fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.xticks(k_values, k_values)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved Hit@K curve plot to: {output_path}")
    plt.close()


def plot_dataset_comparison(df: pd.DataFrame, output_path: Path):
    """Generate dataset comparison plot
    
    Args:
        df: DataFrame with evaluation results
        output_path: Output file path
    """
    logger.info("Generating dataset comparison plot...")
    
    # Aggregate by dataset and method
    agg_df = df.groupby(['dataset', 'method']).agg({
        'hit_at_1': 'mean',
        'hit_at_10': 'mean',
        'mrr': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ('hit_at_1', 'Hit@1'),
        ('hit_at_10', 'Hit@10'),
        ('mrr', 'MRR')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        # Pivot for grouped bar chart
        pivot = agg_df.pivot(index='dataset', columns='method', values=metric)
        
        # Plot
        pivot.plot(kind='bar', ax=ax, rot=45, width=0.8)
        ax.set_title(f'{title} by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.legend(title='Method', title_fontsize=10, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis as percentage for hit metrics
        if metric.startswith('hit_'):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.suptitle('Dataset Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved dataset comparison plot to: {output_path}")
    plt.close()


def plot_method_performance_matrix(df: pd.DataFrame, output_path: Path):
    """Generate method performance matrix heatmap
    
    Args:
        df: DataFrame with evaluation results
        output_path: Output file path
    """
    logger.info("Generating method performance matrix...")
    
    # Aggregate by dataset and method
    agg_df = df.groupby(['dataset', 'method']).agg({
        'hit_at_10': 'mean'
    }).reset_index()
    
    # Pivot for heatmap
    pivot = agg_df.pivot(index='dataset', columns='method', values='hit_at_10')
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
               cbar_kws={'label': 'Hit@10'},
               vmin=0, vmax=1.0)
    
    plt.title('Method Performance Matrix (Hit@10)', fontsize=16, fontweight='bold')
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Dataset', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved method performance matrix to: {output_path}")
    plt.close()


# Available plot functions
PLOT_FUNCTIONS = {
    'hyde_comparison': plot_hyde_comparison,
    'retriever_comparison': plot_retriever_comparison,
    'hit_curve': plot_hit_curve,
    'dataset_comparison': plot_dataset_comparison,
    'performance_matrix': plot_method_performance_matrix,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Input parquet file with aggregated results"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for plots"
    )
    
    # Plot selection
    parser.add_argument(
        "--plots", nargs='+', choices=list(PLOT_FUNCTIONS.keys()),
        default=None,
        help=f"Specific plots to generate (default: all). Options: {list(PLOT_FUNCTIONS.keys())}"
    )
    
    # Filtering
    parser.add_argument(
        "--filter-dataset", nargs='+',
        help="Filter to specific datasets"
    )
    parser.add_argument(
        "--filter-method", nargs='+',
        help="Filter to specific methods"
    )
    parser.add_argument(
        "--filter-retriever", nargs='+',
        help="Filter to specific retriever types"
    )
    
    # Options
    parser.add_argument(
        "--format", choices=['png', 'pdf', 'svg'], default='png',
        help="Output image format (default: png)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=log_level)
    
    # Load data
    if not args.data.exists():
        logger.error(f"Data file does not exist: {args.data}")
        sys.exit(1)
    
    logger.info(f"Loading data from: {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} records")
    
    # Apply filters
    if args.filter_dataset:
        df = df[df['dataset'].isin(args.filter_dataset)]
        logger.info(f"Filtered to datasets: {args.filter_dataset} ({len(df)} records)")
    
    if args.filter_method:
        df = df[df['method'].isin(args.filter_method)]
        logger.info(f"Filtered to methods: {args.filter_method} ({len(df)} records)")
    
    if args.filter_retriever:
        df = df[df['retriever_type'].isin(args.filter_retriever)]
        logger.info(f"Filtered to retrievers: {args.filter_retriever} ({len(df)} records)")
    
    if df.empty:
        logger.error("No data remaining after filtering")
        sys.exit(1)
    
    # Determine which plots to generate
    plots_to_generate = args.plots if args.plots else list(PLOT_FUNCTIONS.keys())
    
    # Generate plots
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_name in plots_to_generate:
        plot_func = PLOT_FUNCTIONS[plot_name]
        output_path = args.output_dir / f"{plot_name}.{args.format}"
        
        try:
            plot_func(df, output_path)
        except Exception as e:
            logger.error(f"Failed to generate {plot_name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    logger.success(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
