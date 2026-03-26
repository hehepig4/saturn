#!/usr/bin/env python3
"""
Aggregate Multiple Evaluation Runs

Combines evaluation results from multiple runs into a single aggregated dataset
for cross-run comparison and analysis.

Usage:
    # Aggregate all parquet files
    python -m evaluation.aggregate_results \
        --input-dir data/lake/lancedb/eval_results/exports/parquet \
        --output data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet
    
    # Aggregate specific runs
    python -m evaluation.aggregate_results \
        --input-files run1.parquet run2.parquet run3.parquet \
        --output aggregated.parquet
    
    # Generate summary statistics
    python -m evaluation.aggregate_results \
        --input-dir exports/parquet \
        --output aggregated.parquet \
        --summary summary.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required for this script")
    print("Install with: pip install pandas pyarrow")
    sys.exit(1)

from loguru import logger


def compute_metrics_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Hit@K and MRR from raw query results
    
    Expects columns: gt_tables, top_k_table_ids (both as lists)
    Adds columns: hit_at_1, hit_at_5, hit_at_10, hit_at_50, hit_at_100, mrr, gt_rank
    """
    if 'gt_tables' not in df.columns or 'top_k_table_ids' not in df.columns:
        logger.warning("Missing gt_tables or top_k_table_ids, skipping metric computation")
        return df
    
    k_values = [1, 5, 10, 50, 100]
    
    def find_best_rank(row):
        """Find the best (minimum) rank among all ground truth tables"""
        gt_tables = row.get('gt_tables', [])
        top_k_ids = row.get('top_k_table_ids', [])
        
        # Handle None or empty values
        if not gt_tables or not top_k_ids:
            return None
        
        # Handle string representation of lists
        if isinstance(gt_tables, str):
            try:
                import ast
                gt_tables = ast.literal_eval(gt_tables)
            except:
                gt_tables = []
        if isinstance(top_k_ids, str):
            try:
                import ast
                top_k_ids = ast.literal_eval(top_k_ids)
            except:
                top_k_ids = []
        
        if not gt_tables or not top_k_ids:
            return None
            
        best_rank = None
        for gt in gt_tables:
            try:
                rank = top_k_ids.index(gt) + 1  # 1-indexed
                if best_rank is None or rank < best_rank:
                    best_rank = rank
            except ValueError:
                continue
        return best_rank
    
    # Compute rank for each query
    df = df.copy()
    df['gt_rank'] = df.apply(find_best_rank, axis=1)
    
    # Compute hit@k for each query
    for k in k_values:
        col_name = f'hit_at_{k}'
        df[col_name] = df['gt_rank'].apply(
            lambda r: 1.0 if r is not None and r <= k else 0.0
        )
    
    # Compute MRR for each query
    df['mrr'] = df['gt_rank'].apply(
        lambda r: 1.0 / r if r is not None else 0.0
    )
    
    logger.info(f"Computed metrics: hit@{k_values}, mrr for {len(df)} queries")
    
    return df


def aggregate_parquet_files(
    input_files: List[Path],
    output_path: Path,
    summary_path: Optional[Path] = None
) -> pd.DataFrame:
    """Aggregate multiple parquet files into one
    
    Args:
        input_files: List of input parquet file paths
        output_path: Output file path for aggregated data
        summary_path: Optional output path for summary CSV
    
    Returns:
        Aggregated DataFrame
    """
    logger.info(f"Aggregating {len(input_files)} parquet files...")
    
    dfs = []
    for file_path in input_files:
        try:
            df = pd.read_parquet(file_path)
            logger.debug(f"  Loaded {len(df)} records from {file_path.name}")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed to load {file_path}: {e}")
    
    if not dfs:
        raise ValueError("No valid parquet files found")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total records")
    
    # Compute metrics from raw data if not already present
    if 'hit_at_1' not in combined_df.columns and 'gt_tables' in combined_df.columns:
        combined_df = compute_metrics_from_raw(combined_df)
    
    # Save aggregated results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"Saved aggregated results to: {output_path}")
    
    # Generate summary if requested
    if summary_path:
        generate_summary(combined_df, summary_path)
    
    return combined_df


def generate_summary(df: pd.DataFrame, output_path: Path):
    """Generate summary statistics from aggregated data
    
    Args:
        df: Aggregated DataFrame
        output_path: Output path for summary CSV
    """
    logger.info("Generating summary statistics...")
    
    # Group by key dimensions
    group_cols = []
    
    # Add available grouping columns
    for col in ['dataset', 'method', 'retriever_type', 'hyde_mode', 'tbox_iteration']:
        if col in df.columns:
            group_cols.append(col)
    
    if not group_cols:
        logger.warning("No grouping columns found, skipping summary")
        return
    
    # Aggregate metrics
    metric_cols = []
    for col in df.columns:
        if col.startswith('hit_at_') or col == 'mrr':
            metric_cols.append(col)
    
    if not metric_cols:
        logger.warning("No metric columns found, skipping summary")
        return
    
    # Compute summary statistics
    summary = df.groupby(group_cols).agg({
        **{col: ['mean', 'std', 'min', 'max'] for col in metric_cols},
        'query_id': 'count'  # Number of queries
    })
    
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.rename(columns={'query_id_count': 'num_queries'})
    summary = summary.reset_index()
    
    # Save summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"Saved summary statistics to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    # Show key metrics (mean values)
    display_cols = group_cols + ['num_queries']
    for col in metric_cols:
        mean_col = f"{col}_mean"
        if mean_col in summary.columns:
            display_cols.append(mean_col)
    
    display_df = summary[display_cols].copy()
    
    # Format percentages
    for col in display_df.columns:
        if col.startswith('hit_'):
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        elif col == 'mrr_mean':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print()


def find_parquet_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all parquet files in directory
    
    Args:
        directory: Directory to search
        recursive: Search recursively if True
    
    Returns:
        List of parquet file paths
    """
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return list(directory.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir", type=Path,
        help="Directory containing parquet files (searches recursively)",
    )
    input_group.add_argument(
        "--input-files", type=Path, nargs='+',
        help="Specific parquet files to aggregate"
    )
    
    # Output options
    parser.add_argument(
        "--output", type=Path, 
        help="Output file path for aggregated parquet",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="Output file path for summary CSV (optional)"
    )
    
    # Options
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Don't search recursively in input directory"
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
    
    # Find input files
    if args.input_dir:
        if not args.input_dir.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        
        input_files = find_parquet_files(args.input_dir, recursive=not args.no_recursive)
        
        if not input_files:
            logger.error(f"No parquet files found in: {args.input_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(input_files)} parquet files in {args.input_dir}")
    else:
        input_files = args.input_files
        
        # Verify all files exist
        missing = [f for f in input_files if not f.exists()]
        if missing:
            logger.error(f"Input files do not exist: {missing}")
            sys.exit(1)
    
    # Aggregate
    try:
        df = aggregate_parquet_files(
            input_files=input_files,
            output_path=args.output,
            summary_path=args.summary
        )
        
        logger.success(f"Successfully aggregated {len(df)} records")
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
