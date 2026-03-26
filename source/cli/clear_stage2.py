#!/usr/bin/env python3
"""
Clear Stage 2 intermediate data for specified datasets.

Drops LanceDB tables and filesystem checkpoint caches so that
Stage 2 (Column Summary) can be re-run from scratch.

Usage:
    python cli/clear_stage2.py --datasets "chembl chicago fetaqapn public_bi"
    python cli/clear_stage2.py --datasets chembl  # single dataset
"""

import argparse
import shutil
import sys
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_SOURCE_DIR = _SCRIPT_DIR.parent
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from loguru import logger

PROJECT_ROOT = _SOURCE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# LanceDB table suffixes to drop per dataset
STAGE2_TABLE_SUFFIXES = [
    "_column_summaries",
    "_column_mappings",
    "_table_defined_classes",
    "_table_summaries_retrieval",
    "_transform_contracts",
]


def clear_stage2(datasets: list[str], dry_run: bool = False) -> None:
    """Clear Stage 2 data for given datasets."""
    import lancedb

    lancedb_dir = DATA_DIR / "lake" / "lancedb"
    db = lancedb.connect(str(lancedb_dir))
    existing_tables = set(db.table_names())

    for ds in datasets:
        logger.info(f"\n--- {ds} ---")

        # 1. Drop LanceDB tables and remove .lance directories
        for suffix in STAGE2_TABLE_SUFFIXES:
            table_name = f"{ds}{suffix}"
            if table_name in existing_tables:
                if dry_run:
                    logger.info(f"  [dry-run] Would drop table: {table_name}")
                else:
                    db.drop_table(table_name)
                    logger.info(f"  ✓ Dropped table: {table_name}")
            else:
                logger.info(f"  ○ Table not found: {table_name}")
            # Also remove orphaned .lance directory on disk
            lance_dir = lancedb_dir / f"{table_name}.lance"
            if lance_dir.exists():
                if dry_run:
                    logger.info(f"  [dry-run] Would remove dir: {lance_dir.name}")
                else:
                    shutil.rmtree(lance_dir)
                    logger.info(f"  ✓ Removed dir: {lance_dir.name}")

        # 2. Clear checkpoint cache
        cache_dir = DATA_DIR / "cache" / "column_summaries"
        if cache_dir.exists():
            # Only remove files matching this dataset
            pattern = f"{ds}_batch_*"
            matched = list(cache_dir.glob(pattern))
            if matched:
                for f in matched:
                    if dry_run:
                        logger.info(f"  [dry-run] Would remove: {f.name}")
                    else:
                        f.unlink()
                        logger.info(f"  ✓ Removed cache: {f.name}")
            else:
                logger.info(f"  ○ No checkpoint files for {ds}")

    logger.info("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Clear Stage 2 intermediate data")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Space-separated list of dataset names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<level>{time:HH:mm:ss}</level> | {message}", level="INFO")

    datasets = args.datasets.split()
    clear_stage2(datasets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
