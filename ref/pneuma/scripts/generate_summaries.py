#!/usr/bin/env python3
"""
Generate Pneuma Summaries Using OpenAI-Compatible API

Uses Pneuma's official format to generate:
1. Schema narrations (column descriptions via LLM)
2. Sample row summaries (direct row value concatenation, no LLM)

Usage:
    python generate_summaries.py --dataset chembl --tables-dir /path/to/tables
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm
from openai import OpenAI

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)


def get_col_narration_prompt(columns: str, column: str) -> str:
    """
    Returns the OFFICIAL Pneuma prompt to narrate a column.
    
    Copied from: pneuma/src/pneuma/summarizer/summarizer.py
    """
    return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


def load_table_csv(csv_file: Path) -> Tuple[List[str], List[List[str]]]:
    """Load columns and rows from CSV file."""
    columns = []
    rows = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                columns = row
            else:
                if row:  # Skip empty rows
                    rows.append(row)
    
    return columns, rows


class SummaryGenerator:
    """Generate Pneuma summaries using OpenAI API."""
    
    def __init__(
        self,
        openai_url: str,
        openai_model: str,
        api_key: str = "token-abc123",
        max_workers: int = 8,
    ):
        self.client = OpenAI(
            base_url=openai_url,
            api_key=api_key,
        )
        self.model = openai_model
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    def generate_column_narration(self, columns: List[str], column: str) -> str:
        """Generate narration for a single column."""
        columns_str = " | ".join(columns)
        prompt = get_col_narration_prompt(columns_str, column)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating narration for {column}: {e}")
            return "No description."
    
    def generate_table_summaries(
        self,
        table_id: str,
        columns: List[str],
        rows: List[List[str]],
        sample_rows: int = 5,
        table_idx: int = 0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate all summaries for a table.
        
        Args:
            table_id: Table identifier
            columns: List of column names
            rows: List of row data
            sample_rows: Number of rows to sample (default 5, like original Pneuma)
            table_idx: Index for random seed
        
        Returns:
            Tuple of (schema_narrations, row_summaries)
        """
        # Generate column narrations using LLM
        schema_narrations = []
        for col in columns:
            if col.strip():  # Skip empty columns
                narration = self.generate_column_narration(columns, col)
                schema_narrations.append({
                    "table": table_id,
                    "column": col,
                    "summary": narration,
                })
        
        # Generate sample row summaries (NO LLM - direct concatenation)
        # This matches original Pneuma's generate_content_sample_rows.py
        row_summaries = []
        
        if rows:
            import random
            random.seed(table_idx)  # For reproducibility
            
            # Sample rows
            n_samples = min(len(rows), sample_rows)
            sampled_indices = random.sample(range(len(rows)), n_samples)
            
            for row_idx, sample_idx in enumerate(sampled_indices):
                row = rows[sample_idx]
                # Format: "col1: val1 | col2: val2 | col3: val3"
                formatted_row = " | ".join([
                    f"{col}: {val}" 
                    for col, val in zip(columns, row)
                    if col.strip()  # Skip empty column names
                ])
                
                row_summaries.append({
                    "id": f"{table_id}_SEP_contents_SEP_row-{row_idx}",
                    "table": table_id,
                    "summary": formatted_row,
                })
        
        return schema_narrations, row_summaries


def process_tables(
    tables_dir: Path,
    output_dir: Path,
    generator: SummaryGenerator,
    resume: bool = True,
    sample_rows: int = 5,
) -> Tuple[int, int]:
    """
    Process all tables and generate summaries.
    
    Args:
        tables_dir: Directory containing CSV tables
        output_dir: Output directory for summaries
        generator: SummaryGenerator instance
        resume: Resume from checkpoint if available
        sample_rows: Number of rows to sample per table
        
    Returns:
        Tuple of (tables_processed, total_narrations)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema_file = output_dir / "schema_narrations.jsonl"
    row_file = output_dir / "sample_rows.jsonl"
    checkpoint_file = output_dir / "checkpoint.json"
    
    # Load checkpoint
    processed_tables = set()
    if resume and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_tables = set(checkpoint.get("processed", []))
        print(f"Resuming from checkpoint: {len(processed_tables)} tables already processed")
    
    # Get table files
    csv_files = sorted(list(tables_dir.glob("*.csv")))
    remaining_files = [(i, f) for i, f in enumerate(csv_files) if f.stem not in processed_tables]
    
    print(f"Total tables: {len(csv_files)}")
    print(f"Remaining: {len(remaining_files)}")
    
    if not remaining_files:
        print("All tables already processed!")
        return len(csv_files), 0
    
    total_narrations = 0
    total_rows = 0
    
    # Process tables
    with open(schema_file, 'a') as f_schema, open(row_file, 'a') as f_row:
        for table_idx, csv_file in tqdm(remaining_files, desc="Generating summaries"):
            table_id = csv_file.stem
            
            try:
                columns, rows = load_table_csv(csv_file)
                
                if not columns:
                    print(f"Warning: Empty table {table_id}")
                    processed_tables.add(table_id)
                    continue
                
                # Generate summaries
                schema_narrations, row_summaries = generator.generate_table_summaries(
                    table_id=table_id,
                    columns=columns,
                    rows=rows,
                    sample_rows=sample_rows,
                    table_idx=table_idx,
                )
                
                # Write schema narrations
                for narration in schema_narrations:
                    f_schema.write(json.dumps(narration, ensure_ascii=False) + "\n")
                    total_narrations += 1
                
                # Write row summaries
                for row_summary in row_summaries:
                    f_row.write(json.dumps(row_summary, ensure_ascii=False) + "\n")
                    total_rows += 1
                
                # Flush periodically
                if len(processed_tables) % 10 == 0:
                    f_schema.flush()
                    f_row.flush()
                
                # Update checkpoint
                processed_tables.add(table_id)
                if len(processed_tables) % 50 == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({"processed": list(processed_tables)}, f)
                
            except Exception as e:
                print(f"Error processing {table_id}: {e}")
                continue
    
    # Final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({"processed": list(processed_tables)}, f)
    
    print(f"\nGenerated {total_narrations} schema narrations")
    print(f"Generated {total_rows} row summaries")
    
    return len(csv_files), total_narrations


def generate_sample_rows_only(
    tables_dir: Path,
    output_dir: Path,
    sample_rows: int = 5,
):
    """
    Generate ONLY sample rows without LLM (fast, for regeneration).
    
    This matches original Pneuma's generate_content_sample_rows.py exactly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    row_file = output_dir / "sample_rows.jsonl"
    
    csv_files = sorted(list(tables_dir.glob("*.csv")))
    print(f"Processing {len(csv_files)} tables...")
    
    all_rows = []
    
    for table_idx, csv_file in enumerate(tqdm(csv_files, desc="Generating sample rows")):
        table_id = csv_file.stem
        
        try:
            columns, rows = load_table_csv(csv_file)
            
            if not columns or not rows:
                continue
            
            import random
            random.seed(table_idx)
            
            # Sample rows (like original Pneuma)
            n_samples = min(len(rows), sample_rows)
            sampled_indices = random.sample(range(len(rows)), n_samples)
            
            for row_idx, sample_idx in enumerate(sampled_indices):
                row = rows[sample_idx]
                # Format: "col1: val1 | col2: val2 | col3: val3"
                formatted_row = " | ".join([
                    f"{col}: {val}" 
                    for col, val in zip(columns, row)
                    if col.strip()
                ])
                
                all_rows.append({
                    "id": f"{table_id}_SEP_contents_SEP_row-{row_idx}",
                    "table": table_id,
                    "summary": formatted_row,
                })
                
        except Exception as e:
            print(f"Error processing {table_id}: {e}")
            continue
    
    # Write all rows
    with open(row_file, 'w') as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(all_rows)} row summaries to {row_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Pneuma Summaries")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    parser.add_argument("--tables-dir", required=True, help="Directory containing CSV tables")
    parser.add_argument("--output-dir", required=True, help="Output directory for summaries")
    parser.add_argument("--openai-url", default="http://10.120.47.91:8000/v1",
                        help="OpenAI-compatible API URL")
    parser.add_argument("--openai-model", default="Qwen3-Next-80B-A3B-Instruct",
                        help="Model name")
    parser.add_argument("--sample-rows", type=int, default=5,
                        help="Number of rows to sample per table (default: 5)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoint")
    parser.add_argument("--rows-only", action="store_true",
                        help="Generate only sample rows (no LLM, fast)")
    
    args = parser.parse_args()
    
    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Dataset: {args.dataset}")
    print(f"Tables directory: {tables_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rows per table: {args.sample_rows}")
    print()
    
    if args.rows_only:
        # Fast mode: only sample rows, no LLM
        generate_sample_rows_only(
            tables_dir=tables_dir,
            output_dir=output_dir,
            sample_rows=args.sample_rows,
        )
    else:
        # Full mode: schema narrations (LLM) + sample rows (no LLM)
        generator = SummaryGenerator(
            openai_url=args.openai_url,
            openai_model=args.openai_model,
        )
        
        process_tables(
            tables_dir=tables_dir,
            output_dir=output_dir,
            generator=generator,
            resume=not args.no_resume,
            sample_rows=args.sample_rows,
        )


if __name__ == "__main__":
    main()
