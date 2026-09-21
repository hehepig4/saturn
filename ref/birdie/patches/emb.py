"""
Birdie Table Embedding Generator

Generates embeddings for tables using BGE-M3 (default) or other embedding models.

Modifications from original BIRDIE:
- Default embedding model changed to BGE-M3 for consistency with project
- Added help text for model_name parameter

Usage:
    python emb.py --table_data_path <path> --output_dir <dir> [--model_name <model>]
"""

import json
import numpy as np
from tqdm import tqdm
import argparse
import os


# Default embedding model: BGE-M3 (aligned with Pneuma and project implementation)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def json_to_markdown(table):
    """Convert table JSON to markdown format."""
    columns = [col.get('text', '') for col in table.get('columns', [])]
    rows = table.get('rows', [])

    # Header
    header_row = '|' + '|'.join(col if col else ' ' for col in columns) + '|'
    separator = '|' + '|'.join(['---'] * len(columns)) + '|'

    # Content rows
    content_rows = []
    for row in rows:
        cells = [cell.get('text', '') for cell in row.get('cells', [])]
        content_rows.append('|' + '|'.join(cells) + '|')

    return '\n'.join([header_row, separator] + content_rows)


def title_schema_text(table):
    """Coarse-grained view: document title + sorted schema column names.
    Mirrors BIRDIE official emb.py title+schema embedding (emd1, shallow layers).
    """
    title = table.get('documentTitle', '')
    columns = [col.get('text', '') for col in table.get('columns', [])]
    schema = ",".join(sorted(columns, key=lambda x: x[0] if len(x) > 0 else ""))
    return f"{title}\n{schema}"


def table2text(table):
    """Fine-grained view: full table content as 'col: cell, ...' per row.
    Mirrors BIRDIE official emb.py table_data embedding (emd2, deep layers).
    """
    columns = [col.get('text', '') for col in table.get('columns', [])]
    table_string = ""
    for row in table.get('rows', []):
        cells = [cell.get('text', '') for cell in row.get('cells', [])]
        row_string = ""
        for i, cell in enumerate(cells):
            col = columns[i] if i < len(columns) else ""
            row_string += f"{col}: {cell}, "
        row_string = row_string[:-2] + "\n" if row_string else "\n"
        table_string += row_string
    return table_string


def generate_embeddings(table_data_path, output_dir, model_name):
    """Generate embeddings for all tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Load tables
    with open(table_data_path, 'r') as f:
        tables = json.load(f)

    print(f"Loaded {len(tables)} tables")
    print(f"Using embedding model: {model_name}")

    # Import embedding model
    from FlagEmbedding import FlagModel

    # Resolve model path - convert relative paths to absolute
    if model_name.startswith('/') or model_name.startswith('BAAI/'):
        resolved_model = model_name
    else:
        # Might be a relative path, try to resolve
        resolved_model = os.path.abspath(model_name)
        if not os.path.isdir(resolved_model):
            resolved_model = model_name  # Fall back to original (HF model name)

    # Check if model_name is a local path
    if os.path.isdir(resolved_model):
        print(f"Loading model from local path: {resolved_model}")
    else:
        print(f"Loading model from HuggingFace: {resolved_model}")

    model = FlagModel(resolved_model, use_fp16=True)

    # Generate two complementary views per table (BIRDIE dual-embedding design):
    #   - title+schema (coarse) drives shallow docid prefix layers (emd1)
    #   - full table content (fine) drives deep docid layers (emd2)
    title_schema_texts = []
    data_texts = []
    table_ids = []

    for table_id, table in tqdm(tables.items(), desc="Preparing tables"):
        title_schema_texts.append(title_schema_text(table))
        data_texts.append(table2text(table))
        table_ids.append(table_id)

    # Generate both embedding views in batches
    print("Generating title+schema embeddings (emd1)...")
    title_schema_emb = np.array(model.encode(title_schema_texts, batch_size=32))
    print("Generating table-data embeddings (emd2)...")
    data_emb = np.array(model.encode(data_texts, batch_size=32))

    # Save both embedding views
    ts_path = os.path.join(output_dir, 'table_title_schema_embedding.npy')
    data_path = os.path.join(output_dir, 'table_data_embedding.npy')
    np.save(ts_path, title_schema_emb)
    np.save(data_path, data_emb)
    print(f"Title+schema embeddings saved to {ts_path}, shape: {title_schema_emb.shape}")
    print(f"Table-data embeddings saved to {data_path}, shape: {data_emb.shape}")

    # Save ID mapping (order matches both embedding arrays)
    id_path = os.path.join(output_dir, 'table_ids.json')
    with open(id_path, 'w') as f:
        json.dump(table_ids, f)
    print(f"Table IDs saved to {id_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate table embeddings")
    parser.add_argument("--table_data_path", required=True, type=str,
                        help="Path to table_data.json")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for embeddings")
    parser.add_argument("--model_name", default=DEFAULT_EMBEDDING_MODEL, type=str,
                       help="Embedding model path. Default: BGE-M3 (aligned with Pneuma)")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for embedding generation")

    args = parser.parse_args()

    generate_embeddings(args.table_data_path, args.output_dir, args.model_name)
