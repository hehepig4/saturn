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
    
    # Generate table texts
    table_texts = []
    table_ids = []
    
    for table_id, table in tqdm(tables.items(), desc="Preparing tables"):
        # Get document title
        title = table.get('documentTitle', '')
        
        # Convert table to markdown
        md_table = json_to_markdown(table)
        
        # Combine title and table
        text = f"Table: {title}\n{md_table}"
        table_texts.append(text)
        table_ids.append(table_id)
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = model.encode(table_texts, batch_size=32)
    embeddings = np.array(embeddings)
    
    # Save embeddings
    emb_path = os.path.join(output_dir, 'table_embeddings.npy')
    np.save(emb_path, embeddings)
    print(f"Embeddings saved to {emb_path}")
    print(f"Shape: {embeddings.shape}")
    
    # Save ID mapping
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
