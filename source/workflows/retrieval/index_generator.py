"""
Retrieval Index Generator

Generates FAISS vector index and BM25 keyword index for semantic search.
This module is used by Stage 5 of the UPO pipeline.

Usage:
    from workflows.retrieval.index_generator import generate_retrieval_index
    
    result = generate_retrieval_index(
        dataset_name='fetaqa',
        index_key='td_cd_cs',  # or 'td', 'td_cd'
        enable_faiss=True,
        enable_bm25=True,
        batch_size=256,
    )
    
Index Keys:
    - td: table_description only
    - td_cd: table_description + column_descriptions
    - td_cd_cs: all three fields (default, most comprehensive)
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger


def generate_retrieval_index(
    dataset_name: str,
    index_key: Optional[str] = None,
    enable_faiss: bool = True,
    enable_bm25: bool = True,
    batch_size: int = 256,
    remove_primitive_classes: bool = False,
    output_base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate FAISS vector index and BM25 keyword index for a dataset.
    
    Reads from: {dataset}_table_summaries_retrieval LanceDB table.
    Outputs: data/lake/indexes/{dataset}/{index_key}/ with FAISS and BM25 indexes.
             Or {output_base_path}/indexes/{index_key}/ if output_base_path is provided.
    
    Args:
        dataset_name: Dataset name (e.g., 'fetaqa')
        index_key: Index configuration key ('td', 'td_cd', 'td_cd_cs')
        enable_faiss: Generate FAISS vector index
        enable_bm25: Generate BM25 keyword index
        batch_size: Batch size for embedding generation
        remove_primitive_classes: If True, remove [Type] markers from documents
            (ablation mode). Indexes will be saved to {index_key}_no_pc/
        output_base_path: Base path for output. If provided, indexes are saved to
            {output_base_path}/indexes/{index_key}/
    
    Returns:
        Dict with:
            - success: bool
            - total_tables: int
            - index_key: str
            - faiss_path: str (if enable_faiss)
            - bm25_path: str (if enable_bm25)
            - elapsed: float
            - error: str (if failed)
    """
    # Import here to avoid circular imports and allow lazy loading
    from workflows.retrieval.nodes.semantic_search import generate_index
    
    start_time = time.time()
    
    logger.info(f"Generating retrieval indexes for {dataset_name}")
    logger.info(f"  Index Key: {index_key or 'td_cd_cs (default)'}")
    logger.info(f"  FAISS: {enable_faiss}, BM25: {enable_bm25}")
    logger.info(f"  Batch size: {batch_size}")
    if remove_primitive_classes:
        logger.info(f"  Ablation Mode: Removing primitive class markers")
    if output_base_path:
        logger.info(f"  Output Base Path: {output_base_path}")
    
    try:
        # Call the existing generate_index function with index_key
        result = generate_index(
            dataset_name=dataset_name,
            index_key=index_key,
            batch_size=batch_size,
            enable_faiss=enable_faiss,
            enable_bm25=enable_bm25,
            remove_primitive_classes=remove_primitive_classes,
            output_base_path=output_base_path,
        )
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'total_tables': result.get('num_tables', 0),
            'index_key': result.get('index_key', index_key),
            'faiss_path': result.get('faiss_path'),
            'bm25_path': result.get('bm25_path'),
            'elapsed': elapsed,
            'remove_primitive_classes': remove_primitive_classes,
        }
        
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        logger.error(f"Index generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed': elapsed,
        }
