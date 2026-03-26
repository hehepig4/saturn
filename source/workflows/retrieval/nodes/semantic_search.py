"""
Node: Semantic Search - Vector + BM25 Hybrid Search

Uses pre-generated FAISS and BM25 indexes for efficient retrieval.
Indexes are generated via `demos/retrieval.py --generate-index`.

Index Storage:
  data/lake/indexes/{dataset}/
  ├── faiss/
  │   ├── index.faiss       # FAISS vector index
  │   └── metadata.pkl      # table metadata
  └── bm25/
      ├── index/            # bm25s index directory
      └── table_ids.pkl     # ID mapping
"""

import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import faiss
import bm25s
import Stemmer
import numpy as np
from loguru import logger
from tqdm import tqdm

from config.hyperparameters import HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT
from workflows.common.node_decorators import graph_node
from workflows.retrieval.state import RetrievalResult, ColumnTypeMatch
from store.embedding.embedding_registry import get_registry
from utils.primitive_class_utils import remove_primitive_class_markers


# ==================== Constants ====================

def get_text_embedder():
    """Get BGE-M3 text embedding function from registry (lazy load)."""
    registry = get_registry()
    # Only register bge-m3 on demand (not all functions)
    return registry.register_function("bge-m3")

EMBEDDING_DIM = 1024


def get_index_path(dataset_name: str, index_key: str = "td_cd_cs", base_path: Path = None) -> Path:
    """
    Get base path for dataset indexes.
    
    Args:
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        index_key: Index configuration key:
            - 'td': table_description only
            - 'td_cd': table_description + column_descriptions
            - 'td_cd_cs': all three fields (default)
        base_path: Base path for indexes. If provided, returns base_path/indexes/{dataset}/{index_key}/
            Otherwise returns default: data/lake/indexes/{dataset}/{index_key}/
            
    Returns:
        Path to index directory
    """
    if base_path:
        # Use custom base path (e.g., experiment variant directory)
        return base_path / "indexes" / dataset_name / index_key
    
    # Default: data/lake/indexes/{dataset}/{index_key}/
    source_dir = Path(__file__).parent.parent.parent.parent
    base_path_default = source_dir.parent / "data" / "lake" / "indexes" / dataset_name
    
    # ALL index keys use subdirectory for consistency
    return base_path_default / index_key


# ==================== Index Key Mapping ====================

# Map index_key to search fields
INDEX_FIELD_MAPPING = {
    'td': ['table_description'],
    'td_cd': ['table_description', 'column_descriptions'],
    'td_cd_cs': ['table_description', 'column_descriptions', 'column_stats'],
}

# Default searchable fields (all three fields)
DEFAULT_SEARCH_FIELDS = ['table_description', 'column_descriptions', 'column_stats']
DEFAULT_INDEX_KEY = 'td_cd_cs'


def _merge_column_desc_and_stats(col_descriptions: str, col_stats: str) -> str:
    """
    Merge column descriptions and stats into a unified per-column format.
    
    Input formats (from generate_summaries.py, ' || ' separated):
        col_descriptions: "[Type1] Col1: Desc1 || [Type2] Col2: Desc2 || ..."
        col_stats: "Col1: stats1 || Col2: stats2 || ..."
    
    Output format (one line per column, desc and stats concatenated with space):
        "[Type1] Col1: Desc1 stats1"
        "[Type2] Col2: Desc2 stats2"
    
    Returns:
        Merged column information with each column on its own line
    """
    # Separator used in stored data (from generate_summaries.py)
    SEP = ' || '
    
    if not col_descriptions and not col_stats:
        return ""
    
    if not col_stats:
        # No stats, convert || to newlines for readability
        return '\n'.join(part.strip() for part in col_descriptions.split(SEP) if part.strip())
    
    if not col_descriptions:
        # No descriptions, format stats with prefix
        lines = []
        for part in col_stats.split(SEP):
            part = part.strip()
            if part:
                lines.append(f"Stats: {part}")
        return '\n'.join(lines)
    
    # Build a mapping from column name to stats
    stats_map = {}
    for part in col_stats.split(SEP):
        part = part.strip()
        if not part or ':' not in part:
            continue
        # Format: "ColName: stats_content"
        col_name, _, stats_content = part.partition(':')
        col_name = col_name.strip()
        stats_content = stats_content.strip()
        if col_name and stats_content:
            stats_map[col_name] = stats_content
    
    # Merge descriptions with stats (space-concatenated, no || separator)
    merged_lines = []
    for part in col_descriptions.split(SEP):
        part = part.strip()
        if not part:
            continue
        
        # Try to extract column name from description
        # Format: "[Type] ColName: Description" or "ColName: Description"
        col_name = None
        if '] ' in part:
            # Format: "[Type] ColName: Description"
            after_bracket = part.split('] ', 1)[1] if '] ' in part else part
            if ':' in after_bracket:
                col_name = after_bracket.split(':')[0].strip()
        elif ':' in part:
            col_name = part.split(':')[0].strip()
        
        # Append stats if available for this column (space-concatenated)
        if col_name and col_name in stats_map:
            merged_lines.append(f"{part} {stats_map[col_name]}")
        else:
            merged_lines.append(part)
    
    return '\n'.join(merged_lines)


def _build_index_document(
    row,
    search_fields: list,
    remove_primitive_classes: bool = False,
) -> str:
    """
    Build index document text with structured format.
    
    Format:
        == Table Description ==
        {table_description}
        
        == Column Information ==
        [Type1] Col1: Desc1 stats1
        [Type2] Col2: Desc2 stats2
        ...
    
    Args:
        row: DataFrame row with table data
        search_fields: List of fields to include in index
        remove_primitive_classes: If True, remove [Type] markers from column info
        
    Returns:
        Formatted document text for embedding
    """
    parts = []
    
    # Table description section
    if 'table_description' in search_fields:
        table_desc = row.get('table_description', '')
        if table_desc:
            parts.append("== Table Description ==")
            parts.append(str(table_desc))
    
    # Column information section
    include_col_desc = 'column_descriptions' in search_fields
    include_col_stats = 'column_stats' in search_fields
    
    if include_col_desc or include_col_stats:
        col_desc = row.get('column_descriptions', '') or row.get('column_narrations', '')
        col_stats = row.get('column_stats', '') if include_col_stats else ''
        
        # Merge descriptions and stats per column
        if include_col_desc:
            col_info = _merge_column_desc_and_stats(col_desc, col_stats)
        else:
            # Only stats, no descriptions
            col_info = col_stats
        
        # Remove primitive class markers if ablation mode
        if remove_primitive_classes and col_info:
            col_info = remove_primitive_class_markers(col_info)
        
        if col_info:
            parts.append("\n== Column Information ==")
            parts.append(col_info)
    
    return '\n'.join(parts).strip()


def generate_index(
    dataset_name: str,
    index_key: str = None,
    batch_size: int = 256,
    enable_faiss: bool = True,
    enable_bm25: bool = True,
    remove_primitive_classes: bool = False,
    output_base_path: Path = None,
) -> Dict[str, Any]:
    """
    Generate FAISS vector index and BM25 index from LanceDB table summaries.
    
    Args:
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        index_key: Index configuration key ('td', 'td_cd', 'td_cd_cs')
            Defaults to 'td_cd_cs' if not provided
        batch_size: Batch size for embedding generation
        enable_faiss: Whether to generate FAISS vector index
        enable_bm25: Whether to generate BM25 keyword index
        remove_primitive_classes: If True, remove [Type] markers from documents
            (ablation mode). Indexes will be saved to {index_key}_no_pc/
        output_base_path: Base path for index output. If provided, indexes are saved to
            {output_base_path}/indexes/{index_key}/. Otherwise uses default path.
        
    Returns:
        Dict with generation statistics
    """
    from store.store_singleton import get_store
    
    # Determine search fields from index_key
    if index_key is None:
        index_key = DEFAULT_INDEX_KEY
    
    # Validate base index_key (strip _no_pc suffix if present)
    base_index_key = index_key.replace('_no_pc', '')
    if base_index_key not in INDEX_FIELD_MAPPING:
        raise ValueError(f"Unknown index_key: {base_index_key}. Valid keys: {list(INDEX_FIELD_MAPPING.keys())}")
    search_fields = INDEX_FIELD_MAPPING[base_index_key]
    
    # Add _no_pc suffix for ablation mode
    effective_index_key = f"{base_index_key}_no_pc" if remove_primitive_classes else base_index_key
    
    logger.info("=" * 60)
    logger.info(f"Generating Indexes for {dataset_name}")
    logger.info("=" * 60)
    logger.info(f"  Index Key: {effective_index_key}")
    logger.info(f"  Search Fields: {search_fields}")
    if remove_primitive_classes:
        logger.info(f"  Ablation Mode: Removing primitive class markers [Type]")
    
    start_time = time.time()
    
    # Load table summaries from LanceDB
    logger.info("  Connecting to LanceDB...")
    store = get_store()
    source_table = f"{dataset_name}_table_summaries_retrieval"
    
    logger.info(f"  Opening table: {source_table}")
    try:
        tbl = store.db.open_table(source_table)
    except Exception:
        source_table = f"{dataset_name}_table_summaries"
        logger.info(f"  Fallback to: {source_table}")
        tbl = store.db.open_table(source_table)
    
    logger.info("  Loading table summaries to pandas...")
    load_start = time.time()
    df = tbl.to_pandas()
    logger.info(f"  ✓ Loaded {len(df)} table summaries in {time.time() - load_start:.1f}s")
    
    # Prepare documents
    logger.info("  Preparing documents for indexing...")
    documents = []
    metadata_list = []
    
    for _, row in df.iterrows():
        # Build structured document with unified format
        text = _build_index_document(row, search_fields, remove_primitive_classes=remove_primitive_classes)
        
        if not text:
            text = str(row.get('table_id', 'unknown'))
        
        documents.append(text)
        
        # Store metadata (support both old and new field names)
        column_types = row.get('column_types', [])
        if isinstance(column_types, str):
            try:
                import json
                column_types = json.loads(column_types)
            except:
                column_types = []
        
        # Support both old (column_narrations) and new (column_descriptions) field names
        col_desc = row.get('column_descriptions', '') or row.get('column_narrations', '')
        col_stats = row.get('column_stats', '')
        
        metadata_list.append({
            'table_id': row.get('table_id', ''),
            'table_description': row.get('table_description', ''),
            'column_descriptions': col_desc,
            'column_stats': col_stats,
            'column_types': column_types,
            'has_fallback_only': row.get('has_fallback_only', False),
            'num_cols': row.get('num_cols', 0),
        })
    
    index_path = get_index_path(dataset_name, effective_index_key, base_path=output_base_path)
    faiss_result_path = None
    bm25_result_path = None
    
    # Generate Vector Index (FAISS)
    if enable_faiss:
        logger.info("\n[1/2] Generating FAISS Vector Index...")
        faiss_path = index_path / "faiss"
        faiss_path.mkdir(parents=True, exist_ok=True)
        
        # Load embedder from registry (handles GPU, model path, etc.)
        logger.info("  Loading BGE-M3 embedder...")
        embed_load_start = time.time()
        embedder = get_text_embedder()
        logger.info(f"  ✓ Embedder loaded in {time.time() - embed_load_start:.1f}s (dim={EMBEDDING_DIM})")
        
        # Generate embeddings - let SentenceTransformers handle batching internally
        logger.info(f"  Generating embeddings for {len(documents)} documents...")
        logger.info(f"  Internal batch size: {batch_size}")
        
        # Update embedder batch_size and generate all at once
        embedder.batch_size = batch_size
        
        start_embed_time = time.time()
        all_embeddings = embedder.compute_source_embeddings(documents)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"  ✓ Embeddings generated: shape={embeddings.shape}, time={time.time() - start_embed_time:.1f}s")
        
        # Create FAISS index (Inner Product for normalized vectors = cosine)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        
        faiss.write_index(index, str(faiss_path / "index.faiss"))
        with open(faiss_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata_list, f)
        
        logger.info(f"  ✓ FAISS index saved: {faiss_path}")
        faiss_result_path = str(faiss_path)
    else:
        logger.info("\n[1/2] Skipping FAISS Vector Index (disabled)")
    
    # Generate BM25 Index
    if enable_bm25:
        logger.info("\n[2/2] Generating BM25 Index...")
        bm25_path = index_path / "bm25"
        bm25_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("  Tokenizing documents for BM25...")
        bm25_start = time.time()
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer, show_progress=False)
        logger.info(f"  ✓ Tokenized in {time.time() - bm25_start:.1f}s")
        
        logger.info("  Building BM25 index...")
        bm25_index_start = time.time()
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        logger.info(f"  ✓ BM25 indexed in {time.time() - bm25_index_start:.1f}s")
        
        logger.info("  Saving BM25 index to disk...")
        retriever.save(str(bm25_path / "index"), corpus=documents)
        
        table_ids = [m['table_id'] for m in metadata_list]
        with open(bm25_path / "table_ids.pkl", 'wb') as f:
            pickle.dump(table_ids, f)
        
        logger.info(f"  ✓ BM25 index saved: {bm25_path}")
        bm25_result_path = str(bm25_path)
    else:
        logger.info("\n[2/2] Skipping BM25 Index (disabled)")
    
    elapsed = time.time() - start_time
    logger.info(f"\n✅ Index generation complete in {elapsed:.1f}s")
    
    return {
        'success': True,
        'dataset': dataset_name,
        'index_key': effective_index_key,
        'search_fields': search_fields,
        'num_tables': len(documents),
        'faiss_path': faiss_result_path,
        'bm25_path': bm25_result_path,
        'elapsed': elapsed,
        'remove_primitive_classes': remove_primitive_classes,
    }


# ==================== Search Functions ====================

def vector_search(
    query_embedding: np.ndarray,
    faiss_index: faiss.Index,
    metadata_list: List[Dict],
    top_k: int,
) -> List[Tuple[str, float, Dict]]:
    """
    Search FAISS index with query embedding.
    
    Returns:
        List of (table_id, score, metadata) tuples, sorted by score descending
    """
    query_vec = np.array([query_embedding], dtype=np.float32)
    scores, indices = faiss_index.search(query_vec, min(top_k, faiss_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        meta = metadata_list[idx]
        results.append((meta['table_id'], float(score), meta))
    
    return results


def bm25_search(
    query: str,
    bm25_retriever: bm25s.BM25,
    table_ids: List[str],
    metadata_list: List[Dict],
    top_k: int,
) -> List[Tuple[str, float, Dict]]:
    """
    Search BM25 index with query text.
    
    Returns:
        List of (table_id, score, metadata) tuples, sorted by score descending
    """
    stemmer = Stemmer.Stemmer("english")
    query_tokens = bm25s.tokenize([query], stemmer=stemmer, show_progress=False)
    
    # Retrieve results (returns Results object with .documents and .scores)
    results_obj = bm25_retriever.retrieve(
        query_tokens, k=min(top_k, len(table_ids)), show_progress=False
    )
    
    # When loaded without corpus, documents contains integer indices directly
    doc_indices = results_obj.documents[0] if len(results_obj.documents) > 0 else []
    scores = results_obj.scores[0] if len(results_obj.scores) > 0 else []
    
    results = []
    for idx, score in zip(doc_indices, scores):
        if score <= 0:
            continue
        
        real_idx = int(idx)
        if 0 <= real_idx < len(metadata_list):
            meta = metadata_list[real_idx]
            results.append((meta['table_id'], float(score), meta))
    
    return results


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float, Dict]],
    bm25_results: List[Tuple[str, float, Dict]],
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    k: int = 60,
) -> List[Tuple[str, float, Dict]]:
    """
    Combine vector and BM25 results using RRF.
    
    RRF formula: score(d) = sum(weight / (k + rank(d)))
    """
    # Build rank maps
    vector_ranks = {tid: rank + 1 for rank, (tid, _, _) in enumerate(vector_results)}
    bm25_ranks = {tid: rank + 1 for rank, (tid, _, _) in enumerate(bm25_results)}
    
    # Build metadata map
    meta_map = {}
    for tid, _, meta in vector_results:
        meta_map[tid] = meta
    for tid, _, meta in bm25_results:
        if tid not in meta_map:
            meta_map[tid] = meta
    
    # Compute RRF scores
    all_tables = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    
    fused = []
    for tid in all_tables:
        score = 0.0
        if tid in vector_ranks:
            score += vector_weight / (k + vector_ranks[tid])
        if tid in bm25_ranks:
            score += bm25_weight / (k + bm25_ranks[tid])
        fused.append((tid, score, meta_map.get(tid, {})))
    
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ==================== Main Node ====================

@graph_node(on_error="continue")
def semantic_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Vector + BM25 hybrid search using pre-generated indexes.
    
    Args:
        state: Workflow state with query, dataset_name, etc.
            - index_key: Optional index configuration ('td', 'td_cd', 'td_cd_cs')
            - rag_type: Optional retrieval type ('bm25', 'vector', 'hybrid')
        
    Returns:
        Updated state with semantic_results
    """
    if not state.enable_semantic:
        logger.info("Semantic search disabled")
        return {'semantic_results': []}
    
    query = state.query
    dataset_name = state.dataset_name
    top_k = state.semantic_top_k
    
    # Index configuration
    index_key = getattr(state, 'index_key', None)
    
    # RAG type configuration
    rag_type = getattr(state, 'rag_type', 'hybrid')
    
    logger.info(f"  Query: {query[:80]}..." if len(query) > 80 else f"  Query: {query}")
    logger.info(f"  Dataset: {dataset_name}, Top-K: {top_k}, Index: {index_key or 'default'}")
    logger.info(f"  RAG Type: {rag_type}")
    
    # Try unified search first
    try:
        from workflows.retrieval.unified_search import unified_search
        from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
        
        idx_key = index_key or INDEX_KEY_TD_CD_CS
        results = unified_search(
            query=query,
            dataset_name=dataset_name,
            top_k=top_k * 2,  # Over-fetch for fusion
            rag_type=rag_type,
            index_key=idx_key,
        )
        
        # Convert to RetrievalResult
        semantic_results = []
        for table_id, score, meta in results[:top_k]:
            column_types = meta.get('column_types', []) if meta else []
            
            tbox_matches = []
            for ct in column_types:
                clean_ct = ct.replace('upo:', '')
                is_fallback = clean_ct.lower() == 'column'
                tbox_matches.append(ColumnTypeMatch(
                    column_type=ct,
                    confidence=0.3 if is_fallback else 1.0,
                    is_fallback=is_fallback,
                ))
            
            result = RetrievalResult(
                table_id=table_id,
                semantic_score=score,
                source=rag_type,
                tbox_matches=tbox_matches,
                has_fallback_only=meta.get('has_fallback_only', False) if meta else False,
                metadata={
                    'table_description': meta.get('table_description', '') if meta else '',
                    'column_descriptions': meta.get('column_descriptions', '') if meta else '',
                    'column_stats': meta.get('column_stats', '') if meta else '',
                    'relationship_view': meta.get('relationship_view', '') if meta else '',
                }
            )
            semantic_results.append(result)
        
        logger.info(f"\n  ✅ Unified search complete: {len(semantic_results)} results")
        return {'semantic_results': semantic_results}
        
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        raise RuntimeError(f'Unified search failed for {dataset_name}: {e}')
