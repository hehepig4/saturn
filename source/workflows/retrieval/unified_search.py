"""
Unified Search Interface

Provides a single entry point for all retrieval operations across the pipeline.
Supports BM25, Vector, and Hybrid search with configurable RAG types.

Also provides unified query embedding management for Stage 1 and beyond.

Usage:
    from workflows.retrieval.unified_search import unified_search, UnifiedSearchResult
    
    # Search using hybrid (default)
    results = unified_search(
        query="What is the population of New York?",
        dataset="fetaqa",
        top_k=10,
        rag_type="hybrid",
    )
    
    # Results are list of (table_id, score, metadata) tuples
    for table_id, score, meta in results:
        print(f"{table_id}: {score:.4f}")
    
    # Load query embeddings for clustering/sampling
    from workflows.retrieval.unified_search import load_query_embeddings
    
    embeddings, query_ids, _ = load_query_embeddings("fetaqa", split="train")
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from loguru import logger

from config.hyperparameters import HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT
from workflows.retrieval.config import (
    RAG_TYPE_BM25,
    RAG_TYPE_VECTOR,
    RAG_TYPE_HYBRID,
    RAG_TYPE_HYBRID_SUM,
    get_search_config,
    validate_rag_type,
    validate_index_key,
    INDEX_KEY_RAW,
    DEFAULT_INDEX_KEY,
)


# ==================== Index Paths ====================

def get_index_base_path() -> Path:
    """Get base path for all indexes."""
    path = os.getenv('SATURN_INDEX_PATH')
    if path:
        p = Path(path)
        if not p.is_absolute():
            from core.paths import get_project_root
            p = get_project_root() / p
        return p / "indexes"
    source_dir = Path(__file__).parent.parent.parent
    return source_dir.parent / "data" / "lake" / "indexes"


def get_index_path(dataset_name: str, index_key: Optional[str] = None, base_path: Optional[Path] = None) -> Path:
    """
    Get path for dataset indexes.
    
    Args:
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        index_key: Index configuration key:
            - 'raw': Raw table_text index from ingest (for Stage 1)
            - 'td': table_description only
            - 'td_cd': table_description + column_descriptions
            - 'td_cd_cs': all three fields (default)
        base_path: Base path for indexes. If provided, returns {base_path}/indexes/{dataset}/{index_key}/
            Otherwise returns default: data/lake/indexes/{dataset}/{index_key}/
            
    Returns:
        Path to index directory
    """
    index_key = validate_index_key(index_key)
    if base_path:
        return base_path / "indexes" / dataset_name / index_key
    return get_index_base_path() / dataset_name / index_key


# ==================== Index Loading ====================

# Global index cache to avoid repeated loading
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def load_unified_indexes(
    dataset_name: str,
    index_key: Optional[str] = None,
    force_reload: bool = False,
    index_base_path: Optional[Path] = None,
) -> Tuple[Any, List[Dict], Any, List[str]]:
    """
    Load FAISS and BM25 indexes for a dataset (unified interface).
    
    Uses caching to avoid repeated loading.
    
    Args:
        dataset_name: Dataset identifier
        index_key: Index configuration key (default: td_cd_cs)
        force_reload: Force reload even if cached
        index_base_path: Base path for indexes. If provided, loads from
            {index_base_path}/indexes/{index_key}/
        
    Returns:
        Tuple of (faiss_index, metadata_list, bm25_retriever, table_ids)
        Any component may be None if not available.
    """
    import faiss
    import bm25s
    
    index_key = validate_index_key(index_key)
    
    # Include base_path in cache key for isolation
    base_path_str = str(index_base_path) if index_base_path else "default"
    cache_key = f"{dataset_name}:{index_key}:{base_path_str}"
    
    # Check cache
    if not force_reload and cache_key in _INDEX_CACHE:
        cached = _INDEX_CACHE[cache_key]
        return (
            cached.get('faiss_index'),
            cached.get('metadata_list', []),
            cached.get('bm25_retriever'),
            cached.get('table_ids', []),
        )
    
    index_path = get_index_path(dataset_name, index_key, base_path=index_base_path)
    logger.debug(f"Loading indexes from: {index_path}")
    
    faiss_index = None
    metadata_list = []
    bm25_retriever = None
    table_ids = []
    
    # Load FAISS index
    faiss_path = index_path / "faiss"
    if faiss_path.exists():
        try:
            index_file = faiss_path / "index.faiss"
            metadata_file = faiss_path / "metadata.pkl"
            
            if index_file.exists():
                faiss_index = faiss.read_index(str(index_file))
                logger.debug(f"  Loaded FAISS index: {faiss_index.ntotal} vectors")
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata_list = pickle.load(f)
                logger.debug(f"  Loaded metadata: {len(metadata_list)} entries")
        except Exception as e:
            logger.warning(f"  Failed to load FAISS index: {e}")
    
    # Load BM25 index
    bm25_path = index_path / "bm25"
    if bm25_path.exists():
        try:
            bm25_index_path = bm25_path / "index"
            table_ids_file = bm25_path / "table_ids.pkl"
            
            if bm25_index_path.exists():
                # Load WITHOUT corpus to avoid index error in retrieve()
                bm25_retriever = bm25s.BM25.load(str(bm25_index_path), load_corpus=False)
                logger.debug(f"  Loaded BM25 index")
            
            if table_ids_file.exists():
                with open(table_ids_file, 'rb') as f:
                    table_ids = pickle.load(f)
                logger.debug(f"  Loaded table IDs: {len(table_ids)} entries")
        except Exception as e:
            logger.warning(f"  Failed to load BM25 index: {e}")
    
    # Cache results
    _INDEX_CACHE[cache_key] = {
        'faiss_index': faiss_index,
        'metadata_list': metadata_list,
        'bm25_retriever': bm25_retriever,
        'table_ids': table_ids,
    }
    
    return faiss_index, metadata_list, bm25_retriever, table_ids


# ==================== Embedding ====================

_TEXT_EMBEDDER = None


def get_text_embedder():
    """Get BGE-M3 text embedding function from registry (lazy load)."""
    global _TEXT_EMBEDDER
    if _TEXT_EMBEDDER is None:
        from store.embedding.embedding_registry import get_registry
        registry = get_registry()
        _TEXT_EMBEDDER = registry.register_function("bge-m3")
    return _TEXT_EMBEDDER


def embed_query(query: str) -> np.ndarray:
    """
    Embed a query using BGE-M3.
    
    Args:
        query: Query text
        
    Returns:
        Query embedding as numpy array
    """
    embedder = get_text_embedder()
    query_emb_list = embedder.compute_query_embeddings(query)
    return np.array(query_emb_list[0], dtype=np.float32)


# ==================== Search Functions ====================

def vector_search(
    query_embedding: np.ndarray,
    faiss_index,
    metadata_list: List[Dict],
    top_k: int,
) -> List[Tuple[str, float, Dict]]:
    """
    Perform vector similarity search using FAISS.
    
    Args:
        query_embedding: Query vector
        faiss_index: FAISS index
        metadata_list: List of metadata dicts
        top_k: Number of results
        
    Returns:
        List of (table_id, score, metadata) tuples
    """
    if faiss_index is None or not metadata_list:
        return []
    
    # Reshape for FAISS
    query_vec = query_embedding.reshape(1, -1)
    
    # Search
    distances, indices = faiss_index.search(query_vec, min(top_k, faiss_index.ntotal))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        
        meta = metadata_list[idx]
        table_id = meta.get('table_id', f'unknown_{idx}')
        
        # Convert L2 distance to cosine similarity (valid for normalized embeddings)
        # For unit vectors: ||a-b||^2 = 2 - 2*cos(a,b), so cos_sim = 1 - dist/2
        # FAISS returns squared L2 distances, so dist is already ||a-b||^2
        score = 1.0 - dist / 2.0
        
        results.append((table_id, score, meta))
    
    return results


def bm25_search(
    query: str,
    bm25_retriever,
    table_ids: List[str],
    metadata_list: Optional[List[Dict]] = None,
    top_k: int = 10,
) -> List[Tuple[str, float, Dict]]:
    """
    Perform BM25 keyword search.
    
    Args:
        query: Query text
        bm25_retriever: BM25 retriever
        table_ids: List of table IDs (aligned with BM25 index)
        metadata_list: Optional list of metadata dicts
        top_k: Number of results
        
    Returns:
        List of (table_id, score, metadata) tuples
    """
    import bm25s
    import Stemmer
    
    if bm25_retriever is None or not table_ids:
        return []
    
    # Build metadata lookup if provided
    meta_lookup = {}
    if metadata_list:
        for meta in metadata_list:
            tid = meta.get('table_id', '')
            if tid:
                meta_lookup[tid] = meta
    
    # Tokenize query (no progress bar)
    stemmer = Stemmer.Stemmer("english")
    query_tokens = bm25s.tokenize([query], stemmer=stemmer, show_progress=False)
    
    # Search - disable progress bar
    results_obj = bm25_retriever.retrieve(
        query_tokens, 
        k=min(top_k, len(table_ids)),
        show_progress=False,
    )
    
    # Extract scores and document indices
    # When loaded without corpus, documents contains integer indices directly
    scores = results_obj.scores[0] if len(results_obj.scores) > 0 else []
    doc_indices = results_obj.documents[0] if len(results_obj.documents) > 0 else []
    
    results = []
    for i, score in enumerate(scores):
        if score <= 0:
            continue
        
        # Get document index
        doc_idx = int(doc_indices[i]) if i < len(doc_indices) else i
        
        # Get table_id from table_ids list
        if doc_idx < len(table_ids):
            table_id = table_ids[doc_idx]
        else:
            continue
        
        meta = meta_lookup.get(table_id, {'table_id': table_id})
        results.append((table_id, float(score), meta))
    
    return results


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float, Dict]],
    bm25_results: List[Tuple[str, float, Dict]],
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    k: int = 60,
) -> List[Tuple[str, float, Dict]]:
    """
    Fuse vector and BM25 results using Reciprocal Rank Fusion.
    
    Args:
        vector_results: Vector search results
        bm25_results: BM25 search results
        vector_weight: Weight for vector results
        bm25_weight: Weight for BM25 results
        k: RRF parameter
        
    Returns:
        Fused results sorted by score
    """
    # Build rank maps
    vector_ranks = {tid: rank for rank, (tid, _, _) in enumerate(vector_results, 1)}
    bm25_ranks = {tid: rank for rank, (tid, _, _) in enumerate(bm25_results, 1)}
    
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


def normalized_score_fusion(
    vector_results: List[Tuple[str, float, Dict]],
    bm25_results: List[Tuple[str, float, Dict]],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    # Extended params for score completion (optional)
    query: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    faiss_index=None,
    bm25_retriever=None,
    table_ids: Optional[List[str]] = None,
    metadata_list: Optional[List[Dict]] = None,
    # Normalization strategy
    normalize_vector: bool = False,  # Whether to min-max normalize vector scores
) -> List[Tuple[str, float, Dict]]:
    """
    Fuse vector and BM25 results using normalized score summation.

    By default, only BM25 scores are min-max normalized to [0, 1], while vector
    scores (already in [0, 1] via cosine similarity) are kept as-is. This works
    best when vector score distribution is narrow (curse of dimensionality).

    Optionally, if normalize_vector=True (Pneuma-style), both sides are 
    min-max normalized for equal contribution.

    If index objects are provided, missing scores are computed rather than 
    assumed to be 0.

    Args:
        vector_results: Vector search results
        bm25_results: BM25 search results with raw scores
        vector_weight: Weight for vector component (default 0.5)
        bm25_weight: Weight for BM25 component (default 0.5)
        query: Original query text (for BM25 score lookup)
        query_embedding: Query embedding (for vector score lookup)
        faiss_index: FAISS index for vector score lookup
        bm25_retriever: BM25 retriever for score lookup
        table_ids: List of table IDs (aligned with BM25 index)
        metadata_list: Metadata list (aligned with FAISS index)
        normalize_vector: Whether to min-max normalize vector scores (default False)

    Returns:
        Fused results sorted by combined score descending
    """
    import bm25s
    import Stemmer
    
    # Extract raw score maps
    vec_scores = {tid: score for tid, score, _ in vector_results}
    bm25_scores = {tid: score for tid, score, _ in bm25_results}
    
    # Build metadata map (prefer vector metadata)
    meta_map = {}
    for tid, _, meta in bm25_results:
        meta_map[tid] = meta
    for tid, _, meta in vector_results:
        meta_map[tid] = meta
    
    # === Optional: Complete missing scores (Pneuma-style) ===
    vec_ids = set(vec_scores.keys())
    bm25_ids = set(bm25_scores.keys())
    vec_only = vec_ids - bm25_ids
    bm25_only = bm25_ids - vec_ids
    
    # Complete missing BM25 scores for vec_only IDs
    if vec_only and bm25_retriever is not None and table_ids and query:
        tid_to_bm25_idx = {tid: idx for idx, tid in enumerate(table_ids)}
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize([query], stemmer=stemmer, show_progress=False)
        from bm25s.tokenization import convert_tokenized_to_string_list
        query_tokens_str = convert_tokenized_to_string_list(query_tokens)[0]
        all_bm25_scores = bm25_retriever.get_scores(query_tokens_str)
        for tid in vec_only:
            if tid in tid_to_bm25_idx:
                idx = tid_to_bm25_idx[tid]
                bm25_scores[tid] = float(all_bm25_scores[idx])
    
    # Complete missing Vector scores for bm25_only IDs
    if bm25_only and faiss_index is not None and metadata_list and query_embedding is not None:
        tid_to_faiss_idx = {meta.get('table_id', ''): idx for idx, meta in enumerate(metadata_list)}
        missing_indices = []
        missing_tids = []
        for tid in bm25_only:
            if tid in tid_to_faiss_idx:
                missing_indices.append(tid_to_faiss_idx[tid])
                missing_tids.append(tid)
        if missing_indices:
            try:
                missing_embeddings = np.array([
                    faiss_index.reconstruct(idx) for idx in missing_indices
                ])
                query_vec = query_embedding.reshape(1, -1)
                dists = np.sum((missing_embeddings - query_vec) ** 2, axis=1)
                for tid, dist in zip(missing_tids, dists):
                    vec_scores[tid] = 1.0 - dist / 2.0
            except Exception:
                pass
    
    # === Normalization ===
    def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mn == mx:
            return {tid: 1.0 for tid in scores}
        rng = mx - mn
        return {tid: (s - mn) / rng for tid, s in scores.items()}
    
    # Always normalize BM25
    bm25_norm = min_max_normalize(bm25_scores)
    
    # Optionally normalize vector (Pneuma-style) or keep as-is (default)
    if normalize_vector:
        vec_norm = min_max_normalize(vec_scores)
    else:
        vec_norm = vec_scores  # Keep original [0, 1] scores
    
    # === Combine with weighted sum ===
    all_tables = set(vec_norm.keys()) | set(bm25_norm.keys())
    fused = []
    for tid in all_tables:
        score = (
            vector_weight * vec_norm.get(tid, 0.0)
            + bm25_weight * bm25_norm.get(tid, 0.0)
        )
        fused.append((tid, score, meta_map.get(tid, {})))
    
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ==================== Main Unified Search ====================

def unified_search(
    query: str,
    dataset_name: str,
    top_k: int = 10,
    rag_type: str = RAG_TYPE_HYBRID,
    index_key: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    index_base_path: Optional[Path] = None,
) -> List[Tuple[str, float, Dict]]:
    """
    Unified search interface for all retrieval operations.
    
    This is the main entry point for all retrieval across the pipeline.
    Supports BM25, Vector, and Hybrid search modes.
    
    Args:
        query: Query text
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        top_k: Number of results to return
        rag_type: Retrieval type - 'bm25', 'vector', or 'hybrid'
        index_key: Index configuration - 'raw', 'td', 'td_cd', 'td_cd_cs' (default)
        query_embedding: Pre-computed query embedding (optional, avoids re-embedding)
        index_base_path: Base path for indexes. If provided, loads from
            {index_base_path}/indexes/{index_key}/
        
    Returns:
        List of (table_id, score, metadata) tuples sorted by score
        
    Raises:
        ValueError: If rag_type or index_key is invalid
        RuntimeError: If no indexes are available
    """
    rag_type = validate_rag_type(rag_type)
    config = get_search_config(rag_type)
    
    logger.debug(f"Unified search: query='{query[:50]}...', dataset={dataset_name}, "
                 f"rag_type={rag_type}, index_key={index_key}")
    
    # Load indexes
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(
        dataset_name, index_key, index_base_path=index_base_path
    )
    
    # Check if we have required indexes
    if config.enable_vector and faiss_index is None:
        logger.warning(f"Vector search requested but no FAISS index for {dataset_name}/{index_key}")
    
    if config.enable_bm25 and bm25_retriever is None:
        logger.warning(f"BM25 search requested but no BM25 index for {dataset_name}/{index_key}")
    
    if faiss_index is None and bm25_retriever is None:
        raise RuntimeError(
            f"No indexes found for {dataset_name}/{index_key}. "
            f"Run index generation first."
        )
    
    # Embed query if needed for vector search
    if config.enable_vector and faiss_index is not None:
        if query_embedding is None:
            query_embedding = embed_query(query)
    
    # Perform searches based on config
    vec_results = []
    bm25_results = []
    
    if config.enable_vector and faiss_index is not None:
        vec_results = vector_search(
            query_embedding, faiss_index, metadata_list, top_k * 2
        )
    
    if config.enable_bm25 and bm25_retriever is not None:
        bm25_results = bm25_search(
            query, bm25_retriever, table_ids, metadata_list, top_k * 2
        )
    
    # Fuse or select results
    if vec_results and bm25_results:
        if rag_type == RAG_TYPE_HYBRID_SUM:
            fused = normalized_score_fusion(
                vec_results, bm25_results,
                vector_weight=config.vector_weight,
                bm25_weight=config.bm25_weight,
                # Pneuma-style: pass indexes for score completion
                query=query,
                query_embedding=query_embedding,
                faiss_index=faiss_index,
                bm25_retriever=bm25_retriever,
                table_ids=table_ids,
                metadata_list=metadata_list,
            )
        else:
            fused = reciprocal_rank_fusion(
                vec_results, bm25_results,
                vector_weight=config.vector_weight,
                bm25_weight=config.bm25_weight,
            )
    elif vec_results:
        fused = vec_results
    else:
        fused = bm25_results
    
    return fused[:top_k]


def unified_search_batch(
    queries: List[str],
    dataset_name: str,
    top_k: int = 10,
    rag_type: str = RAG_TYPE_HYBRID,
    index_key: Optional[str] = None,
    show_progress: bool = True,
) -> List[List[Tuple[str, float, Dict]]]:
    """
    Batch unified search for multiple queries.
    
    More efficient than calling unified_search repeatedly as it pre-loads
    indexes and can batch embedding generation.
    
    Args:
        queries: List of query texts
        dataset_name: Dataset identifier
        top_k: Number of results per query
        rag_type: Retrieval type
        index_key: Index configuration
        show_progress: Show progress bar
        
    Returns:
        List of results for each query
    """
    from tqdm import tqdm
    
    rag_type = validate_rag_type(rag_type)
    config = get_search_config(rag_type)
    
    # Pre-load indexes
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(
        dataset_name, index_key
    )
    
    # Pre-compute embeddings if needed
    query_embeddings = []
    if config.enable_vector and faiss_index is not None:
        embedder = get_text_embedder()
        query_embeddings = embedder.compute_query_embeddings(queries)
        query_embeddings = [np.array(e, dtype=np.float32) for e in query_embeddings]
    
    # Search each query
    results = []
    iterator = tqdm(enumerate(queries), total=len(queries), desc="Searching") if show_progress else enumerate(queries)
    
    for i, query in iterator:
        query_emb = query_embeddings[i] if query_embeddings else None
        result = unified_search(
            query, dataset_name, top_k, rag_type, index_key, query_embedding=query_emb
        )
        results.append(result)
    
    return results


# ==================== Query Embedding Management ====================

# Cache for query embeddings
_QUERY_EMBEDDING_CACHE: Dict[str, Dict[str, Any]] = {}


def load_query_embeddings(
    dataset_name: str,
    split: str = "train",
    force_reload: bool = False,
) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Load query embeddings from independent index files.
    
    Embeddings are stored in:
    - data/lake/indexes/{dataset}/raw/train_query_faiss/
    - data/lake/indexes/{dataset}/raw/test_query_faiss/
    
    This is the unified interface for Stage 1 and other components
    that need access to query embeddings (e.g., for spectral clustering).
    
    Args:
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        split: Query split - 'train' or 'test' (required)
        force_reload: Force reload even if cached
        
    Returns:
        Tuple of (embeddings_array, query_ids, query_records)
        - embeddings_array: np.ndarray of shape (N, dim)
        - query_ids: List of query ID strings
        - query_records: List of full query dicts
        
    Raises:
        FileNotFoundError: If query index not found (run ingest with index_mode='all' or 'vector')
        ValueError: If split is not 'train' or 'test'
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    
    cache_key = f"{dataset_name}:{split}_query_embeddings"
    
    # Check cache
    if not force_reload and cache_key in _QUERY_EMBEDDING_CACHE:
        cached = _QUERY_EMBEDDING_CACHE[cache_key]
        return (
            cached['embeddings'],
            cached['query_ids'],
            cached['query_records'],
        )
    
    # Load from split-specific index
    index_path = get_index_path(dataset_name, "raw") / f"{split}_query_faiss"
    metadata_path = index_path / "metadata.pkl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Query embeddings not found at {metadata_path}. "
            f"Run ingest with index_mode='all' or 'vector' to generate."
        )
    
    logger.info(f"Loading {split} query embeddings from {index_path}...")
    
    with open(metadata_path, "rb") as f:
        data = pickle.load(f)
    
    query_ids = data["query_ids"]
    metadata_list = data["metadata_list"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    
    logger.info(f"  Loaded {len(query_ids)} {split} query embeddings, shape={embeddings.shape}")
    
    # Cache results
    _QUERY_EMBEDDING_CACHE[cache_key] = {
        'embeddings': embeddings,
        'query_ids': query_ids,
        'query_records': metadata_list,
    }
    
    return embeddings, query_ids, metadata_list


def load_table_embeddings(
    dataset_name: str,
    index_key: str = "raw",
    force_reload: bool = False,
) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Load table embeddings from independent index files.
    
    Embeddings are stored in: data/lake/indexes/{dataset}/{index_key}/faiss/
    
    Args:
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        index_key: Index key ('raw', 'td', 'td_cd', 'td_cd_cs')
        force_reload: Force reload even if cached
        
    Returns:
        Tuple of (embeddings_array, table_ids, metadata_list)
        - embeddings_array: np.ndarray of shape (N, dim)
        - table_ids: List of table ID strings
        - metadata_list: List of metadata dicts
        
    Raises:
        FileNotFoundError: If table index not found
    """
    # Load the unified indexes (which includes FAISS with metadata)
    faiss_index, metadata_list, _, table_ids = load_unified_indexes(
        dataset_name, index_key, force_reload
    )
    
    if faiss_index is None:
        raise FileNotFoundError(
            f"Table embeddings not found for {dataset_name}/{index_key}. "
            f"Run ingest with index_mode='all' or 'vector' to generate."
        )
    
    # Reconstruct embeddings from FAISS index
    # Note: For IndexFlatIP, we can reconstruct all vectors
    n = faiss_index.ntotal
    dim = faiss_index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        embeddings[i] = faiss_index.reconstruct(i)
    
    logger.info(f"  Loaded {n} table embeddings, shape={embeddings.shape}")
    
    return embeddings, table_ids, metadata_list


def get_query_embedding_map(
    dataset_name: str,
    split: str = "train",
) -> Dict[str, np.ndarray]:
    """
    Get query embeddings as a mapping from query_id to embedding.
    
    Convenience wrapper around load_query_embeddings for dict-based access.
    
    Args:
        dataset_name: Dataset identifier
        split: Query split - 'train' or 'test' (required)
        
    Returns:
        Dict mapping query_id to embedding vector
    """
    embeddings, query_ids, _ = load_query_embeddings(dataset_name, split)
    
    return {qid: emb for qid, emb in zip(query_ids, embeddings)}


def clear_query_embedding_cache(dataset_name: Optional[str] = None):
    """Clear query embedding cache."""
    global _QUERY_EMBEDDING_CACHE
    
    if dataset_name is None:
        _QUERY_EMBEDDING_CACHE.clear()
    else:
        keys_to_remove = [k for k in _QUERY_EMBEDDING_CACHE if k.startswith(f"{dataset_name}:")]
        for key in keys_to_remove:
            del _QUERY_EMBEDDING_CACHE[key]


def compute_query_embeddings_batch(
    queries: List[str],
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for a batch of query strings.
    
    Uses BGE-M3 model via the embedding registry.
    
    Args:
        queries: List of query texts
        show_progress: Show progress bar
        
    Returns:
        np.ndarray of shape (N, dim)
    """
    embedder = get_text_embedder()
    embeddings = embedder.compute_query_embeddings(queries)
    return np.array(embeddings, dtype=np.float32)
