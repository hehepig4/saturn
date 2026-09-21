"""
Unified Similarity Interface for Clustering and Sampling.

This module provides a unified interface for computing similarity matrices
that can be used for:
1. K-means/Spectral Clustering (based on similarity or embeddings)
2. DPP (Determinantal Point Process) sampling for diversity
3. Hybrid retrieval combining dense vectors and BM25

Supported modes:
- vector: Dense BGE-M3 embeddings → cosine similarity
- bm25: BM25 pairwise similarity matrix
- hybrid: Weighted combination of vector and BM25 similarities

Usage:
    from workflows.retrieval.unified_similarity import (
        compute_similarity_matrix,
        dpp_sample_from_similarity,
        SimilarityMode,
    )
    
    # Compute similarity matrix
    texts = ["query 1", "query 2", "query 3"]
    sim_matrix = compute_similarity_matrix(texts, mode="hybrid")
    
    # DPP sampling for diversity
    selected_indices = dpp_sample_from_similarity(sim_matrix, k=2)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from loguru import logger


class SimilarityMode(Enum):
    """Supported similarity computation modes."""
    VECTOR = "vector"   # Dense embeddings (BGE-M3)
    BM25 = "bm25"       # BM25 sparse retrieval
    HYBRID = "hybrid"   # Weighted combination


# ==================== Caching ====================

_SIMILARITY_CACHE: Dict[str, np.ndarray] = {}
_EMBEDDINGS_CACHE: Dict[str, np.ndarray] = {}


def clear_similarity_cache():
    """Clear all similarity caches."""
    global _SIMILARITY_CACHE, _EMBEDDINGS_CACHE
    _SIMILARITY_CACHE.clear()
    _EMBEDDINGS_CACHE.clear()
    logger.info("Cleared similarity caches")


# ==================== Dense Embeddings ====================

def _compute_dense_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute BGE-M3 dense embeddings for texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        np.ndarray of shape (N, 1024)
    """
    from workflows.retrieval.unified_search import get_text_embedder
    
    embedder = get_text_embedder()
    embeddings = embedder.compute_source_embeddings(texts)
    return np.array(embeddings, dtype=np.float32)


def _compute_vector_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix from embeddings.
    
    Args:
        embeddings: (N, D) array of normalized embeddings
        
    Returns:
        (N, N) cosine similarity matrix with values in [-1, 1]
    """
    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.clip(norms, 1e-10, None)
    
    # Cosine similarity = dot product of normalized vectors
    similarity = embeddings_norm @ embeddings_norm.T
    
    return similarity


# ==================== BM25 Similarity ====================

def _compute_bm25_similarity(texts: List[str]) -> np.ndarray:
    """
    Compute BM25 pairwise similarity matrix.
    
    Each document is treated as both a query and a document.
    Uses bm25s library for efficient computation.
    
    Note: BM25 is inherently asymmetric (query vs document roles differ).
    We symmetrize by averaging: S_ij = (BM25(i,j) + BM25(j,i)) / 2
    
    Normalization: Row-wise softmax after symmetrization to convert
    scores to [0, 1] range while preserving relative ordering.
    
    Args:
        texts: List of text strings
        
    Returns:
        (N, N) symmetric BM25 similarity matrix normalized to [0, 1]
    """
    import bm25s
    
    n = len(texts)
    if n == 0:
        return np.array([])
    
    logger.info(f"Computing BM25 similarity matrix for {n} texts...")
    
    # Tokenize all texts
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    
    # Build BM25 index
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    
    # Compute pairwise similarity
    # Each text is queried against the corpus
    raw_scores = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        # Get scores for this query against all documents
        query_tokens = bm25s.tokenize([texts[i]], stopwords="en", show_progress=False)
        _, scores = retriever.retrieve(query_tokens, k=n, show_progress=False)
        raw_scores[i] = scores[0]
    
    logger.info(f"  BM25 raw score range: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
    
    # Symmetrize: BM25 is asymmetric, but similarity should be symmetric
    # S_ij = (BM25(i,j) + BM25(j,i)) / 2
    symmetric_scores = (raw_scores + raw_scores.T) / 2
    
    # Normalize using min-max scaling (after symmetrization)
    # This ensures [0, 1] range with max = 1 on diagonal (self-similarity)
    min_score = symmetric_scores.min()
    max_score = symmetric_scores.max()
    if max_score > min_score:
        similarity_matrix = (symmetric_scores - min_score) / (max_score - min_score)
    else:
        similarity_matrix = np.ones_like(symmetric_scores)
    
    logger.info(f"  BM25 normalized range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    return similarity_matrix


# ==================== Main Interface ====================

def compute_similarity_matrix(
    texts: List[str],
    mode: Union[str, SimilarityMode] = "vector",
    hybrid_alpha: float = 0.5,
    cache_key: Optional[str] = None,
) -> np.ndarray:
    """
    Compute N×N similarity matrix for clustering/sampling.
    
    This is the ONLY external interface for similarity computation.
    Embeddings are internal implementation details - not exposed.
    
    Args:
        texts: List of text strings
        mode: Similarity mode - "vector", "bm25", or "hybrid"
        hybrid_alpha: Weight for vector similarity in hybrid mode (0-1)
        cache_key: Cache key for storing/retrieving results
        
    Returns:
        (N, N) similarity matrix with values normalized to [0, 1]
        
    Note:
        - vector: Uses BGE-M3 embeddings internally → cosine similarity
        - bm25: Uses BM25 pairwise similarity → softmax normalization
        - hybrid: α * vector + (1-α) * bm25
    """
    # Normalize mode
    if isinstance(mode, str):
        mode = SimilarityMode(mode.lower())
    
    # Check cache
    if cache_key and cache_key in _SIMILARITY_CACHE:
        logger.info(f"Using cached similarity matrix: {cache_key}")
        return _SIMILARITY_CACHE[cache_key]
    
    n = len(texts)
    if n == 0:
        return np.array([])
    
    logger.info(f"Computing {mode.value} similarity matrix for {n} texts...")
    
    if mode == SimilarityMode.VECTOR:
        # Dense vector cosine similarity (embeddings computed internally)
        embeddings = _compute_dense_embeddings(texts)
        sim_matrix = _compute_vector_similarity(embeddings)
        # Normalize cosine similarity from [-1, 1] to [0, 1]
        sim_matrix = (sim_matrix + 1) / 2
        
    elif mode == SimilarityMode.BM25:
        # BM25 pairwise similarity (already normalized in _compute_bm25_similarity)
        sim_matrix = _compute_bm25_similarity(texts)
        
    elif mode == SimilarityMode.HYBRID:
        # Weighted combination of vector and BM25
        sim_vec = compute_similarity_matrix(texts, mode=SimilarityMode.VECTOR)
        sim_bm25 = compute_similarity_matrix(texts, mode=SimilarityMode.BM25)
        
        # Both are already in [0, 1]
        sim_matrix = hybrid_alpha * sim_vec + (1 - hybrid_alpha) * sim_bm25
        
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")
    
    # Ensure diagonal is 1.0 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Cache result
    if cache_key:
        _SIMILARITY_CACHE[cache_key] = sim_matrix
    
    logger.info(f"  Similarity matrix shape: {sim_matrix.shape}, range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
    
    return sim_matrix


# ==================== DPP Sampling ====================

def dpp_sample_from_similarity(
    similarity_matrix: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Sample k diverse items using Determinantal Point Process (DPP).
    
    DPP naturally selects diverse subsets based on similarity kernel.
    Higher similarity between items → lower probability of co-selection.
    
    Args:
        similarity_matrix: (N, N) positive semi-definite similarity matrix (L kernel)
        k: Number of items to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of selected indices
    """
    from dppy.finite_dpps import FiniteDPP
    
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"DPP sampling: selecting {k} from {n} items...")
    
    try:
        # Ensure matrix is valid for DPP (positive semi-definite)
        # 1. Clip to [0, 1] range
        L = np.clip(similarity_matrix, 0.0, 1.0).astype(np.float64)
        # 2. Ensure symmetric
        L = (L + L.T) / 2
        # 3. Add small diagonal to ensure positive definiteness
        L = L + 1e-6 * np.eye(n)
        
        # Create DPP with L-ensemble (likelihood kernel)
        dpp = FiniteDPP('likelihood', **{'L': L})
        
        # Exact k-DPP sampling - returns indices directly
        selected_indices = dpp.sample_exact_k_dpp(size=k)
        
        logger.info(f"  DPP selected {len(selected_indices)} items")
        return list(selected_indices)
        
    except Exception as e:
        logger.warning(f"DPP sampling failed: {e}, falling back to random sampling")
        return list(np.random.choice(n, size=k, replace=False))


def stratified_dpp_sample(
    similarity_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Stratified DPP sampling: sample from each cluster proportionally.
    
    This ensures representation from all clusters while maintaining
    diversity within each cluster.
    
    Args:
        similarity_matrix: (N, N) similarity matrix
        cluster_labels: (N,) array of cluster assignments
        k: Total number of items to sample
        seed: Random seed
        
    Returns:
        List of selected indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    if k >= n:
        return list(range(n))
    
    logger.info(f"Stratified DPP: {k} samples from {n_clusters} clusters, {n} total items")
    
    # Calculate samples per cluster (proportional to cluster size)
    cluster_sizes = {c: (cluster_labels == c).sum() for c in unique_clusters}
    total_size = sum(cluster_sizes.values())
    
    # Allocate samples proportionally, with minimum 1 per cluster if possible
    samples_per_cluster = {}
    remaining = k
    
    for c in unique_clusters:
        # Proportional allocation
        proportion = cluster_sizes[c] / total_size
        allocated = max(1, int(k * proportion))
        allocated = min(allocated, cluster_sizes[c], remaining)
        samples_per_cluster[c] = allocated
        remaining -= allocated
    
    # Distribute any remaining samples to largest clusters
    if remaining > 0:
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_sizes[c], reverse=True)
        for c in sorted_clusters:
            if remaining <= 0:
                break
            can_add = cluster_sizes[c] - samples_per_cluster[c]
            add = min(can_add, remaining)
            samples_per_cluster[c] += add
            remaining -= add
    
    # Sample from each cluster using DPP
    selected = []
    for c in unique_clusters:
        cluster_indices = np.where(cluster_labels == c)[0]
        n_sample = samples_per_cluster[c]
        
        if n_sample <= 0:
            continue
        
        if n_sample >= len(cluster_indices):
            selected.extend(cluster_indices.tolist())
        else:
            # Extract sub-matrix for this cluster
            sub_sim = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # DPP sample within cluster
            sub_selected = dpp_sample_from_similarity(sub_sim, n_sample, seed=seed)
            selected.extend(cluster_indices[sub_selected].tolist())
    
    logger.info(f"  Stratified DPP selected {len(selected)} items from {n_clusters} clusters")
    return selected


# ==================== Spectral Clustering ====================

def spectral_cluster_balanced(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    seed: Optional[int] = None,
    min_cluster_ratio: float = 0.5,
) -> np.ndarray:
    """
    Spectral Clustering with balanced cluster assignment.
    
    This unifies clustering for all similarity modes (vector, bm25, hybrid).
    Uses spectral embedding followed by balanced K-means.
    
    Algorithm:
        1. Compute normalized Laplacian from similarity matrix
        2. Extract k smallest eigenvectors as spectral embedding
        3. Run balanced K-means on the embedding
    
    Args:
        similarity_matrix: (N, N) symmetric similarity matrix
        n_clusters: Number of clusters
        seed: Random seed for reproducibility
        min_cluster_ratio: Minimum cluster size as ratio of (n/k), default 0.5
        
    Returns:
        (N,) array of cluster labels
    """
    from scipy.sparse.csgraph import laplacian
    from scipy.linalg import eigh
    
    n = similarity_matrix.shape[0]
    
    # Edge case: fewer samples than clusters
    if n <= n_clusters:
        return np.arange(n)
    
    logger.info(f"Spectral clustering: {n} items into {n_clusters} clusters...")
    
    # Step 1: Compute normalized Laplacian
    # L_norm = D^{-1/2} * L * D^{-1/2} where L = D - S
    # Using scipy's laplacian with normed=True
    L_norm = laplacian(similarity_matrix, normed=True)
    
    # Step 2: Compute k smallest eigenvectors
    # The first eigenvector is trivial (constant), so we take indices 1 to k
    try:
        eigenvalues, eigenvectors = eigh(L_norm)
        # Take k eigenvectors (skip index 0)
        embedding = eigenvectors[:, 1:n_clusters+1]
    except Exception as e:
        logger.warning(f"Eigendecomposition failed: {e}, using fallback")
        # Fallback: use random embedding
        rng = np.random.RandomState(seed)
        embedding = rng.randn(n, n_clusters)
    
    # Step 3: Normalize rows of embedding (for balanced K-means)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding_norm = embedding / np.clip(norms, 1e-10, None)
    
    # Step 4: Balanced K-means on the embedding
    from workflows.retrieval.samplers.cluster_sampler import ClusterSampler
    
    sampler = ClusterSampler(
        n_clusters=n_clusters,
        seed=seed,
        min_cluster_ratio=min_cluster_ratio,
    )
    labels, _ = sampler._balanced_kmeans(
        embedding_norm, 
        n_clusters=n_clusters, 
        seed=seed,
        min_cluster_ratio=min_cluster_ratio,
    )
    
    # Log cluster sizes
    cluster_sizes = [np.sum(labels == k) for k in range(n_clusters)]
    logger.info(f"  Spectral clustering: sizes={cluster_sizes}, range=[{min(cluster_sizes)}, {max(cluster_sizes)}]")
    
    return labels


# ==================== Export ====================

__all__ = [
    "SimilarityMode",
    "compute_similarity_matrix",
    "dpp_sample_from_similarity",
    "stratified_dpp_sample",
    "spectral_cluster_balanced",
    "clear_similarity_cache",
]
