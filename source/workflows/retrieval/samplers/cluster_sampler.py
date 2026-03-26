"""
Cluster-based sampling strategy for query selection.

This strategy uses K-means clustering to ensure diverse coverage of
the query embedding space, sampling representative queries from each cluster.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from .base import QuerySampler, SamplingResult

# Try to import k-means-constrained for balanced clustering
try:
    from k_means_constrained import KMeansConstrained
    _HAS_KMEANS_CONSTRAINED = True
except ImportError:
    _HAS_KMEANS_CONSTRAINED = False


class ClusterSampler(QuerySampler):
    """
    Cluster-based sampling using K-means.
    
    This strategy clusters query embeddings and samples representative
    queries from each cluster, ensuring diverse coverage of the semantic space.
    
    Algorithm:
        1. Run K-means on query embeddings
        2. For each cluster, sample proportionally to cluster size
        3. Prefer queries closest to cluster centroids (representatives)
    
    Args:
        prefer_with_gt: Whether to prioritize queries with ground truth
        n_clusters: Number of clusters (auto if None, based on sample_size)
        cluster_ratio: Ratio of sample_size to n_clusters (default 5)
                      Used when n_clusters is None: n_clusters = sample_size / ratio
        seed: Random seed for reproducibility
        sample_per_cluster: Strategy for within-cluster sampling
                           "centroid" = closest to centroid (default)
                           "random" = random within cluster
        min_cluster_ratio: Minimum cluster size as a ratio of (n/k).
                          0.0 = no constraint, 1.0 = hard balance.
                          Default 0.5 means each cluster must have at least 50% of (n/k).
    """
    
    def __init__(
        self,
        prefer_with_gt: bool = True,
        n_clusters: Optional[int] = None,
        cluster_ratio: float = 5.0,
        seed: Optional[int] = 42,
        sample_per_cluster: str = "centroid",
        min_cluster_ratio: float = 0.5,
    ):
        super().__init__(prefer_with_gt=prefer_with_gt)
        self.n_clusters = n_clusters
        self.cluster_ratio = cluster_ratio
        self.seed = seed
        self.sample_per_cluster = sample_per_cluster
        self.min_cluster_ratio = min_cluster_ratio
        self._rng = random.Random(seed)
        
    @property
    def name(self) -> str:
        return "ClusterSampler"
    
    @property
    def requires_embeddings(self) -> bool:
        return True
    
    def _sample_impl(
        self,
        queries: List[Dict[str, Any]],
        sample_size: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> SamplingResult:
        """
        Sample queries using K-means clustering.
        
        Args:
            queries: Pool of queries to sample from
            sample_size: Number of queries to sample
            embeddings: Query embedding vectors (required)
            
        Returns:
            SamplingResult with cluster-sampled queries
        """
        # Determine number of clusters
        if self.n_clusters is not None:
            k = min(self.n_clusters, len(queries))
        else:
            # Auto-determine based on sample size
            k = max(2, int(sample_size / self.cluster_ratio))
            k = min(k, len(queries))
        
        logger.info(f"[{self.name}] Clustering into {k} clusters...")
        
        # Run K-means
        labels, centroids = self._kmeans(
            embeddings, 
            n_clusters=k,
            max_iter=100,
            seed=self.seed
        )
        
        # Get cluster statistics
        cluster_sizes = np.bincount(labels, minlength=k)
        logger.info(f"[{self.name}] Cluster sizes: min={cluster_sizes.min()}, "
                   f"max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
        
        # Calculate samples per cluster (proportional to size)
        samples_per_cluster = self._allocate_samples(cluster_sizes, sample_size)
        
        # Sample from each cluster
        sampled_indices = []
        cluster_info = {}
        
        for cluster_id in range(k):
            # Get indices in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            n_from_cluster = samples_per_cluster[cluster_id]
            if n_from_cluster == 0 or len(cluster_indices) == 0:
                continue
            
            # Sample from cluster
            if self.sample_per_cluster == "centroid":
                # Select points closest to centroid
                selected = self._sample_by_centroid(
                    cluster_indices, 
                    embeddings, 
                    centroids[cluster_id],
                    n_from_cluster
                )
            else:
                # Random sampling within cluster
                selected = self._rng.sample(
                    list(cluster_indices), 
                    min(n_from_cluster, len(cluster_indices))
                )
            
            sampled_indices.extend(selected)
            cluster_info[cluster_id] = {
                "size": int(len(cluster_indices)),
                "sampled": len(selected),
            }
        
        # Sort for consistent output
        sampled_indices = sorted(set(sampled_indices))
        sampled_queries = [queries[i] for i in sampled_indices]
        
        logger.info(
            f"[{self.name}] Sampled {len(sampled_queries)}/{len(queries)} queries "
            f"from {k} clusters"
        )
        
        return SamplingResult(
            sampled_queries=sampled_queries,
            sample_indices=sampled_indices,
            metadata={
                "strategy": "cluster",
                "n_clusters": k,
                "cluster_info": cluster_info,
                "seed": self.seed,
                "labels": labels.tolist(),  # Full cluster assignments
            }
        )
    
    def _kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple K-means implementation without sklearn dependency.
        
        Args:
            X: Data matrix (n_samples, n_features)
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            seed: Random seed
            
        Returns:
            Tuple of (labels, centroids)
        """
        rng = np.random.RandomState(seed)
        n_samples, n_features = X.shape
        
        # Initialize centroids using k-means++ style
        centroids = self._kmeans_plus_plus_init(X, n_clusters, rng)
        
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for iteration in range(max_iter):
            # Assign points to nearest centroid
            distances = self._pairwise_distances(X, centroids)
            new_labels = np.argmin(distances, axis=1)
            
            # Check convergence
            if np.array_equal(labels, new_labels):
                logger.debug(f"K-means converged at iteration {iteration}")
                break
            
            labels = new_labels
            
            # Update centroids
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    centroids[k] = X[mask].mean(axis=0)
        
        return labels, centroids
    
    def _balanced_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
        seed: Optional[int] = None,
        min_cluster_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balanced K-means: K-means with size-balanced cluster assignment.
        
        Uses k-means-constrained library if available, otherwise falls back
        to a greedy balanced assignment algorithm.
        
        Args:
            X: Data matrix (n_samples, n_features)
            n_clusters: Number of clusters
            max_iter: Maximum iterations for K-means
            seed: Random seed
            min_cluster_ratio: Minimum cluster size as ratio of (n/k).
                              0.0 = no constraint, 1.0 = hard balance.
                              If None, uses self.min_cluster_ratio.
            
        Returns:
            Tuple of (labels, centroids)
        """
        n_samples = X.shape[0]
        
        # Use instance attribute if not provided
        if min_cluster_ratio is None:
            min_cluster_ratio = self.min_cluster_ratio
        
        # Edge case: fewer samples than clusters
        if n_samples <= n_clusters:
            labels = np.arange(n_samples)
            centroids = X.copy()
            if n_samples < n_clusters:
                # Pad centroids with zeros for missing clusters
                padding = np.zeros((n_clusters - n_samples, X.shape[1]))
                centroids = np.vstack([centroids, padding])
            return labels, centroids
        
        # Calculate size constraints
        avg_size = n_samples / n_clusters
        size_max = int(np.ceil(avg_size))  # ceil(n/k)
        size_min = max(1, int(avg_size * min_cluster_ratio))
        
        # Try k-means-constrained if available
        if _HAS_KMEANS_CONSTRAINED and min_cluster_ratio > 0:
            try:
                return self._balanced_kmeans_constrained(
                    X, n_clusters, max_iter, seed, size_min, size_max
                )
            except Exception as e:
                logger.warning(f"k-means-constrained failed: {e}, falling back to greedy")
        
        # Fallback to greedy balanced assignment
        return self._balanced_kmeans_greedy(
            X, n_clusters, max_iter, seed, size_min, size_max
        )
    
    def _balanced_kmeans_constrained(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int,
        seed: Optional[int],
        size_min: int,
        size_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balanced K-means using k-means-constrained library.
        
        This uses Google OR-Tools' Minimum Cost Flow solver for optimal
        balanced cluster assignment.
        """
        logger.info(
            f"Using k-means-constrained: n={X.shape[0]}, k={n_clusters}, "
            f"size_min={size_min}, size_max={size_max}"
        )
        
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            init='k-means++',
            n_init=10,
            max_iter=max_iter,
            random_state=seed,
            n_jobs=1,  # Single thread for reproducibility
        )
        
        labels = clf.fit_predict(X)
        centroids = clf.cluster_centers_
        
        # Log cluster sizes
        cluster_sizes = [np.sum(labels == k) for k in range(n_clusters)]
        logger.info(
            f"k-means-constrained: sizes={cluster_sizes}, "
            f"range=[{min(cluster_sizes)}, {max(cluster_sizes)}]"
        )
        
        return labels, centroids
    
    def _balanced_kmeans_greedy(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int,
        seed: Optional[int],
        size_min: int,
        size_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balanced K-means using greedy assignment (fallback).
        
        Algorithm:
            1. Run standard K-means to find centroids
            2. Compute distance from each point to all centroids
            3. Greedily assign points to nearest non-full cluster
        """
        n_samples = X.shape[0]
        
        logger.info(
            f"Using greedy balanced K-means: n={n_samples}, k={n_clusters}, "
            f"size_min={size_min}, size_max={size_max}"
        )
        
        # Step 1: Find centroids using standard K-means
        _, centroids = self._kmeans(X, n_clusters, max_iter=max_iter, seed=seed)
        
        # Step 2: Compute all distances
        distances = self._pairwise_distances(X, centroids)  # (n_samples, n_clusters)
        
        # Step 3: Balanced assignment with capacity constraints
        cluster_counts = np.zeros(n_clusters, dtype=np.int32)
        labels = np.full(n_samples, -1, dtype=np.int32)
        
        # Sort points by their minimum distance to any centroid (most confident first)
        min_distances = distances.min(axis=1)
        sorted_indices = np.argsort(min_distances)
        
        for idx in sorted_indices:
            point_distances = distances[idx]
            
            # Find nearest cluster that still has capacity
            sorted_clusters = np.argsort(point_distances)
            for cluster_id in sorted_clusters:
                if cluster_counts[cluster_id] < size_max:
                    labels[idx] = cluster_id
                    cluster_counts[cluster_id] += 1
                    break
        
        # Verify all points assigned
        unassigned = np.sum(labels == -1)
        if unassigned > 0:
            logger.warning(f"Greedy balanced K-means: {unassigned} points unassigned, falling back")
            # Fallback: assign to any cluster
            for idx in np.where(labels == -1)[0]:
                labels[idx] = idx % n_clusters
        
        # Recompute centroids based on balanced assignment
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids[k] = X[mask].mean(axis=0)
        
        # Log balance info
        cluster_sizes = [np.sum(labels == k) for k in range(n_clusters)]
        logger.info(
            f"Greedy balanced K-means: sizes={cluster_sizes}, "
            f"range=[{min(cluster_sizes)}, {max(cluster_sizes)}]"
        )
        
        return labels, centroids
    
    def _kmeans_plus_plus_init(
        self,
        X: np.ndarray,
        n_clusters: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        K-means++ initialization for better centroid selection.
        
        Args:
            X: Data matrix
            n_clusters: Number of clusters
            rng: Random state
            
        Returns:
            Initial centroids
        """
        n_samples = X.shape[0]
        centroids = np.zeros((n_clusters, X.shape[1]))
        
        # First centroid: random
        idx = rng.randint(n_samples)
        centroids[0] = X[idx]
        
        # Remaining centroids: proportional to squared distance
        for k in range(1, n_clusters):
            distances = self._pairwise_distances(X, centroids[:k])
            min_distances = distances.min(axis=1)
            
            # Probability proportional to squared distance
            probs = min_distances ** 2
            probs /= probs.sum()
            
            idx = rng.choice(n_samples, p=probs)
            centroids[k] = X[idx]
        
        return centroids
    
    def _pairwise_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.
        
        Args:
            X: First matrix (n_samples_x, n_features)
            Y: Second matrix (n_samples_y, n_features)
            
        Returns:
            Distance matrix (n_samples_x, n_samples_y)
        """
        # Using ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        
        distances_sq = X_norm_sq + Y_norm_sq.T - 2 * X @ Y.T
        
        # Numerical stability
        distances_sq = np.maximum(distances_sq, 0)
        
        return np.sqrt(distances_sq)
    
    def _allocate_samples(
        self,
        cluster_sizes: np.ndarray,
        total_samples: int,
    ) -> List[int]:
        """
        Allocate samples to clusters proportionally to their sizes.
        
        Ensures at least 1 sample per non-empty cluster if possible.
        
        Args:
            cluster_sizes: Number of points in each cluster
            total_samples: Total number of samples to allocate
            
        Returns:
            List of samples per cluster
        """
        n_clusters = len(cluster_sizes)
        non_empty_clusters = np.sum(cluster_sizes > 0)
        
        if non_empty_clusters == 0:
            return [0] * n_clusters
        
        # Proportional allocation
        total_points = cluster_sizes.sum()
        proportions = cluster_sizes / total_points
        
        # Initial allocation
        allocation = np.floor(proportions * total_samples).astype(int)
        
        # Ensure at least 1 per non-empty cluster if we have enough samples
        if total_samples >= non_empty_clusters:
            for i in range(n_clusters):
                if cluster_sizes[i] > 0 and allocation[i] == 0:
                    allocation[i] = 1
        
        # Distribute remaining samples to largest clusters
        remaining = total_samples - allocation.sum()
        while remaining > 0:
            # Find cluster with most unsampled points
            gaps = cluster_sizes - allocation
            gaps[allocation >= cluster_sizes] = 0  # Don't over-sample
            
            if gaps.max() <= 0:
                break
            
            best_cluster = np.argmax(gaps)
            allocation[best_cluster] += 1
            remaining -= 1
        
        return allocation.tolist()
    
    def _sample_by_centroid(
        self,
        cluster_indices: np.ndarray,
        embeddings: np.ndarray,
        centroid: np.ndarray,
        n_samples: int,
    ) -> List[int]:
        """
        Sample points closest to cluster centroid.
        
        Args:
            cluster_indices: Indices of points in cluster
            embeddings: All embeddings
            centroid: Cluster centroid
            n_samples: Number to sample
            
        Returns:
            Selected indices
        """
        # Get embeddings for cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Sort by distance (closest first)
        sorted_local_indices = np.argsort(distances)
        
        # Take top n_samples
        selected_local = sorted_local_indices[:n_samples]
        
        # Map back to original indices
        return [int(cluster_indices[i]) for i in selected_local]
