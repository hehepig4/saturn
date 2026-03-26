"""
Base classes for query sampling strategies.

This module defines the abstract interface that all samplers must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


class SamplingStrategy(str, Enum):
    """Enumeration of available sampling strategies."""
    RANDOM = "random"
    CLUSTER = "cluster"
    # Future: DPP (Determinantal Point Process), stratified, etc.
    # DPP = "dpp"
    # STRATIFIED = "stratified"


@dataclass
class SamplingResult:
    """
    Result of a sampling operation.
    
    Attributes:
        sampled_queries: List of sampled query dictionaries
        sample_indices: Original indices of sampled queries
        metadata: Strategy-specific metadata (e.g., cluster assignments)
    """
    sampled_queries: List[Dict[str, Any]]
    sample_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.sampled_queries)
    
    def __iter__(self):
        return iter(self.sampled_queries)


class QuerySampler(ABC):
    """
    Abstract base class for query sampling strategies.
    
    All sampling strategies must implement the `sample` method.
    Samplers should handle edge cases gracefully:
    - Empty query list
    - sample_size > len(queries)
    - Missing embeddings when required
    """
    
    def __init__(self, prefer_with_gt: bool = True):
        """
        Initialize the sampler.
        
        Args:
            prefer_with_gt: Whether to prioritize queries with ground truth
        """
        self.prefer_with_gt = prefer_with_gt
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this sampling strategy."""
        pass
    
    @property
    @abstractmethod
    def requires_embeddings(self) -> bool:
        """Whether this sampler requires embeddings."""
        pass
    
    @abstractmethod
    def _sample_impl(
        self,
        queries: List[Dict[str, Any]],
        sample_size: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> SamplingResult:
        """
        Internal implementation of sampling logic.
        
        Args:
            queries: Pool of queries to sample from
            sample_size: Number of queries to sample
            embeddings: Optional embedding vectors for each query
            
        Returns:
            SamplingResult with sampled queries and metadata
        """
        pass
    
    def sample(
        self,
        queries: List[Dict[str, Any]],
        sample_size: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> SamplingResult:
        """
        Sample queries from the pool.
        
        This method handles common preprocessing:
        1. Validates inputs
        2. Handles edge cases (empty list, oversized sample)
        3. Optionally prioritizes queries with ground truth
        4. Delegates to strategy-specific implementation
        
        Args:
            queries: Pool of queries to sample from
            sample_size: Number of queries to sample
            embeddings: Optional embedding vectors for each query
                       Shape: (n_queries, embedding_dim)
                       
        Returns:
            SamplingResult containing sampled queries and metadata
            
        Raises:
            ValueError: If embeddings required but not provided
        """
        # Validate inputs
        if not queries:
            return SamplingResult(
                sampled_queries=[],
                sample_indices=[],
                metadata={"warning": "Empty query pool"}
            )
        
        if self.requires_embeddings and embeddings is None:
            raise ValueError(
                f"{self.name} requires embeddings but none were provided"
            )
        
        # Validate embeddings shape if provided
        if embeddings is not None:
            if len(embeddings) != len(queries):
                raise ValueError(
                    f"Embeddings length ({len(embeddings)}) doesn't match "
                    f"queries length ({len(queries)})"
                )
        
        # Handle oversized sample request
        if sample_size >= len(queries):
            return SamplingResult(
                sampled_queries=queries.copy(),
                sample_indices=list(range(len(queries))),
                metadata={"note": "Returned all queries (sample_size >= pool_size)"}
            )
        
        # Prioritize queries with ground truth if enabled
        if self.prefer_with_gt:
            queries, embeddings, index_mapping = self._prioritize_gt(
                queries, sample_size, embeddings
            )
        else:
            index_mapping = list(range(len(queries)))
        
        # Delegate to implementation
        result = self._sample_impl(queries, sample_size, embeddings)
        
        # Map indices back to original if we reordered
        if self.prefer_with_gt:
            result.sample_indices = [
                index_mapping[i] for i in result.sample_indices
            ]
        
        return result
    
    def _prioritize_gt(
        self,
        queries: List[Dict[str, Any]],
        sample_size: int,
        embeddings: Optional[np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], List[int]]:
        """
        Reorder queries to prioritize those with ground truth.
        
        If there are enough GT queries, only use those.
        Otherwise, use all GT queries and fill from non-GT.
        
        Returns:
            Tuple of (reordered_queries, reordered_embeddings, original_indices)
        """
        # Separate indices by GT presence
        gt_indices = []
        non_gt_indices = []
        
        for i, q in enumerate(queries):
            if q.get("ground_truth_table_id"):
                gt_indices.append(i)
            else:
                non_gt_indices.append(i)
        
        # Build prioritized order
        if len(gt_indices) >= sample_size:
            # Enough GT queries - use only those
            prioritized_indices = gt_indices
        else:
            # Use all GT + some non-GT
            prioritized_indices = gt_indices + non_gt_indices
        
        # Reorder queries
        prioritized_queries = [queries[i] for i in prioritized_indices]
        
        # Reorder embeddings if provided
        prioritized_embeddings = None
        if embeddings is not None:
            prioritized_embeddings = embeddings[prioritized_indices]
        
        return prioritized_queries, prioritized_embeddings, prioritized_indices
