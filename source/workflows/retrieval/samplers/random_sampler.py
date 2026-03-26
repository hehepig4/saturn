"""
Random sampling strategy for query selection.

This is the simplest baseline strategy that randomly samples queries
without considering any semantic structure.
"""

import random
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger

from .base import QuerySampler, SamplingResult


class RandomSampler(QuerySampler):
    """
    Random sampling strategy.
    
    Randomly samples queries from the pool. This is the simplest and fastest
    strategy but may miss important query types or oversample common patterns.
    
    Args:
        prefer_with_gt: Whether to prioritize queries with ground truth
        seed: Random seed for reproducibility (None for random)
    """
    
    def __init__(
        self,
        prefer_with_gt: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(prefer_with_gt=prefer_with_gt)
        self.seed = seed
        self._rng = random.Random(seed)
        
    @property
    def name(self) -> str:
        return "RandomSampler"
    
    @property
    def requires_embeddings(self) -> bool:
        return False
    
    def _sample_impl(
        self,
        queries: List[Dict[str, Any]],
        sample_size: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> SamplingResult:
        """
        Randomly sample queries from the pool.
        
        Args:
            queries: Pool of queries to sample from
            sample_size: Number of queries to sample
            embeddings: Not used in random sampling
            
        Returns:
            SamplingResult with randomly sampled queries
        """
        # Get random indices
        all_indices = list(range(len(queries)))
        sampled_indices = self._rng.sample(all_indices, sample_size)
        
        # Sort indices for consistent output
        sampled_indices.sort()
        
        # Get sampled queries
        sampled_queries = [queries[i] for i in sampled_indices]
        
        logger.info(
            f"[{self.name}] Sampled {len(sampled_queries)}/{len(queries)} queries"
        )
        
        return SamplingResult(
            sampled_queries=sampled_queries,
            sample_indices=sampled_indices,
            metadata={
                "strategy": "random",
                "seed": self.seed,
            }
        )
    
    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random generator with a new seed."""
        self.seed = seed
        self._rng = random.Random(seed)
