"""
Query Sampling Strategies for Primitive TBox Generation.

This module provides various sampling strategies for selecting queries
from a pool for CQ generation. Different strategies offer different
trade-offs between diversity and computational cost.

Available Samplers:
    - RandomSampler: Simple random sampling (baseline)
    - ClusterSampler: K-means clustering for diverse coverage
    - (Future) DPPSampler: Determinantal Point Process for maximum diversity

Usage:
    from workflows.retrieval.samplers import get_sampler
    
    sampler = get_sampler("cluster", n_clusters=10)
    sampled = sampler.sample(queries, sample_size=50, embeddings=embeddings)
"""

from .base import QuerySampler, SamplingStrategy
from .random_sampler import RandomSampler
from .cluster_sampler import ClusterSampler
from .tree_design import AgentTreeDesigner, TreeConfig, compute_tree_structure


def get_sampler(strategy: str = "random", **kwargs) -> QuerySampler:
    """
    Factory function to get a sampler by strategy name.
    
    Args:
        strategy: Sampling strategy name ("random", "cluster", etc.)
        **kwargs: Strategy-specific parameters
        
    Returns:
        Instantiated sampler
        
    Raises:
        ValueError: If strategy is not supported
    """
    strategy = strategy.lower()
    
    if strategy == "random":
        return RandomSampler(**kwargs)
    elif strategy == "cluster":
        return ClusterSampler(**kwargs)
    # Future: DPP sampler
    # elif strategy == "dpp":
    #     return DPPSampler(**kwargs)
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Supported strategies: random, cluster"
        )


__all__ = [
    "QuerySampler",
    "SamplingStrategy",
    "RandomSampler",
    "ClusterSampler",
    "AgentTreeDesigner",
    "TreeConfig",
    "compute_tree_structure",
    "get_sampler",
]
