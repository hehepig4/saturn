"""
Sampling Utilities

Provides consistent random sampling functions across the codebase.
Replaces naive slicing like `values[:N]` with proper random sampling.

Key principle: When we need "sample values", we should randomly sample
to get a representative view, not just take the first N items which
may be biased (e.g., sorted data, header rows, etc.).
"""

import random
from typing import List, TypeVar, Optional, Sequence

T = TypeVar('T')

# Default random seed for reproducibility in tests
# Set to None for true randomness in production
_DEFAULT_SEED: Optional[int] = None



def sample_values(
    values: Sequence[T],
    n: int,
    seed: Optional[int] = None,
    preserve_order: bool = False,
) -> List[T]:
    """
    Randomly sample n values from a sequence.
    
    Args:
        values: Source sequence to sample from
        n: Number of samples to return
        seed: Random seed for reproducibility (overrides global seed)
        preserve_order: If True, maintain original order of sampled items
        
    Returns:
        List of n randomly sampled values (or all values if len < n)
        
    Example:
        >>> sample_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        [4, 7, 2]  # Random 3 values
        
        >>> sample_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, preserve_order=True)
        [2, 4, 7]  # Same 3 values but in original order
    """
    if not values:
        return []
    
    # Use provided seed or global seed
    effective_seed = seed if seed is not None else _DEFAULT_SEED
    if effective_seed is not None:
        rng = random.Random(effective_seed)
    else:
        rng = random.Random()
    
    # If we want all or more than available, return all
    if n >= len(values):
        return list(values)
    
    if preserve_order:
        # Sample indices and sort them to preserve order
        indices = sorted(rng.sample(range(len(values)), n))
        return [values[i] for i in indices]
    else:
        # Direct random sample
        return rng.sample(list(values), n)


def sample_for_llm_prompt(
    values: Sequence[str],
    n: int = 15,
    max_value_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Sample values and format them for LLM prompt.
    
    This is the primary function for preparing sample values to show LLMs.
    It:
    1. Filters out null/empty values
    2. Randomly samples n values
    3. Truncates overly long values
    4. Formats as quoted, comma-separated string
    
    Args:
        values: Source values to sample from
        n: Number of samples (default 15)
        max_value_length: Maximum length per value before truncation
        seed: Random seed for reproducibility
        
    Returns:
        Formatted string like: "value1", "value2", "value3"
    """
    # Use centralized truncation limits
    if max_value_length is None:
        from config.truncation_limits import TruncationLimits
        max_value_length = TruncationLimits.SAMPLE_VALUE_MAX_LENGTH
    
    # Filter and clean
    clean_values = [
        str(v).strip() for v in values 
        if v is not None and str(v).strip()
    ]
    
    if not clean_values:
        return ""
    
    # Random sample
    sampled = sample_values(clean_values, n, seed=seed)
    
    # Truncate long values
    truncated = [
        v[:max_value_length] + "..." if len(v) > max_value_length else v
        for v in sampled
    ]
    
    # Format as quoted string
    return ", ".join(f'"{v}"' for v in truncated)


