"""
Centralized Hyperparameters for DataLake Project

This module contains ONLY parameters that are actively imported and used.
All unused legacy parameters have been removed.

Usage:
    from config.hyperparameters import (
        SUCCESSIVE_HALVING_ETA,
        HYBRID_VECTOR_WEIGHT,
    )
"""

from pathlib import Path

# Auto-load .env file if it exists (for local development)
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)


# ============================= HYBRID RETRIEVAL WEIGHTS =============================
# Reciprocal Rank Fusion (RRF) weights for hybrid search
# Used in: workflows/retrieval/unified_search.py, semantic_search.py
# Score formula: score = vector_weight / (k + vector_rank) + bm25_weight / (k + bm25_rank)

HYBRID_VECTOR_WEIGHT = 0.5  # Weight for dense vector retrieval (semantic similarity)
                            # Higher = prefer FAISS/embedding results

HYBRID_BM25_WEIGHT = 0.5    # Weight for sparse BM25 retrieval (keyword matching)
                            # Higher = prefer BM25/lexical results


# ============================= SUCCESSIVE HALVING (CONTRACT SELECTION) =============================
# Successive Halving algorithm for TransformContract selection
# Reference: Jamieson & Talwalkar (2015) "Non-stochastic Best Arm Identification"
# https://arxiv.org/abs/1502.07943
#
# Algorithm: Given n candidates and budget B, run s_max = ⌈log₂(n)⌉ rounds.
#   - Round k: each candidate gets r_k = ⌊B / (|S_k| × s_max)⌋ samples
#   - After evaluation, keep top 50% (halving)
#   - Total complexity: O(B) evaluations regardless of n

SUCCESSIVE_HALVING_BUDGET_MULTIPLIER = 1.0  # budget = k × n_values (where n_values = column sample count)
                                              # Higher = more thorough evaluation, slower
                                              # Lower = faster but may miss best contract

SUCCESSIVE_HALVING_BUDGET_CAP = 1000  # Upper limit on total sample budget
                                        # Prevents excessive CPU time on large columns
                                        # budget = min(k * n_values, BUDGET_CAP)

SUCCESSIVE_HALVING_ETA = 2  # Elimination ratio per round (η)
                            # η=2: keep top 50% each round (halving)
                            # η=3: keep top 33% each round (more aggressive)

SUCCESSIVE_HALVING_MIN_SAMPLES_PER_ARM = 5  # Minimum samples per candidate per round
                                             # Original paper: no minimum (r_k can be 1)
                                             # Our addition: ensures statistical reliability
                                             # Set to 0 or 1 to follow original paper exactly

SUCCESSIVE_HALVING_MIN_SUCCESS_RATE = 0.85  # Minimum success rate to accept a contract
                                             # Range: 0.0-1.0
                                             # A value must match pattern AND transform successfully
