"""
Configuration Package

Provides hyperparameters and truncation limits for the project.

Usage:
    from config.hyperparameters import HYBRID_VECTOR_WEIGHT, SUCCESSIVE_HALVING_ETA
    from config.truncation_limits import TruncationLimits
"""

from .truncation_limits import TruncationLimits

__all__ = [
    'TruncationLimits',
]
