"""
Identifier Types

Defines ID types for the saturn project.
"""

from enum import Enum


class IDType(Enum):
    """ID type enumeration"""
    NODE = "node"
    HYPEREDGE = "hyperedge"
    CACHE = "cache"
    NAMESPACE = "namespace"
    DATASET = "dataset"
    GENERIC = "generic"


__all__ = [
    'IDType',
]
