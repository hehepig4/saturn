"""
Unified Retrieval Configuration Module

Provides centralized configuration for all retrieval operations across the pipeline.
Supports three retrieval types: bm25, vector, hybrid.

Usage:
    from workflows.retrieval.config import RAG_TYPE_HYBRID, get_search_config
    
    config = get_search_config(RAG_TYPE_HYBRID)
    # Returns: {'enable_bm25': True, 'enable_vector': True, 'vector_weight': 0.5, 'bm25_weight': 0.5}
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

# ==================== RAG Type Constants ====================

RAG_TYPE_BM25 = "bm25"
RAG_TYPE_VECTOR = "vector"
RAG_TYPE_HYBRID = "hybrid"
RAG_TYPE_HYBRID_SUM = "hybrid-sum"

VALID_RAG_TYPES = [RAG_TYPE_BM25, RAG_TYPE_VECTOR, RAG_TYPE_HYBRID, RAG_TYPE_HYBRID_SUM]


# ==================== Search Configuration ====================

@dataclass
class SearchConfig:
    """Configuration for search operations."""
    enable_bm25: bool
    enable_vector: bool
    vector_weight: float
    bm25_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_bm25": self.enable_bm25,
            "enable_vector": self.enable_vector,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
        }


# Pre-defined configurations for each RAG type
_SEARCH_CONFIGS: Dict[str, SearchConfig] = {
    RAG_TYPE_BM25: SearchConfig(
        enable_bm25=True,
        enable_vector=False,
        vector_weight=0.0,
        bm25_weight=1.0,
    ),
    RAG_TYPE_VECTOR: SearchConfig(
        enable_bm25=False,
        enable_vector=True,
        vector_weight=1.0,
        bm25_weight=0.0,
    ),
    RAG_TYPE_HYBRID: SearchConfig(
        enable_bm25=True,
        enable_vector=True,
        vector_weight=0.5,
        bm25_weight=0.5,
    ),
    RAG_TYPE_HYBRID_SUM: SearchConfig(
        enable_bm25=True,
        enable_vector=True,
        vector_weight=0.5,
        bm25_weight=0.5,
    ),
}


def get_search_config(rag_type: str) -> SearchConfig:
    """
    Get search configuration for a given RAG type.
    
    Args:
        rag_type: One of 'bm25', 'vector', 'hybrid'
        
    Returns:
        SearchConfig with appropriate settings
        
    Raises:
        ValueError: If rag_type is not valid
    """
    if rag_type not in _SEARCH_CONFIGS:
        raise ValueError(
            f"Invalid rag_type: {rag_type}. Must be one of {VALID_RAG_TYPES}"
        )
    return _SEARCH_CONFIGS[rag_type]


def validate_rag_type(rag_type: str) -> str:
    """
    Validate and normalize RAG type.
    
    Args:
        rag_type: RAG type string to validate
        
    Returns:
        Normalized RAG type string
        
    Raises:
        ValueError: If rag_type is not valid
    """
    rag_type = rag_type.lower().strip()
    if rag_type not in VALID_RAG_TYPES:
        raise ValueError(
            f"Invalid rag_type: {rag_type}. Must be one of {VALID_RAG_TYPES}"
        )
    return rag_type


# ==================== Index Key Constants ====================

INDEX_KEY_TD = "td"           # table_description only
INDEX_KEY_TD_CD = "td_cd"     # table_description + column_descriptions
INDEX_KEY_TD_CD_CS = "td_cd_cs"  # all three fields (default)
INDEX_KEY_RAW = "raw"         # raw table_text from ingest (for Stage 1)

DEFAULT_INDEX_KEY = INDEX_KEY_TD_CD_CS
VALID_INDEX_KEYS = [INDEX_KEY_TD, INDEX_KEY_TD_CD, INDEX_KEY_TD_CD_CS, INDEX_KEY_RAW]

# Ablation suffix for primitive class removal
INDEX_KEY_NO_PC_SUFFIX = "_no_pc"


def validate_index_key(index_key: Optional[str]) -> str:
    """
    Validate and normalize index key.
    
    Supports _no_pc suffix for primitive class ablation mode.
    E.g., "td_cd_cs_no_pc" is valid if "td_cd_cs" is valid.
    
    Args:
        index_key: Index key string to validate (None returns default)
        
    Returns:
        Normalized index key string
        
    Raises:
        ValueError: If index_key is not valid
    """
    if index_key is None:
        return DEFAULT_INDEX_KEY
    
    index_key = index_key.lower().strip()
    
    # Check for _no_pc suffix (ablation mode)
    has_no_pc_suffix = index_key.endswith(INDEX_KEY_NO_PC_SUFFIX)
    base_key = index_key[:-len(INDEX_KEY_NO_PC_SUFFIX)] if has_no_pc_suffix else index_key
    
    if base_key not in VALID_INDEX_KEYS:
        raise ValueError(
            f"Invalid index_key: {index_key}. Base key must be one of {VALID_INDEX_KEYS}"
        )
    
    # Return with suffix if it was provided
    return index_key

