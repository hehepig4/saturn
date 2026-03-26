"""
Retrieval Subgraph Package

Dual-modal retrieval combining semantic (vector) and structural (TBox/ABox) search.

Unified Search Interface (NEW):
    from workflows.retrieval.unified_search import unified_search
    from workflows.retrieval.config import RAG_TYPE_BM25, RAG_TYPE_VECTOR, RAG_TYPE_HYBRID
    
    results = unified_search(query, dataset_name, top_k=100, rag_type="hybrid", index_key="td_cd_cs")
"""

from workflows.retrieval.graph import build_retrieval_graph, run_retrieval
from workflows.retrieval.state import (
    RetrievalState,
    TBoxConstraints,
    ABoxConstraints,
    RetrievalResult,
)

# Unified Search Interface
from workflows.retrieval.config import (
    RAG_TYPE_BM25,
    RAG_TYPE_VECTOR,
    RAG_TYPE_HYBRID,
    INDEX_KEY_RAW,
    INDEX_KEY_TD,
    INDEX_KEY_TD_CD,
    INDEX_KEY_TD_CD_CS,
    get_search_config,
)
from workflows.retrieval.unified_search import (
    unified_search,
    unified_search_batch,
    load_unified_indexes,
)

__all__ = [
    # Graph building
    'build_retrieval_graph',
    'run_retrieval',
    # State and results
    'RetrievalState',
    'TBoxConstraints',
    'ABoxConstraints',
    'RetrievalResult',
    # Unified search interface
    'unified_search',
    'unified_search_batch',
    'load_unified_indexes',
    # RAG types
    'RAG_TYPE_BM25',
    'RAG_TYPE_VECTOR',
    'RAG_TYPE_HYBRID',
    # Index keys
    'INDEX_KEY_RAW',
    'INDEX_KEY_TD',
    'INDEX_KEY_TD_CD',
    'INDEX_KEY_TD_CD_CS',
    'get_search_config',
]
