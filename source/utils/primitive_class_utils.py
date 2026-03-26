"""
Primitive Class Text Processing Utilities

Functions for handling primitive class tokens in text:
- [Parent > Child] format: Chain format from TBox hierarchy
- Extraction, removal, and replacement operations

Used by:
- Index generation (semantic_search.py)
- RAG context formatting (rag_unified_query_analysis.py)
- BM25/Vector ablation experiments
"""

import re


def remove_primitive_class_markers(text: str) -> str:
    """
    Remove [Parent > Child] chain markers from text.

    The column description format is: [Parent > Child] ColumnName: description
    The bracket markers with chain content are removed entirely.
    
    Args:
        text: Text containing primitive class markers
        
    Returns:
        Text with all [Content] markers removed
        
    Example:
        "[Identifier > Name] PersonName: Full name of person"
        -> "PersonName: Full name of person"
    """
    return re.sub(r'\[[^\]]+\]\s*', '', text)
