"""
Final DataProperty Generation for Federated Primitive TBox

This module generates DataProperties for all classes in the final TBox
after all class iterations are complete.

Design:
- Parallel generation: Each class gets DP generated independently
- Guarantee: Every class has at least 1 DP
- Async execution: Uses asyncio for concurrent LLM calls
- Readout templates: Each DP has a natural language template for column description

Based on: subgraph/primitive_tbox/nodes/enhance_data_properties.py
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field

from llm.manager import get_llm_by_purpose
from llm.invoke_with_stats import invoke_structured_llm

# Import from unified datatype module (single source of truth)
from core.datatypes.el_datatypes import (
    EL_SUPPORTED_DATATYPES,
    DATATYPE_STATISTICS,
)


# ============== Output Schemas ==============

class SingleDataPropertyOutput(BaseModel):
    """
    Output for DP generation: EXACTLY ONE DataProperty per class.
    
    Design: Each Primitive Class has ONE primary DataProperty that captures
    the core semantic meaning with standardized naming: has{ClassName}Value.
    """
    range_type: str  # xsd:string, xsd:integer, xsd:decimal, xsd:dateTime, or xsd:anyURI
    readout_template: str  # Template with {placeholder} for statistics
    description: str = ""  # Brief description of what this property captures


# ============== Datatype Reference Generator ==============

# ============== Root Class Hardcoded DataProperty ==============
# The root class "Column" gets a fixed, generic DP for stability.
# This ensures all columns have at least a basic text value property.
# Subclasses will override with more specific, semantically meaningful DPs.

ROOT_CLASS_DATAPROPERTY = {
    "property_name": "hasCellValue",
    "description": "The raw text content of an individual cell",
    "domain": ["Column"],
    "range_type": "xsd:string",
    "readout_template": "Contains {distinct_count} unique values such as {sample_values}.",
}


# ============== Datatype Reference ==============

PRIMARY_DATATYPES = [
    "xsd:integer",
    "xsd:decimal", 
    "xsd:string",
    "xsd:dateTime",
    "xsd:anyURI",
]

DATATYPE_DESCRIPTIONS = {
    "xsd:integer": "e.g., Years, counts, ranks, positions",
    "xsd:decimal": "e.g., Percentages, scores, ratings, amounts",
    "xsd:string": "e.g., Names, text, identifiers",
    "xsd:dateTime": "e.g., Dates and times",
    "xsd:anyURI": "e.g., URLs and URIs",
}


def _build_datatype_reference() -> str:
    """Build datatype reference table from DATATYPE_STATISTICS."""
    lines = ["| Datatype | Use Samples | Statistics |", "|----------|-----------|------------|"]
    for dt in PRIMARY_DATATYPES:
        stats = DATATYPE_STATISTICS.get(dt, [])
        desc = DATATYPE_DESCRIPTIONS.get(dt, "")
        lines.append(f"| {dt} | {desc} | {', '.join(stats)} |")
    return "\n".join(lines)


# ============== Prompt Template ==============
# NOTE: Static content FIRST to maximize prefix cache hit rate.
# Dynamic content (class_name, class_description, context_block) at the END.

# Simplified prompt: ONE DataProperty per class with standardized naming
# The property name is auto-derived as has{ClassName}Value - LLM only selects range_type and readout_template
DP_GENERATION_PROMPT_TEMPLATE = """Design the DataProperty for a column type class.

## What is a DataProperty?

A DataProperty describes the semantic information extracted from each cell in a column.
Each Primitive Class has EXACTLY ONE DataProperty that captures its core meaning.

**Processing Pipeline:**
1. A Python transform extracts values from raw cell text
2. Statistics are computed on extracted values
3. Statistics fill the readout_template for natural language summaries

## Your Task

Choose the appropriate `range_type` and design the `readout_template`.

## Datatypes & Available Statistics

{datatype_reference}

Use ONLY the statistics listed for your chosen datatype in the template placeholders.

## Examples

| Class | range_type | readout_template |
|-------|------------|------------------|
| YearColumn | xsd:integer | "Years from {{min}} to {{max}}" |
| PercentageColumn | xsd:decimal | "Range: {{min}}-{{max}}, avg {{mean}}" |
| PersonNameColumn | xsd:string | "{{distinct_count}} people, e.g., {{sample_values}}" |
| DateColumn | xsd:dateTime | "Dates from {{min}} to {{max}}" |
| URLColumn | xsd:anyURI | "{{distinct_count}} unique URLs" |

---

## Target Class

Name: {class_name}
Description: {class_description}
Parent: {parent_class}

{context_block}
"""


def _build_prompt(
    class_name: str,
    class_description: str,
    parent_class: str,
    context_block: str = "",
) -> str:
    """Build the complete prompt with dynamic datatype reference and optional context."""
    return DP_GENERATION_PROMPT_TEMPLATE.format(
        class_name=class_name,
        class_description=class_description,
        parent_class=parent_class,
        context_block=context_block,
        datatype_reference=_build_datatype_reference(),
    )


# ============== Global for one-time prompt logging ==============

_DP_GENERATION_PROMPT_LOGGED = False


# ============== CQ Retrieval for DP Generation ==============

def _precompute_cq_data(all_cqs: List[Dict], rag_type: str = "vector") -> Optional[Dict]:
    """
    Prepare CQ retrieval data (call once in main thread).
    
    NOTE: CQs are dynamically generated at runtime, so embeddings must be computed
    on-the-fly (cannot be pre-stored like table embeddings).
    
    For vector mode: computes BGE-M3 embeddings only
    For bm25 mode: builds BM25 index only (no embedding model needed)
    For hybrid mode: computes BOTH embeddings AND BM25 index (for RRF fusion)
    
    Args:
        all_cqs: List of all CQs
        rag_type: 'vector', 'bm25', or 'hybrid'
        
    Returns:
        Dict with retrieval data (varies by rag_type)
    """
    if not all_cqs:
        return None
    
    try:
        cq_texts = [cq.get("question", "") for cq in all_cqs]
        result = {
            "cqs": all_cqs,
            "rag_type": rag_type,
        }
        
        # Build BM25 index for bm25 or hybrid mode
        if rag_type in ("bm25", "hybrid"):
            import bm25s
            import Stemmer
            
            stemmer = Stemmer.Stemmer("english")
            corpus_tokens = bm25s.tokenize(cq_texts, stopwords="en", stemmer=stemmer, show_progress=False)
            bm25_retriever = bm25s.BM25()
            bm25_retriever.index(corpus_tokens, show_progress=False)
            
            result["bm25_retriever"] = bm25_retriever
            logger.info(f"  Built BM25 index for CQs: {len(all_cqs)} CQs")
        
        # Compute embeddings for vector or hybrid mode
        if rag_type in ("vector", "hybrid"):
            from store.embedding.embedding_registry import get_registry
            import numpy as np
            
            registry = get_registry()
            embedder = registry.register_function("bge-m3")
            
            cq_embeddings = embedder.compute_source_embeddings(cq_texts)
            cq_embeddings = np.array(cq_embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(cq_embeddings, axis=1, keepdims=True)
            cq_embeddings_normalized = cq_embeddings / norms
            
            result["embeddings"] = cq_embeddings_normalized
            logger.info(f"  Pre-computed CQ embeddings: {len(all_cqs)} CQs")
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to pre-compute CQ data: {e}")
        return None


def _precompute_table_data(table_store_name: str, rag_type: str = "vector") -> Optional[Dict]:
    """
    Pre-load table data and optionally embeddings (call once in main thread).
    
    NOTE: Table embeddings are PRE-STORED at data/lake/indexes/{dataset}/raw/faiss/
    during ingest phase. This function LOADS them, not computes.
    
    For vector mode: loads pre-computed embeddings from disk
    For bm25 mode: builds BM25 index from table texts (no embedding model)
    For hybrid mode: loads embeddings AND builds BM25 index (for RRF fusion)
    
    Args:
        table_store_name: LanceDB table name (e.g., 'fetaqa_tables_entries')
        rag_type: 'vector', 'bm25', or 'hybrid'
        
    Returns:
        Dict with table data and retrieval index
    """
    try:
        from store.store_singleton import get_store
        import json
        
        # Extract dataset name (e.g., 'fetaqa_tables_entries' -> 'fetaqa')
        dataset_name = table_store_name.replace("_tables_entries", "").replace("_tables", "")
        
        # Load metadata from LanceDB for full table info
        store = get_store()
        tbl = store.db.open_table(table_store_name)
        df = tbl.to_pandas()
        
        if df.empty:
            return None
        
        # Pre-parse columns and sample_rows
        tables = []
        table_texts = []  # For BM25 indexing
        for _, row in df.iterrows():
            columns = row.get("columns", [])
            if isinstance(columns, str):
                try:
                    columns = json.loads(columns)
                except:
                    columns = []
            
            sample_rows = row.get("sample_rows", [])
            if isinstance(sample_rows, str):
                try:
                    sample_rows = json.loads(sample_rows)
                except:
                    sample_rows = []
            
            tables.append({
                "table_id": row.get("table_id", ""),
                "document_title": row.get("document_title", ""),
                "columns": columns,
                "sample_rows": sample_rows[:3] if isinstance(sample_rows, list) else [],
            })
            
            # Build table text for BM25
            table_text = row.get("table_text", row.get("document_title", ""))
            table_texts.append(table_text)
        
        result = {
            "tables": tables,
            "rag_type": rag_type,
        }
        
        # Build BM25 index for bm25 or hybrid mode
        if rag_type in ("bm25", "hybrid"):
            import bm25s
            import Stemmer
            
            stemmer = Stemmer.Stemmer("english")
            corpus_tokens = bm25s.tokenize(table_texts, stopwords="en", stemmer=stemmer, show_progress=False)
            bm25_retriever = bm25s.BM25()
            bm25_retriever.index(corpus_tokens, show_progress=False)
            
            result["bm25_retriever"] = bm25_retriever
            logger.info(f"  Built BM25 index for tables: {len(tables)} tables")
        
        # Load embeddings for vector or hybrid mode
        if rag_type in ("vector", "hybrid"):
            from workflows.retrieval.unified_search import load_table_embeddings
            import numpy as np
            
            embeddings, table_ids, metadata_list = load_table_embeddings(
                dataset_name=dataset_name,
                index_key="raw",  # Use raw index for pre-computed embeddings
            )
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / norms
            
            # Build table ID to index mapping
            table_id_to_idx = {tid: i for i, tid in enumerate(table_ids)}
            
            result["embeddings"] = embeddings_normalized
            result["table_id_to_idx"] = table_id_to_idx
            logger.info(f"  Pre-loaded table embeddings: {len(tables)} tables")
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to pre-load table data: {e}")
        return None


def _retrieve_relevant_cqs_fast(
    class_name: str,
    class_description: str,
    precomputed: Dict,
    embedder_func=None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Fast CQ retrieval using pre-computed data (thread-safe).
    
    Supports vector, BM25, and hybrid modes based on precomputed['rag_type'].
    For hybrid mode, uses RRF (Reciprocal Rank Fusion) to combine results.
    
    Args:
        class_name: Name of the class
        class_description: Description of the class
        precomputed: Pre-computed CQ data from _precompute_cq_data
        embedder_func: Function to embed query text (only needed for vector/hybrid mode)
        top_k: Number of CQs to retrieve
        
    Returns:
        List of top-k relevant CQs
    """
    if not precomputed:
        return []
    
    try:
        import numpy as np
        cqs = precomputed["cqs"]
        rag_type = precomputed.get("rag_type", "vector")
        class_query = f"{class_name}: {class_description}"
        
        vec_indices = []
        bm25_indices = []
        
        # Vector retrieval
        if rag_type in ("vector", "hybrid") and "embeddings" in precomputed:
            cq_embeddings = precomputed["embeddings"]
            query_embedding = embedder_func([class_query])[0]
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            similarities = cq_embeddings @ query_norm
            vec_indices = np.argsort(similarities)[::-1][:top_k * 2].tolist()
        
        # BM25 retrieval
        if rag_type in ("bm25", "hybrid") and "bm25_retriever" in precomputed:
            import bm25s
            import Stemmer
            
            bm25_retriever = precomputed["bm25_retriever"]
            stemmer = Stemmer.Stemmer("english")
            query_tokens = bm25s.tokenize([class_query], stopwords="en", stemmer=stemmer, show_progress=False)
            results, scores = bm25_retriever.retrieve(query_tokens, k=top_k * 2, show_progress=False)
            bm25_indices = [int(i) for i in results[0] if i < len(cqs)]
        
        # Fusion for hybrid mode
        if rag_type == "hybrid" and vec_indices and bm25_indices:
            # RRF fusion: score = 1/(k + rank)
            k = 60
            scores = {}
            for rank, idx in enumerate(vec_indices):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
            for rank, idx in enumerate(bm25_indices):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
            
            # Sort by fused score
            sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return [cqs[i] for i in sorted_indices[:top_k]]
        elif vec_indices:
            return [cqs[i] for i in vec_indices[:top_k]]
        elif bm25_indices:
            return [cqs[i] for i in bm25_indices[:top_k]]
        else:
            return []
        
    except Exception as e:
        logger.warning(f"CQ retrieval failed for {class_name}: {e}")
        return []


def _retrieve_relevant_tables_fast(
    class_name: str,
    class_description: str,
    precomputed: Dict,
    embedder_func=None,
    top_k: int = 3,
) -> List[Dict]:
    """
    Fast table retrieval using pre-computed data (thread-safe).
    
    Supports vector, BM25, and hybrid modes based on precomputed['rag_type'].
    For hybrid mode, uses RRF (Reciprocal Rank Fusion) to combine results.
    
    Args:
        class_name: Name of the class
        class_description: Description of the class
        precomputed: Pre-computed table data from _precompute_table_data
        embedder_func: Function to embed query text (only needed for vector/hybrid mode)
        top_k: Number of tables to retrieve
        
    Returns:
        List of top-k relevant tables
    """
    if not precomputed:
        return []
    
    try:
        import numpy as np
        tables = precomputed["tables"]
        rag_type = precomputed.get("rag_type", "vector")
        class_query = f"{class_name}: {class_description}"
        
        vec_indices = []
        bm25_indices = []
        
        # Vector retrieval
        if rag_type in ("vector", "hybrid") and "embeddings" in precomputed:
            table_embeddings = precomputed["embeddings"]
            query_embedding = embedder_func([class_query])[0]
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            similarities = table_embeddings @ query_norm
            vec_indices = np.argsort(similarities)[::-1][:top_k * 2].tolist()
        
        # BM25 retrieval
        if rag_type in ("bm25", "hybrid") and "bm25_retriever" in precomputed:
            import bm25s
            import Stemmer
            
            bm25_retriever = precomputed["bm25_retriever"]
            stemmer = Stemmer.Stemmer("english")
            query_tokens = bm25s.tokenize([class_query], stopwords="en", stemmer=stemmer, show_progress=False)
            results, scores = bm25_retriever.retrieve(query_tokens, k=top_k * 2, show_progress=False)
            bm25_indices = [int(i) for i in results[0] if i < len(tables)]
        
        # Fusion for hybrid mode
        if rag_type == "hybrid" and vec_indices and bm25_indices:
            # RRF fusion: score = 1/(k + rank)
            k = 60
            scores = {}
            for rank, idx in enumerate(vec_indices):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
            for rank, idx in enumerate(bm25_indices):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
            
            # Sort by fused score
            sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return [tables[i] for i in sorted_indices[:top_k]]
        elif vec_indices:
            return [tables[i] for i in vec_indices[:top_k]]
        elif bm25_indices:
            return [tables[i] for i in bm25_indices[:top_k]]
        else:
            return []
        
    except Exception as e:
        logger.warning(f"Table retrieval failed for {class_name}: {e}")
        return []


def _build_context_block(
    relevant_cqs: List[Dict],
    relevant_tables: List[Dict],
) -> str:
    """
    Build context block for DP generation prompt.
    
    Args:
        relevant_cqs: List of relevant CQs
        relevant_tables: List of relevant tables with sample data
        
    Returns:
        Formatted context block string
    """
    parts = []
    
    # Add CQs section
    if relevant_cqs:
        cq_lines = ["## Relevant Competency Questions", "These questions define what this class should capture:", ""]
        for i, cq in enumerate(relevant_cqs[:5], 1):
            question = cq.get("question", "")
            cq_type = cq.get("cq_type", "CQ")
            cq_lines.append(f"{i}. [{cq_type}] {question}")
        parts.append("\n".join(cq_lines))
    
    # Add tables section
    if relevant_tables:
        table_lines = ["## Sample Table Data", "Reference these real examples when designing properties:", ""]
        for table in relevant_tables[:3]:
            title = table.get("document_title", "Unknown")
            columns = table.get("columns", [])
            col_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in columns[:8]]
            sample_rows = table.get("sample_rows", [])
            
            table_lines.append(f"### {title}")
            table_lines.append(f"Columns: {', '.join(col_names)}")
            
            if sample_rows and isinstance(sample_rows, list):
                table_lines.append("Sample rows:")
                for j, row in enumerate(sample_rows[:2]):
                    if isinstance(row, list):
                        row_str = " | ".join(str(cell)[:25] for cell in row[:6])
                        table_lines.append(f"  [{j+1}] {row_str}")
            table_lines.append("")
        
        parts.append("\n".join(table_lines))
    
    return "\n\n".join(parts) if parts else ""


# ============== Auto-generation Functions ==============

def _auto_generate_single_template(dp: Dict) -> Dict:
    """
    Auto-generate readout template for a single DataProperty.
    
    Based on range_type, generates appropriate template and statistics.
    """
    dp_copy = dict(dp)
    prop_name = dp.get("property_name", dp.get("name", ""))
    range_type = dp.get("range_type", "xsd:string")
    
    # Get default statistics for this datatype
    default_stats = DATATYPE_STATISTICS.get(range_type, ["count", "distinct_count"])
    
    # Generate template based on datatype
    if range_type in ("xsd:integer", "xsd:decimal", "xsd:nonNegativeInteger"):
        template = f"{prop_name}: {{min}}-{{max}} (avg {{mean}})"
        stats = ["min", "max", "mean"]
    elif range_type in ("xsd:dateTime", "xsd:dateTimeStamp"):
        template = f"{prop_name}: {{min}} to {{max}}"
        stats = ["min", "max"]
    elif range_type == "xsd:anyURI":
        template = f"{prop_name}: {{distinct_count}} unique URLs"
        stats = ["distinct_count"]
    elif range_type == "xsd:string":
        template = f"{prop_name}: {{distinct_count}} unique values"
        stats = ["distinct_count"]
    else:
        template = f"{prop_name}: {{count}} values"
        stats = ["count"]
    
    # Only set if not already present
    dp_copy["readout_template"] = dp.get("readout_template") or template
    dp_copy["statistics_requirements"] = dp.get("statistics_requirements") or stats
    
    return dp_copy


def _validate_dp_template(dp: Dict) -> List[str]:
    """
    Validate a DataProperty's readout template.
    
    Returns list of warnings.
    """
    warnings = []
    prop_name = dp.get("property_name", dp.get("name", "unknown"))
    template = dp.get("readout_template", "")
    stats = dp.get("statistics_requirements", [])
    range_type = dp.get("range_type", "xsd:string")
    
    # Check EL compliance
    if range_type not in EL_SUPPORTED_DATATYPES:
        warnings.append(f"{prop_name}: range_type '{range_type}' not OWL 2 EL compliant")
    
    # Check template placeholders match statistics
    placeholders = set(re.findall(r'\{(\w+)\}', template))
    stats_set = set(stats)
    
    missing = placeholders - stats_set
    if missing:
        warnings.append(f"{prop_name}: template uses undefined stats: {missing}")
    
    unused = stats_set - placeholders
    if unused and len(unused) > 2:  # Allow some unused stats
        warnings.append(f"{prop_name}: stats defined but not in template: {unused}")
    
    return warnings


def _validate_and_fix_dps(dps: List[Dict]) -> List[Dict]:
    """
    Validate all DPs and fix any issues.
    
    Returns list of fixed DPs.
    """
    fixed_dps = []
    
    for dp in dps:
        dp_copy = dict(dp)
        
        # Ensure readout_template and statistics_requirements exist
        if not dp_copy.get("readout_template") or not dp_copy.get("statistics_requirements"):
            dp_copy = _auto_generate_single_template(dp_copy)
        
        # Validate and log warnings
        warnings = _validate_dp_template(dp_copy)
        for w in warnings:
            logger.warning(f"    ⚠ {w}")
        
        fixed_dps.append(dp_copy)
    
    return fixed_dps


# ============== Main Functions ==============

def _derive_dp_name(class_name: str) -> str:
    """
    Derive standardized DataProperty name from class name.
    
    Convention: has{ClassName without "Column"}Value
    
    Examples:
        RankColumn → hasRankValue
        YearColumn → hasYearValue
        PersonNameColumn → hasPersonNameValue
        Column → hasCellValue (root class)
    """
    if class_name in ("Column", "upo:Column"):
        return "hasCellValue"
    
    # Remove "Column" suffix if present
    base_name = class_name.replace("Column", "").replace("upo:", "")
    
    # Handle edge cases
    if not base_name:
        base_name = "Cell"
    
    return f"has{base_name}Value"


def generate_dp_for_class(
    class_info: Dict[str, Any],
    llm: Any,
    precomputed_cqs: Optional[Dict] = None,
    precomputed_tables: Optional[Dict] = None,
    embedder_func = None,
) -> List[Dict[str, Any]]:
    """
    Generate EXACTLY ONE DataProperty for a class.
    
    Design: Each Primitive Class has ONE primary DataProperty.
    The property name is auto-generated as has{ClassName}Value.
    LLM only decides range_type and readout_template.
    
    Args:
        class_info: Class dict with name, description, parent_class/parent_classes
        llm: LLM instance
        precomputed_cqs: Pre-computed CQ data (from _precompute_cq_data)
        precomputed_tables: Pre-computed table data (from _precompute_table_data)
        embedder_func: Function to embed query text (only for vector/hybrid mode)
    
    Returns:
        List containing EXACTLY ONE DP dict for this class
    """
    global _DP_GENERATION_PROMPT_LOGGED
    
    class_name = class_info.get("name", "Unknown")
    class_description = class_info.get("description", class_info.get("definition", ""))
    # Support both parent_class and parent_classes formats
    parent_class = class_info.get("parent_class")
    if not parent_class:
        parent_classes = class_info.get("parent_classes", ["Column"])
        parent_class = parent_classes[0] if parent_classes else "Column"
    
    # Auto-generate property name
    property_name = _derive_dp_name(class_name)
    
    # ========== Retrieve relevant context using pre-computed embeddings ==========
    context_block = ""
    
    # Retrieve relevant CQs
    relevant_cqs = []
    if precomputed_cqs and embedder_func:
        relevant_cqs = _retrieve_relevant_cqs_fast(
            class_name=class_name,
            class_description=class_description,
            precomputed=precomputed_cqs,
            embedder_func=embedder_func,
            top_k=5,
        )
    
    # Retrieve relevant tables
    relevant_tables = []
    if precomputed_tables and embedder_func:
        relevant_tables = _retrieve_relevant_tables_fast(
            class_name=class_name,
            class_description=class_description,
            precomputed=precomputed_tables,
            embedder_func=embedder_func,
            top_k=3,
        )
    
    # Build context block
    if relevant_cqs or relevant_tables:
        context_block = _build_context_block(relevant_cqs, relevant_tables)
    
    # Build prompt with dynamic datatype reference and context
    prompt = _build_prompt(
        class_name=class_name,
        class_description=class_description,
        parent_class=parent_class,
        context_block=context_block,
    )
    
    # One-time prompt logging for debug
    if not _DP_GENERATION_PROMPT_LOGGED:
        logger.debug("=" * 80)
        logger.debug("[DP Generation] FIRST PROMPT (one-time log):")
        logger.debug("=" * 80)
        logger.debug(prompt)
        logger.debug("=" * 80)
        _DP_GENERATION_PROMPT_LOGGED = True
    
    try:
        # Use simplified schema: SingleDataPropertyOutput
        structured_llm = llm.with_structured_output(SingleDataPropertyOutput)
        result: SingleDataPropertyOutput = invoke_structured_llm(structured_llm, prompt)
        
        # Extract statistics_requirements from template placeholders
        template = result.readout_template or ""
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Build single DP dict with auto-generated name
        dp_dict = {
            "property_name": property_name,  # Auto-generated, NOT from LLM
            "description": result.description or f"Primary value for {class_name}",
            "domain": [class_name],
            "range_type": result.range_type,
            "readout_template": result.readout_template,
            "statistics_requirements": placeholders if placeholders else ["count"],
        }
        
        # Validate and fix (returns list)
        dp_list = _validate_and_fix_dps([dp_dict])
        
        logger.debug(f"  Generated DP for {class_name}: {property_name} ({result.range_type})")
        
        return dp_list
        
    except Exception as e:
        logger.error(f"DP generation failed for {class_name}: {e}")
        # Return a fallback generic DP with auto-generated name and template
        fallback_dp = {
            "property_name": property_name,  # Use standardized name
            "description": f"Primary value for {class_name}",
            "domain": [class_name],
            "range_type": "xsd:string",
        }
        return [_auto_generate_single_template(fallback_dp)]


async def _generate_dp_async(
    class_info: Dict[str, Any],
    llm: Any,
    semaphore: asyncio.Semaphore,
    precomputed_cqs: Optional[Dict] = None,
    precomputed_tables: Optional[Dict] = None,
    embedder_func = None,
) -> List[Dict[str, Any]]:
    """Async wrapper for DP generation with semaphore."""
    import contextvars
    async with semaphore:
        # Capture context (caller tracking ContextVars) before run_in_executor
        # since run_in_executor does NOT auto-propagate context in this Python build
        ctx = contextvars.copy_context()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            ctx.run,
            generate_dp_for_class,
            class_info,
            llm,
            precomputed_cqs,
            precomputed_tables,
            embedder_func,
        )


def generate_all_dp_parallel(
    final_tbox: Dict[str, Any],
    llm_purpose: str = "gemini",
    llm_override_config: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 5,
    all_cqs: Optional[List[Dict]] = None,
    table_store_name: Optional[str] = None,
    rag_type: str = "vector",
) -> List[Dict[str, Any]]:
    """
    Generate DataProperties for all classes in parallel.
    
    The root class "Column" uses a hardcoded DP for stability.
    All other classes get LLM-generated DPs via parallel execution.
    
    Args:
        final_tbox: TBox dict with 'classes'
        llm_purpose: LLM purpose key
        llm_override_config: Optional LLM config override
        max_concurrent: Max concurrent LLM calls
        all_cqs: Optional CQs for context retrieval
        table_store_name: Optional table store for context retrieval
        rag_type: Retrieval type ('vector', 'bm25', 'hybrid')
    
    Returns:
        List of all generated DPs
    """
    classes = final_tbox.get("classes", [])
    if not classes:
        logger.warning("No classes in TBox for DP generation")
        return []
    
    # Separate root class from subclasses
    root_classes = [c for c in classes if c.get("name") in ("Column", "upo:Column")]
    subclasses = [c for c in classes if c.get("name") not in ("Column", "upo:Column")]
    
    all_dps = []
    
    # Add hardcoded DP for root class
    if root_classes:
        logger.info(f"  Using hardcoded DP for root class Column")
        all_dps.append(ROOT_CLASS_DATAPROPERTY.copy())
    
    if not subclasses:
        return all_dps
    
    logger.info(f"  Generating DPs for {len(subclasses)} subclasses (max_concurrent={max_concurrent}, rag_type={rag_type})")
    
    # ========== Pre-compute retrieval data based on rag_type ==========
    precomputed_cqs = None
    precomputed_tables = None
    embedder_func = None
    
    if all_cqs or table_store_name:
        try:
            # Only load BGE-M3 for vector/hybrid mode
            if rag_type in ("vector", "hybrid"):
                from store.embedding.embedding_registry import get_registry
                registry = get_registry()
                embedder = registry.register_function("bge-m3")
                embedder_func = embedder.compute_source_embeddings
            
            if all_cqs:
                precomputed_cqs = _precompute_cq_data(all_cqs, rag_type=rag_type)
                logger.info(f"    Pre-computed CQ retrieval data: {len(all_cqs)} CQs (mode={rag_type})")
            if table_store_name:
                precomputed_tables = _precompute_table_data(table_store_name, rag_type=rag_type)
                if precomputed_tables:
                    logger.info(f"    Pre-loaded table data: {len(precomputed_tables['tables'])} tables (mode={rag_type})")
        except Exception as e:
            logger.warning(f"Failed to pre-compute retrieval data: {e}")
    
    llm = get_llm_by_purpose(purpose=llm_purpose, override_config=llm_override_config)
    
    async def batch_generate():
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            _generate_dp_async(
                cls, llm, semaphore, 
                precomputed_cqs, precomputed_tables, embedder_func
            )
            for cls in subclasses
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    # Run async batch
    try:
        results = asyncio.run(batch_generate())
    except RuntimeError:
        # Already in async context, use different approach
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(batch_generate())
    
    # Flatten results, handle exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            cls = subclasses[i]
            class_name = cls.get('name', 'Unknown')
            logger.error(f"  DP generation failed for {class_name}: {result}")
            # Add fallback DP with auto-generated template
            fallback_dp = {
                "property_name": f"has{class_name.replace('Column', '')}Value",
                "description": f"Value for {class_name}",
                "domain": [class_name],
                "range_type": "xsd:string",
                "functional": True,
            }
            all_dps.append(_auto_generate_single_template(fallback_dp))
        else:
            all_dps.extend(result)
    
    return all_dps


def final_dp_generation_node(state) -> Dict[str, Any]:
    """
    LangGraph node for final DP generation.
    
    Called after all class iterations are complete.
    Generates DPs for all classes in the final TBox.
    Uses retrieval (BGE-M3 or BM25 based on rag_type) to provide CQ + table context.
    
    Returns:
        Updated state with data_properties in current_tbox
    """
    from workflows.conceptualization.state import FederatedPrimitiveTBoxState
    from llm.statistics import set_current_caller
    set_current_caller("final_dp_generation_node")
    
    logger.info("Phase DP: Final DataProperty Generation")
    
    current_tbox = state.current_tbox
    if not current_tbox:
        logger.error("No TBox available for DP generation")
        return {
            "phase_errors": {"phase_dp": ["No TBox available"]},
        }
    
    classes = current_tbox.get("classes", [])
    logger.info(f"  Generating DPs for {len(classes)} classes")
    
    # Get CQs and table store for context retrieval
    all_cqs = getattr(state, 'competency_questions', []) or getattr(state, 'backbone_cqs', [])
    table_store_name = getattr(state, 'table_store_name', None)
    
    # Get max concurrent and rag_type from state
    dp_max_concurrent = getattr(state, 'dp_max_concurrent', 5)
    rag_type = getattr(state, 'rag_type', 'vector')
    
    if all_cqs:
        logger.info(f"    CQ context available: {len(all_cqs)} CQs")
    if table_store_name:
        logger.info(f"    Table context available: {table_store_name}")
    logger.info(f"    DP Max Concurrent: {dp_max_concurrent}, RAG Type: {rag_type}")
    
    # Use parallel generation with context
    all_dps = generate_all_dp_parallel(
        final_tbox=current_tbox,
        llm_purpose=state.llm_purpose,
        llm_override_config=state.llm_override_config,
        max_concurrent=dp_max_concurrent,
        all_cqs=all_cqs,
        table_store_name=table_store_name,
        rag_type=rag_type,
    )
    
    # Update TBox with DPs
    updated_tbox = {
        **current_tbox,
        "data_properties": all_dps,
    }
    
    logger.info(f"  ✓ Generated {len(all_dps)} DataProperties for {len(classes)} classes")
    
    # Log readout template statistics
    with_templates = sum(1 for dp in all_dps if dp.get("readout_template"))
    logger.info(f"    With readout templates: {with_templates}/{len(all_dps)}")
    
    # Verify coverage
    dp_domains = set()
    for dp in all_dps:
        dp_domains.update(dp.get("domain", []))
    
    missing = [cls.get("name") for cls in classes if cls.get("name") not in dp_domains]
    if missing:
        logger.warning(f"  Classes without DPs: {missing}")
    else:
        logger.info(f"    All {len(classes)} classes have DataProperties")
    
    return {
        "current_tbox": updated_tbox,
    }
