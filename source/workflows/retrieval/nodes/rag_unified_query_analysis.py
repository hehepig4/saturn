"""
Node: RAG-Enhanced Unified Query Analysis

Enhances unified query analysis with RAG (Retrieval-Augmented Generation):
1. First retrieves top-K similar tables using raw query
2. Uses retrieved table descriptions as style reference in prompt
3. LLM generates hypothetical descriptions following the actual index style

Benefits:
- table_description: Learns real table description style
- column_descriptions: Learns format "[TypeChain] ColName: Desc || ..."
- column_types: Sees which TBox types are actually used
- Better alignment between generated descriptions and indexed content

Prompt Structure (Prefix Cache Friendly):
- Static prefix: System instructions, TBox schema, value specs, output format
- Dynamic suffix: User query + RAG context (similar tables)

Note on column_type validation:
- Uses dynamic Pydantic model with Literal constraint
- Ensures LLM can ONLY output valid TBox class names
- Same approach as classify_columns.py for consistency
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Type
from pydantic import BaseModel, Field, create_model
from loguru import logger

from llm.invoke_with_stats import invoke_structured_llm_with_retry
from llm.async_client import invoke_structured_llm_with_retry_async

from utils.primitive_class_utils import remove_primitive_class_markers
from workflows.retrieval.matcher.constraints import (
    PathConstraint, ConstraintSet, TBoxConstraint, ABoxConstraint
)


# ==================== Helper Functions ====================

def _merge_column_desc_and_stats(col_descriptions: str, col_stats: str) -> str:
    """
    Merge column descriptions and stats into a unified per-column format.
    
    Input formats (from generate_summaries.py, ' || ' separated):
        col_descriptions: "[Type1] Col1: Desc1 || [Type2] Col2: Desc2 || ..."
        col_stats: "Col1: stats1 || Col2: stats2 || ..."
    
    Output format (one line per column, desc and stats concatenated with space):
        "[Type1] Col1: Desc1 stats1"
        "[Type2] Col2: Desc2 stats2"
    """
    SEP = ' || '
    
    if not col_descriptions and not col_stats:
        return ""
    
    if not col_stats:
        return '\n'.join(part.strip() for part in col_descriptions.split(SEP) if part.strip())
    
    if not col_descriptions:
        lines = []
        for part in col_stats.split(SEP):
            part = part.strip()
            if part:
                lines.append(f"Stats: {part}")
        return '\n'.join(lines)
    
    # Build mapping from column name to stats
    stats_map = {}
    for part in col_stats.split(SEP):
        part = part.strip()
        if not part or ':' not in part:
            continue
        col_name, _, stats_content = part.partition(':')
        col_name = col_name.strip()
        stats_content = stats_content.strip()
        if col_name and stats_content:
            stats_map[col_name] = stats_content
    
    # Merge descriptions with stats
    merged_lines = []
    for part in col_descriptions.split(SEP):
        part = part.strip()
        if not part:
            continue
        
        col_name = None
        if '] ' in part:
            after_bracket = part.split('] ', 1)[1] if '] ' in part else part
            if ':' in after_bracket:
                col_name = after_bracket.split(':')[0].strip()
        elif ':' in part:
            col_name = part.split(':')[0].strip()
        
        if col_name and col_name in stats_map:
            merged_lines.append(f"{part} {stats_map[col_name]}")
        else:
            merged_lines.append(part)
    
    return '\n'.join(merged_lines)


# ==================== Output Schema ====================
# Note: ColumnAnalysis and UnifiedQueryAnalysis are dynamically created
# with Literal constraint for column_type to ensure valid TBox classes.
# See _create_constrained_analysis_schema() below.

# Fallback static schema for type hints only (not used at runtime)
class ColumnAnalysis(BaseModel):
    """Per-column analysis for both semantic and structural matching."""
    column_type: str  # Primitive class name from TBox (e.g., 'YearColumn')
    description: str  # Column content description
    value: Optional[str] = None  # Explicit value from query (standard format)


class UnifiedQueryAnalysis(BaseModel):
    """Unified query analysis output."""
    table_description: str  # Table description for semantic matching
    columns: List[ColumnAnalysis]  # Columns needed for the query


# Schema for no-primitive-classes ablation
class ColumnAnalysisNoPC(BaseModel):
    """Per-column analysis WITHOUT primitive class types (for ablation)."""
    description: str  # Column content description


class UnifiedQueryAnalysisNoPC(BaseModel):
    """Unified query analysis output WITHOUT primitive classes (for ablation)."""
    table_description: str  # Table description for semantic matching
    columns: List[ColumnAnalysisNoPC]  # Columns needed for the query


# ==================== Dynamic Schema Creation ====================

def _create_constrained_analysis_schema(
    valid_class_names: List[str],
) -> Type[BaseModel]:
    """
    Create a UnifiedQueryAnalysis schema with Literal constraint on column_type.
    
    This ensures LLM can ONLY output valid TBox class names, preventing
    hallucinated classes like 'StateProvinceIDColumn' that don't exist.
    
    Args:
        valid_class_names: List of valid TBox class names from the ontology
        
    Returns:
        ConstrainedUnifiedQueryAnalysis Pydantic model class
    """
    from typing import Literal
    
    if not valid_class_names:
        # Fallback to unconstrained schema if no classes provided
        logger.warning("No valid class names provided, using unconstrained schema")
        return UnifiedQueryAnalysis
    
    # Create Literal type for valid class names
    ClassLiteral = Literal[tuple(valid_class_names)]
    
    # Create constrained ColumnAnalysis
    ConstrainedColumnAnalysis = create_model(
        'ConstrainedColumnAnalysis',
        column_type=(ClassLiteral, ...),  # Primitive class name from TBox
        description=(str, ...),  # Column content description (include value if extracted)
        value=(Optional[str], None),  # Explicit value from query
    )
    # Create constrained UnifiedQueryAnalysis
    ConstrainedUnifiedQueryAnalysis = create_model(
        'ConstrainedUnifiedQueryAnalysis',
        table_description=(str, ...),  # Table description for HyDE
        columns=(List[ConstrainedColumnAnalysis], ...),  # Required columns
    )
    return ConstrainedUnifiedQueryAnalysis


# ==================== Prompt Templates ====================

# Static prefix - can be cached by LLM providers (with primitive classes)
RAG_PROMPT_PREFIX = """You are helping a table retrieval system find the right database table to answer a user's question.

The system has two retrieval paths:
1. **Semantic path**: Matches table descriptions using text similarity
2. **Structural path**: Matches column types and specific values

Your job is to generate both:
- A hypothetical table description (for semantic matching)
- Column type constraints with optional values (for structural matching)

## Available Column Types (TBox Schema)
Below is the hierarchy of available column types with their descriptions.
Indented types are more specific subtypes. Use ONLY these types.

{tbox_schema}

## Value Format Specifications
When extracting values, convert to the standard format shown below:

{data_property_specs}

## Output Format (JSON)
```json
{{
    "table_description": "<hypothetical table description>",
    "columns": [
        {{"column_type": "<TBox class name>", "description": "<IMITATE from Similar Tables>", "value": "<extracted value or null>"}}
    ]
}}
```

## Important Rules
- Include only columns that are necessary to answer the query
- Prefer specific subtypes over general parent types when they match better
- Extract values only if they are explicitly stated in the query (do not infer)
- Use standard format from Value Format Specifications for extracted values

"""

# Static prefix - WITHOUT primitive classes (for ablation)
RAG_PROMPT_PREFIX_NO_PC = """You are helping a table retrieval system find the right database table to answer a user's question.

The system uses **semantic matching** to find tables based on text similarity.

Your job is to generate:
- A hypothetical table description (what kind of table would answer this query)
- Column descriptions (what information the columns should contain)

## Output Format (JSON)
```json
{{
    "table_description": "<hypothetical table description>",
    "columns": [
        {{"description": "<column content description>"}}
    ]
}}
```

## Important Rules
- Include only columns that are necessary to answer the query
- If the query mentions specific values, include them in the description

"""

# Dynamic suffix - changes per query (with primitive classes)
RAG_PROMPT_SUFFIX = """
## User Query
{query}

## Similar Tables from Index (Reference Style)
The following tables are retrieved from the index for similar queries.

{similar_tables}

## How to Generate `description` Field
**CRITICAL**: Your `description` field must IMITATE the style from Similar Tables above.

Look at the **Column Descriptions** in Similar Tables. They follow this format:
`[TypeChain] ColumnName: Semantic description of the column content`
where TypeChain may be a hierarchy chain like `A > B > C`.

Your task:
1. Find a column with similar `column_type` in Similar Tables
2. **Copy and adapt** its description style for your hypothetical column
3. If the query contains a specific value, append it in parentheses: `"description": "...description... ('specific value')"`
4. **IMPORTANT**: For `column_type`, output ONLY the leaf class name (e.g., `YearColumn`), NOT the hierarchy chain

Example adaptation:
- Similar Table has: `[Temporal > Year] Year: The year the film was released`
- Query asks about year 2015
- Your column_type: `"YearColumn"` (leaf class name only, not `Temporal > Year`)
- Your description: `"The year the production was released (2015)"`

## Generate
Based on the user query and **imitating the description style** from Similar Tables:

Output JSON only:"""

# Dynamic suffix - WITHOUT primitive classes (for ablation)
RAG_PROMPT_SUFFIX_NO_PC = """
## User Query
{query}

## Similar Tables from Index (Reference Style)
The following tables are retrieved from the index for similar queries.

{similar_tables}

## How to Generate Descriptions
Look at the **Column Descriptions** in Similar Tables and imitate their style.

If the query contains specific values (names, dates, numbers), include them in the column descriptions.

## Generate
Based on the user query and **imitating the style** from Similar Tables:

Output JSON only:"""


# ==================== RAG Type Constants ====================

RAG_TYPE_BM25 = "bm25"
RAG_TYPE_VECTOR = "vector"
RAG_TYPE_HYBRID = "hybrid"

RAG_TYPES = [RAG_TYPE_BM25, RAG_TYPE_VECTOR, RAG_TYPE_HYBRID]


# ==================== RAG Retrieval Functions ====================

def retrieve_similar_tables(
    query: str,
    dataset: str,
    top_k: int = 3,
    rag_type: str = RAG_TYPE_HYBRID,
    use_primitive_classes: bool = True,
    index_base_path: Path = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K similar tables using raw query.
    
    Args:
        query: User query text
        dataset: Dataset identifier
        top_k: Number of similar tables to retrieve
        rag_type: Retrieval type - "bm25", "vector", or "hybrid" (default)
        use_primitive_classes: If False, use ablated index without [Type] markers
        index_base_path: Base path for indexes. If provided, loads from
            {index_base_path}/indexes/{index_key}/
    
    Returns table metadata including:
    - table_id
    - table_description
    - column_descriptions
    - column_types (empty if use_primitive_classes=False)
    """
    from workflows.retrieval.unified_search import unified_search
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    
    # Select index key based on ablation mode
    index_key = INDEX_KEY_TD_CD_CS
    if not use_primitive_classes:
        index_key = f"{INDEX_KEY_TD_CD_CS}_no_pc"
    
    # Use unified search
    results = unified_search(
        query=query,
        dataset_name=dataset,
        top_k=top_k,
        rag_type=rag_type,
        index_key=index_key,
        index_base_path=index_base_path,
    )
    
    # Extract metadata
    similar_tables = []
    for tid, score, meta in results[:top_k]:
        similar_tables.append({
            'table_id': tid,
            'table_description': meta.get('table_description', '') if meta else '',
            'column_descriptions': meta.get('column_descriptions', '') if meta else '',
            'column_stats': meta.get('column_stats', '') if meta else '',
            'column_types': meta.get('column_types', []) if meta else [],
            'score': score,
        })
    
    return similar_tables


def format_similar_tables(
    tables: List[Dict[str, Any]],
    use_primitive_classes: bool = True,
) -> str:
    """Format similar tables for prompt.
    
    Shows the index document structure to help LLM understand the target format:
        == Table Description ==
        {description}
        
        == Column Information ==
        [Type] ColName: Description stats
        ...
    
    Uses longer truncation lengths (2000 chars) to provide more style reference context.
    
    Args:
        tables: List of similar table metadata dicts
        use_primitive_classes: If False, remove [Type] markers from column info
            and skip Column Types Used line (ablation mode)
    """
    if not tables:
        return "(No similar tables found)"
    
    lines = []
    for i, t in enumerate(tables, 1):
        lines.append(f"### Table {i}: \"{t['table_id']}\"")
        
        # Format as structured index document
        lines.append("```")
        lines.append("== Table Description ==")
        lines.append(t['table_description'] or "(no description)")
        
        # Column information with types
        col_desc = t.get('column_descriptions', '')
        col_stats = t.get('column_stats', '')
        
        if col_desc or col_stats:
            lines.append("")
            lines.append("== Column Information ==")
            
            # Merge descriptions and stats per column (use local function)
            if col_desc:
                merged_cols = _merge_column_desc_and_stats(col_desc, col_stats)
                
                # Remove [Type] markers in ablation mode
                if not use_primitive_classes:
                    merged_cols = remove_primitive_class_markers(merged_cols)
                
                # Truncate if too long
                if len(merged_cols) > 2000:
                    merged_cols = merged_cols[:2000] + "..."
                lines.append(merged_cols)
            elif col_stats:
                # Only stats available
                if len(col_stats) > 2000:
                    col_stats = col_stats[:2000] + "..."
                lines.append(col_stats)
        
        lines.append("```")
        
        # Column types (for TBox reference) - skip in ablation mode
        if use_primitive_classes:
            col_types = t.get('column_types', [])
            if hasattr(col_types, 'tolist'):
                col_types = col_types.tolist()
            if len(col_types) > 0:
                types_str = ', '.join(str(ct) for ct in col_types[:10])
                if len(col_types) > 10:
                    types_str += f"... (+{len(col_types)-10} more)"
                lines.append(f"**Column Types Used**: {types_str}")
        
        lines.append("")
    
    return '\n'.join(lines)


# ==================== Processing Functions ====================

def process_unified_analysis(analysis, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert UnifiedQueryAnalysis (or UnifiedQueryAnalysisNoPC) to retrieval-ready format.
    
    Builds hypothetical table/column descriptions for HyDE semantic matching:
        - table_description: Hypothetical table description
        - column_descriptions: Formatted as [TypeChain] TypeName: Desc || value: extracted_value
    
    Handles both with and without primitive classes modes.
    
    Args:
        analysis: UnifiedQueryAnalysis Pydantic model from LLM
        dataset_name: Dataset identifier for loading ontology chain map (optional)
    """
    # Check if this is the no-primitive-classes version
    has_primitive_classes = hasattr(analysis.columns[0], 'column_type') if analysis.columns else False
    
    if has_primitive_classes:
        # Build TBox constraints (class names only)
        tbox_constraints = [col.column_type for col in analysis.columns]
        
        # Build ABox constraints (values only, with their types)
        abox_constraints = []
        for col in analysis.columns:
            if hasattr(col, 'value') and col.value is not None:
                abox_constraints.append({
                    'column_type': col.column_type,
                    'value': col.value,
                })
        
        # Build hypothetical descriptions (for HyDE semantic matching)
        # Format matches index document (space-concatenated)
        hypothetical_table_description = analysis.table_description
        col_descriptions = []
        for col in analysis.columns:
            col_type = col.column_type.replace('Column', '')
            # Use leaf class name for display
            col_line = f"[{col_type}] {col_type}: {col.description}"
            if hasattr(col, 'value') and col.value is not None:
                col_line += f" value: {col.value}"
            col_descriptions.append(col_line)
        hypothetical_column_descriptions = '\n'.join(col_descriptions)
        
        return {
            'hypothetical_table_description': hypothetical_table_description,
            'hypothetical_column_descriptions': hypothetical_column_descriptions,
            'tbox_constraints': tbox_constraints,
            'abox_constraints': abox_constraints,
            'columns_raw': [col.model_dump() for col in analysis.columns],
        }
    else:
        # No primitive classes mode - simpler output
        hypothetical_table_description = analysis.table_description
        col_descriptions = [col.description for col in analysis.columns]
        hypothetical_column_descriptions = '\n'.join(col_descriptions)
        
        return {
            'hypothetical_table_description': hypothetical_table_description,
            'hypothetical_column_descriptions': hypothetical_column_descriptions,
            'tbox_constraints': [],  # No TBox constraints in ablation mode
            'abox_constraints': [],  # No ABox constraints in ablation mode
            'columns_raw': [col.model_dump() for col in analysis.columns],
        }


# ==================== Helper Functions ====================

def format_class_hierarchy(classes: List[Dict[str, Any]]) -> str:
    """Format class hierarchy using the unified tree formatter.
    
    Uses utils.tree_formatter for consistency with other parts of the codebase.
    Description is not truncated to provide full context for query analysis.
    """
    from utils.tree_formatter import format_class_hierarchy as _format_tree
    # max_desc_length=None means no truncation
    return _format_tree(classes, include_column_root=True, max_desc_length=None)


def get_available_primitive_classes(dataset_name: str) -> List[Dict[str, Any]]:
    """Get available primitive classes with hierarchy info for a specific dataset."""
    try:
        from store.store_singleton import get_store
        store = get_store()
        
        try:
            om = store.db.open_table('ontology_metadata').to_pandas()
            fedtbox = om[(om['dataset_name'] == dataset_name) & 
                        (om['ontology_type'] == 'federated_primitive_tbox')]
            fedtbox = fedtbox.sort_values('created_at', ascending=False)
            
            if len(fedtbox) == 0:
                logger.warning(f"No federated_primitive_tbox found for {dataset_name}")
                return []
            
            latest_ontology_id = fedtbox.iloc[0]['ontology_id']
            logger.debug(f"Using ontology: {latest_ontology_id}")
        except Exception as e:
            logger.warning(f"Failed to find latest ontology for {dataset_name}: {e}")
            return []
        
        try:
            oc = store.db.open_table('ontology_classes').to_pandas()
            oc = oc[oc['ontology_id'] == latest_ontology_id]
            
            classes = []
            for _, row in oc.iterrows():
                name = row['class_name'].replace('upo:', '')
                parent_classes = row.get('parent_classes', [])
                
                if parent_classes is None:
                    parent_classes = []
                elif hasattr(parent_classes, 'tolist'):
                    parent_classes = parent_classes.tolist()
                elif isinstance(parent_classes, str):
                    parent_classes = [parent_classes] if parent_classes else []
                
                classes.append({
                    'name': name,
                    'description': row.get('description', ''),
                    'parent_classes': parent_classes,
                })
            
            return classes
        except Exception as e:
            logger.warning(f"Failed to load ontology classes: {e}")
            return []
            
    except Exception as e:
        logger.warning(f"Failed to get primitive classes: {e}")
        return []


def get_data_property_specs(class_names: List[str], dataset_name: str) -> str:
    """Get data property formatting specifications."""
    from workflows.retrieval.nodes.unified_query_analysis import get_data_property_specs as _get_specs
    return _get_specs(class_names, dataset_name)


# ==================== Main Implementation ====================

async def _rag_unified_query_analysis_impl(
    query: str,
    dataset_name: str,
    llm_purpose: str,
    rag_top_k: int,
    rag_type: str = RAG_TYPE_HYBRID,
    use_primitive_classes: bool = True,
    index_base_path: Path = None,
) -> UnifiedQueryAnalysis:
    """
    RAG-enhanced implementation of unified query analysis.
    
    Args:
        query: User query text
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        llm_purpose: LLM purpose key from llm_models.json (e.g., 'gemini', 'default')
        rag_top_k: Number of similar tables to retrieve for RAG context
        rag_type: Retrieval type for RAG - "bm25", "vector", or "hybrid" (default)
        use_primitive_classes: Whether to include primitive class types (default True)
        index_base_path: Base path for indexes (optional, for experiment isolation)
    """
    from llm.manager import get_llm_by_purpose
    
    # Step 1: Retrieve similar tables
    logger.debug(f"Retrieving top-{rag_top_k} similar tables (rag_type={rag_type}, use_pc={use_primitive_classes})...")
    start_time = time.time()
    similar_tables = retrieve_similar_tables(
        query, dataset_name, top_k=rag_top_k, rag_type=rag_type,
        use_primitive_classes=use_primitive_classes,
        index_base_path=index_base_path,
    )
    retrieve_time = time.time() - start_time
    logger.debug(f"Retrieved {len(similar_tables)} tables in {retrieve_time:.2f}s")
    
    # Step 2: Build prompt
    # Format similar tables (pass use_primitive_classes for ablation support)
    similar_tables_text = format_similar_tables(similar_tables, use_primitive_classes=use_primitive_classes)
    
    if use_primitive_classes:
        # Full prompt with TBox schema and data property specs
        primitive_classes = get_available_primitive_classes(dataset_name)
        tbox_schema = format_class_hierarchy(primitive_classes)
        class_names = [c['name'] for c in primitive_classes]
        data_property_specs = get_data_property_specs(class_names, dataset_name)
        
        prefix = RAG_PROMPT_PREFIX.format(
            tbox_schema=tbox_schema,
            data_property_specs=data_property_specs,
        )
        suffix = RAG_PROMPT_SUFFIX.format(
            query=query,
            similar_tables=similar_tables_text,
        )
        
        # Create constrained schema with valid TBox class names
        ConstrainedSchema = _create_constrained_analysis_schema(class_names)
    else:
        # Simplified prompt WITHOUT primitive classes (ablation)
        prefix = RAG_PROMPT_PREFIX_NO_PC
        suffix = RAG_PROMPT_SUFFIX_NO_PC.format(
            query=query,
            similar_tables=similar_tables_text,
        )
        
        # Use unconstrained schema without column_type
        ConstrainedSchema = UnifiedQueryAnalysisNoPC
    
    full_prompt = prefix + suffix
    
    # Step 3: Call LLM with structured output using EBNF grammar
    logger.debug(f"Calling LLM with RAG-enhanced prompt (purpose={llm_purpose}, use_pc={use_primitive_classes})...")
    
    def llm_factory(temperature: float):
        return get_llm_by_purpose(llm_purpose, temperature_override=temperature)
    
    # Use async version to avoid blocking event loop when called with asyncio.gather()
    result = await invoke_structured_llm_with_retry_async(
        llm_factory=llm_factory,
        output_schema=ConstrainedSchema,
        prompt=full_prompt,
        max_retries=2,
        # timeout uses TruncationLimits.LLM_TIMEOUT default
    )
    
    if result is None:
        raise ValueError("LLM returned None")
    
    return result
