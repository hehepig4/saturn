"""
Node: Unified Query Analysis (TBox-HyDE)

Single LLM call that:
1. Analyzes query to identify required column types (TBox constraints)
2. Extracts explicit values from query (ABox constraints)
3. Generates hypothetical table/column descriptions for HyDE-style semantic matching

The output serves dual purpose:
- Semantic path: hypothetical descriptions for FAISS/BM25 matching
- Structural path: PathConstraints for Scorer V3 matching

Design Decision:
- Merge TBox+ABox extraction and HyDE generation into ONE LLM call
- Reuse `reasoning` field as both explanation AND hypothetical description
- Save LLM tokens by avoiding redundant calls
"""

import json
from typing import Dict, Any, List, Optional, Tuple, Type
from pydantic import BaseModel, Field, create_model
from loguru import logger

from llm.invoke_with_stats import invoke_structured_llm_with_retry
from workflows.common.node_decorators import graph_node
from workflows.retrieval.matcher.constraints import (
    PathConstraint, ConstraintSet, TBoxConstraint, ABoxConstraint
)


# ==================== Output Schema ====================

class ColumnAnalysis(BaseModel):
    """
    Per-column analysis serving dual purpose:
    - description → hypothetical column description for HyDE semantic matching
    - column_type → TBox constraint for Scorer V3 structural matching
    - value → ABox constraint (if explicit value present in query)
    """
    column_type: str  # Primitive class name from TBox (e.g., 'YearColumn')
    description: str  # Column content description (include extracted value if any)
    value: Optional[str] = None  # Explicit value from query (standard format)

class UnifiedQueryAnalysis(BaseModel):
    """
    Unified query analysis output.
    
    Serves dual purpose:
    - table_description/columns[].description → HyDE embedding input
    - columns[].column_type/value → Scorer V3 constraint input
    """
    table_description: str  # 1-2 sentence description of the table needed
    columns: List[ColumnAnalysis]  # Columns needed to answer the query


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
        description=(str, ...),  # Column content description
        value=(Optional[str], None),  # Explicit value from query
    )
    
    # Create constrained UnifiedQueryAnalysis
    ConstrainedUnifiedQueryAnalysis = create_model(
        'ConstrainedUnifiedQueryAnalysis',
        table_description=(str, ...),  # Table description for HyDE
        columns=(List[ConstrainedColumnAnalysis], ...),  # Required columns
    )
    
    return ConstrainedUnifiedQueryAnalysis


# ==================== Prompt ====================

UNIFIED_QUERY_ANALYSIS_PROMPT = """You are helping a table retrieval system find the right database table to answer a user's question.

The system has two retrieval paths:
1. **Semantic path**: Matches table descriptions using text similarity
2. **Structural path**: Matches column types and specific values

Your job is to generate both:
- A hypothetical table description (for semantic matching)
- Column type constraints with optional values (for structural matching)

## Query
{query}

## Available Column Types (TBox Schema)
Below is the hierarchy of available column types with their descriptions.
Indented types are more specific subtypes. Use ONLY these types.

{tbox_schema}

## Value Format Specifications
When extracting values, convert to the standard format shown below:

{data_property_specs}

## Task
Generate the following:

1. **table_description**: Describe what the target table should contain.
   - Write 1-2 sentences like a real database table description
   - Critical: Include ALL specific entity names from the query (person names, organization names, work titles, place names, event names, etc.)
   - Tables are often named after their main subject (e.g., "Blake Hood , Filmography" or "Ben Platt (actor) , Theatre credits")
   - Good examples:
     - Query about Blake Hood → "Blake Hood's filmography listing roles in TV shows including 90210."
     - Query about Ben Platt → "Ben Platt's theatrical performances and credits."
   - Bad examples (missing entities!):
     - "Actors and their filmography." (missing "Blake Hood")
     - "Theatrical performances at various venues." (missing "Ben Platt")

2. **columns**: List each column type needed to answer the query.
   - **column_type**: Use exact names from Column Types above (e.g., "YearColumn")
   - **description**: Brief description of what this column stores
   - **value**: If query mentions a specific value for this column, extract it in standard format. Otherwise set null.

## Important Rules
- Include only columns that are ncecessary to answer the query
- Prefer specific subtypes over general parent types when they match better
- Extract values only if they are explicitly stated in the query (do not infer)
- Use standard format from Value Format Specifications for extracted values
- **CRITICAL: When a value is extracted, ALWAYS include it in the description field**
  - Good: `"description": "The album title ('Most Messed Up')"`
  - Bad: `"description": "The album title"` (missing value!)
  - This ensures the description can be matched with table content during retrieval

## Output Format (JSON)
```json
{{
    "table_description": "Blake Hood's television and film roles, including appearances in 90210",
    "columns": [
        {{"column_type": "PersonNameColumn", "description": "Actor name ('Blake Hood')", "value": "blake hood"}},
        {{"column_type": "CreativeWorkColumn", "description": "TV show or film title ('90210')", "value": "90210"}},
        {{"column_type": "CharacterRoleColumn", "description": "Character portrayed by the actor", "value": null}}
    ]
}}
```"""


# ==================== Helper Functions ====================

def format_class_hierarchy(classes: List[Dict[str, Any]], indent: int = 0) -> str:
    """
    Format class hierarchy as indented tree structure.
    
    Reuses logic from extract_constraints.py.
    """
    # Build parent -> children mapping
    children_map: Dict[str, List[Dict]] = {}
    roots = []
    
    for cls in classes:
        name = cls['name']
        parents = cls.get('parent_classes', [])
        
        if not parents:
            roots.append(cls)
        else:
            for parent in parents:
                parent_name = parent.replace('upo:', '')
                if parent_name not in children_map:
                    children_map[parent_name] = []
                children_map[parent_name].append(cls)
    
    def build_tree(node: Dict, level: int = 0) -> List[str]:
        """Recursively build tree lines."""
        name = node['name']
        desc = node.get('description', '')
        
        # Format: indentation + name + description
        prefix = "  " * level
        if desc:
            line = f"{prefix}- **{name}**: {desc}"
        else:
            line = f"{prefix}- **{name}**"
        
        lines = [line]
        
        # Add children
        for child in children_map.get(name, []):
            lines.extend(build_tree(child, level + 1))
        
        return lines
    
    result_lines = []
    for root in roots:
        result_lines.extend(build_tree(root))
    
    return "\n".join(result_lines) if result_lines else "No column types available."


def get_available_primitive_classes(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Get available primitive classes with hierarchy info for a specific dataset.
    
    Reuses logic from extract_constraints.py.
    """
    try:
        from store.store_singleton import get_store
        store = get_store()
        
        # Find the latest federated_primitive_tbox for this dataset
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
        
        # Get classes from ontology_classes filtered by ontology_id
        try:
            oc = store.db.open_table('ontology_classes').to_pandas()
            oc = oc[oc['ontology_id'] == latest_ontology_id]
            
            classes = []
            for _, row in oc.iterrows():
                name = row['class_name'].replace('upo:', '')
                parent_classes = row.get('parent_classes', [])
                
                # Normalize parent_classes
                if parent_classes is None:
                    parent_classes = []
                elif isinstance(parent_classes, str):
                    parent_classes = [parent_classes] if parent_classes and parent_classes != 'null' else []
                elif hasattr(parent_classes, 'tolist'):
                    parent_classes = parent_classes.tolist()
                
                # Filter out 'null' strings
                parent_classes = [p for p in parent_classes if p and p != 'null']
                
                classes.append({
                    'name': name,
                    'parent_classes': parent_classes,
                    'description': row.get('description', ''),
                })
            
            return classes
        except Exception as e:
            logger.warning(f"Failed to get classes from ontology_classes: {e}")
            return []
        
    except Exception as e:
        logger.warning(f"Failed to get primitive classes: {e}")
        return []


def get_data_property_specs(class_names: List[str], dataset_name: str) -> str:
    """
    Get data property format specifications for value extraction guidance.
    
    Args:
        class_names: List of all available primitive class names
        dataset_name: Dataset name for property lookup
        
    Returns:
        Formatted string with format specifications
    """
    from core.datatypes.el_datatypes import DATATYPE_SPECS
    
    try:
        from store.store_singleton import get_store
        store = get_store()
        
        # Get class to data range mapping
        class_ranges: Dict[str, str] = {}
        try:
            # Get data properties with their domains and ranges
            dp_df = store.db.open_table('ontology_properties').to_pandas()
            for _, row in dp_df.iterrows():
                prop_type = row.get('property_type', '')
                if prop_type != 'data':
                    continue
                
                # Domain is the class, range is the XSD type
                domain = row.get('domain', '')
                # Handle numpy array or list
                if hasattr(domain, 'tolist'):
                    domain = domain.tolist()
                if isinstance(domain, list):
                    domain = domain[0] if domain else ''
                if domain and isinstance(domain, str):
                    domain = domain.replace('upo:', '')
                    
                range_type = row.get('range', 'xsd:string')
                if hasattr(range_type, 'tolist'):
                    range_type = range_type.tolist()
                if isinstance(range_type, list):
                    range_type = range_type[0] if range_type else 'xsd:string'
                
                if domain:
                    class_ranges[domain] = range_type
        except Exception as e:
            logger.debug(f"Failed to get data property ranges: {e}")
        
        # Build format specs for classes that have specific formats
        lines = []
        for name in class_names:
            data_range = class_ranges.get(name, 'xsd:string')
            spec = DATATYPE_SPECS.get(data_range, DATATYPE_SPECS.get('xsd:string'))
            
            if spec:
                examples = ', '.join(spec.example_values[:3])
                lines.append(f"- **{name}**: Format: {spec.standard_format}, Examples: {examples}")
        
        return "\n".join(lines) if lines else "Use standard string format for all values."
        
    except Exception as e:
        logger.warning(f"Failed to get data property specs: {e}")
        return "Use standard string format for all values."


# ==================== Output Processing ====================

def process_unified_analysis(
    result: UnifiedQueryAnalysis,
    available_classes: List[str],
) -> Dict[str, Any]:
    """
    Process unified output for both semantic and structural paths.
    
    The reasoning field is reused:
    - For HyDE: reasoning serves as the hypothetical column description
    - For Scorer V3: column_type and value derive PathConstraints
    
    Args:
        result: LLM output as UnifiedQueryAnalysis
        available_classes: List of valid class names for validation
        
    Returns:
        Dict with:
        - hypothetical_table_description: str (for HyDE semantic matching)
        - hypothetical_column_descriptions: str (for HyDE semantic matching)
        - constraint_set: ConstraintSet (for Scorer V3)
        - columns_raw: List[Dict] (for debugging)
    """
    # === Semantic Path (HyDE) ===
    
    # table_description → hypothetical_table_description
    hypothetical_table_description = result.table_description
    
    # columns[].description → hypothetical_column_descriptions
    # Format matches index document (space-concatenated):
    #   [Type] TypeName: Description value: extracted_value
    col_parts = []
    for col in result.columns:
        type_short = col.column_type.replace('Column', '')
        col_line = f"[{type_short}] {type_short}: {col.description}"
        if col.value is not None:
            col_line += f" value: {col.value}"
        col_parts.append(col_line)
    hypothetical_column_descriptions = '\n'.join(col_parts)
    
    # === Structural Path (Scorer V3) ===
    
    # Derive PathConstraint list from columns
    path_constraints = []
    valid_class_set = set(available_classes)
    
    for col in result.columns:
        class_name = col.column_type
        
        # Validate class name
        if class_name not in valid_class_set:
            logger.warning(f"Invalid class name '{class_name}' - skipping")
            continue
        
        tbox = TBoxConstraint(class_name=class_name)
        abox = None
        
        if col.value:
            # Normalize value for BF query
            normalized = str(col.value).strip().lower()
            abox = ABoxConstraint(
                value=col.value,
                class_name=class_name,
                normalized_value=normalized,
            )
        
        path_constraints.append(PathConstraint(tbox=tbox, abox=abox))
    
    constraint_set = ConstraintSet(constraints=path_constraints)
    
    return {
        # For semantic path (HyDE)
        "hypothetical_table_description": hypothetical_table_description,
        "hypothetical_column_descriptions": hypothetical_column_descriptions,
        # For structural path (Scorer V3)
        "constraint_set": constraint_set,
        # Raw for debugging
        "columns_raw": [c.model_dump() for c in result.columns],
    }


# ==================== Serialization ====================

def serialize_unified_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize unified query analysis result for JSON storage.
    
    Converts ConstraintSet to serializable format while preserving all HyDE fields.
    
    Args:
        result: Output from process_unified_analysis()
        
    Returns:
        JSON-serializable dict
    """
    constraint_set = result.get("constraint_set")
    
    # Serialize constraint_set
    if constraint_set:
        tbox_constraints = [c.class_name for c in constraint_set.constraints]
        abox_constraints = [
            {"column_type": c.class_name, "value": c.value}
            for c in constraint_set.constraints
            if c.has_value
        ]
    else:
        tbox_constraints = []
        abox_constraints = []
    
    return {
        # HyDE fields (already serializable)
        "hypothetical_table_description": result.get("hypothetical_table_description", ""),
        "hypothetical_column_descriptions": result.get("hypothetical_column_descriptions", ""),
        # Constraint fields (converted from ConstraintSet)
        "tbox_constraints": tbox_constraints,
        "abox_constraints": abox_constraints,
        # Raw analysis (already serializable via model_dump)
        "columns_raw": result.get("columns_raw", []),
    }


# ==================== Core Logic (Testable) ====================

async def _unified_query_analysis_impl(
    query: str,
    dataset_name: str,
    llm_purpose: str = "default",
) -> Dict[str, Any]:
    """
    Core implementation of unified query analysis.
    
    Single LLM call that:
    1. Identifies required column types (TBox)
    2. Extracts explicit values (ABox)
    3. Generates hypothetical descriptions (HyDE)
    
    Args:
        query: Natural language query
        dataset_name: Dataset name for schema lookup
        llm_purpose: LLM purpose key for model selection
        
    Returns:
        Dict with:
        - hypothetical_table_description: For semantic path
        - hypothetical_column_descriptions: For semantic path
        - constraint_set: ConstraintSet for Scorer V3
        - query_analysis_raw: Raw columns for debugging
    """
    from llm.manager import get_llm_by_purpose
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Get available primitive classes
    classes = get_available_primitive_classes(dataset_name)
    if not classes:
        logger.warning(f"No primitive classes found for {dataset_name}")
        return {
            "hypothetical_table_description": query,
            "hypothetical_column_descriptions": "",
            "constraint_set": ConstraintSet(constraints=[]),
            "query_analysis_raw": [],
        }
    
    # Format TBox schema as hierarchy
    tbox_schema = format_class_hierarchy(classes)
    
    # Get class names for spec lookup
    class_names = [c['name'] for c in classes]
    
    # Get data property format specs
    data_property_specs = get_data_property_specs(class_names, dataset_name)
    
    # Build prompt
    prompt = UNIFIED_QUERY_ANALYSIS_PROMPT.format(
        query=query,
        tbox_schema=tbox_schema,
        data_property_specs=data_property_specs,
    )
    
    # Create constrained schema with valid TBox class names
    # This ensures LLM can ONLY output valid class names
    ConstrainedSchema = _create_constrained_analysis_schema(class_names)
    
    # Call LLM with structured output using EBNF grammar
    try:
        def llm_factory(temperature: float):
            return get_llm_by_purpose(llm_purpose, temperature_override=temperature)
        
        messages = [
            SystemMessage(content="You are a precise query analysis assistant."),
            HumanMessage(content=prompt)
        ]
        
        result = invoke_structured_llm_with_retry(
            llm_factory=llm_factory,
            output_schema=ConstrainedSchema,
            prompt=messages,
            max_retries=3,
            # timeout uses TruncationLimits.LLM_TIMEOUT default
        )
        
        if result is None:
            raise ValueError("LLM returned None")
        
        logger.debug(f"Unified query analysis result: {result}")
        
        # Process output for dual purpose
        processed = process_unified_analysis(result, class_names)
        
        return {
            "hypothetical_table_description": processed["hypothetical_table_description"],
            "hypothetical_column_descriptions": processed["hypothetical_column_descriptions"],
            "constraint_set": processed["constraint_set"],
            "query_analysis_raw": processed["columns_raw"],
        }
        
    except Exception as e:
        logger.error(f"Failed to run unified query analysis: {e}")
        # Fallback: use query as hypothetical description, no constraints
        return {
            "hypothetical_table_description": query,
            "hypothetical_column_descriptions": "",
            "constraint_set": ConstraintSet(constraints=[]),
            "query_analysis_raw": [],
        }


# ==================== Graph Node (for LangGraph) ====================

@graph_node(node_type="processing", log_level="INFO")
async def unified_query_analysis_node(state) -> Dict[str, Any]:
    """
    LangGraph node wrapper for unified query analysis.
    
    Extracts query and dataset_name from state and calls the implementation.
    
    Args:
        state: LangGraph state with 'query' and 'dataset_name' fields
        
    Returns:
        State update dict with analysis results
    """
    query = getattr(state, 'query', '')
    dataset_name = getattr(state, 'dataset_name', '')
    llm_purpose = getattr(state, 'llm_purpose', 'default')
    
    return await _unified_query_analysis_impl(query, dataset_name, llm_purpose)


# ==================== Sync Wrapper ====================

def unified_query_analysis_node_sync(
    query: str,
    dataset_name: str,
    llm_purpose: str = "default",
) -> Dict[str, Any]:
    """Synchronous wrapper for direct calls (not via LangGraph)."""
    import asyncio
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        # Already in async context - use nest_asyncio or run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                _unified_query_analysis_impl(query, dataset_name, llm_purpose)
            )
            return future.result()
    else:
        return asyncio.run(_unified_query_analysis_impl(query, dataset_name, llm_purpose))
