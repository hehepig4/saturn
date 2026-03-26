"""
Node: Extract Constraints

Extracts TBox and ABox constraints from natural language query using LLM.

Design Decision:
- LLM outputs STRUCTURED constraints (JSON), NOT raw SPARQL
- This ensures syntax correctness and enables validation
- Programs assemble SPARQL from validated constraints

Two-stage extraction:
1. TBox constraints: Required primitive classes (column types)
2. ABox constraints: Entity values in standardized format
"""

import json
from typing import Dict, Any, List, Tuple, Type, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from pydantic import BaseModel

from workflows.common.node_decorators import graph_node
from workflows.retrieval.matcher.constraints import (
    PathConstraint, ConstraintSet, TBoxConstraint, ABoxConstraint
)
from workflows.retrieval.state import (
    TBoxConstraints, TBoxClassConstraint,
    ABoxConstraints, EntityConstraint,
)


# ==================== Prompts ====================

TBOX_CONSTRAINT_PROMPT = """You are an expert at analyzing natural language queries to identify schema-level constraints for table retrieval.

## Query
{query}

## Available Primitive Classes (Column Types)
Below is the class hierarchy. Indented classes are subclasses of their parent.

{primitive_classes}

## Task
Analyze the query and identify which column types MUST exist in a table to answer this query.

## Rules
- Only select classes that are NECESSARY to answer the query
- Be conservative - fewer constraints is better than wrong constraints
- Prefer more SPECIFIC subclasses when they precisely match the query intent
- Use PARENT class only when no subclass description precisely fits, or when broader coverage is needed
- class_name MUST be from the available list above (without ** markers)

## Output (JSON only)
```json
{{
    "primitive_class_constraints": ["YearColumn", "PersonNameColumn"]
}}
```"""


ABOX_CONSTRAINT_PROMPT = """You are an expert at extracting entity values from natural language queries for precise table matching.

## Query
{query}

## Column Types with Value Format Specifications
For each column type from TBox extraction, convert values to the EXACT standard format specified:

{column_type_details}

## Task
Extract specific entity values that should be matched in the table data.

For each column type listed above that has an explicit value in the query:
1. Identify the value mentioned in the query
2. Convert to the EXACT standard format shown above for that column type

## Rules
- Only extract EXPLICIT values mentioned in the query
- Do NOT infer values that are not directly stated
- column_type MUST be from the column types listed above
- value MUST follow the EXACT format specification shown above

## Output Format
{output_schema}
"""


# ==================== Helper Functions ====================

def get_available_primitive_classes(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Get available primitive classes with hierarchy info for a specific dataset.
    
    Args:
        dataset_name: Dataset name (e.g., 'fetaqa')
        
    Returns:
        List of class dicts with 'name', 'parent_classes', 'description' for tree formatting
    """
    try:
        from store.store_singleton import get_store
        store = get_store()
        
        # Step 1: Find the latest federated_primitive_tbox for this dataset
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
        
        # Step 2: Get classes from ontology_classes filtered by ontology_id
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


def get_column_type_details(class_names: List[str]) -> Tuple[str, Type['BaseModel'], Dict[str, str]]:
    """
    Get detailed information about column types for ABox prompt.
    
    Reads format specifications from core.datatypes.el_datatypes.DATATYPE_SPECS.
    Also creates a dynamic Pydantic model for structured output.
    
    Args:
        class_names: List of primitive class names from TBox extraction
        
    Returns:
        Tuple of:
        - Formatted string with class details and format specifications
        - Dynamic Pydantic model class for structured LLM output
        - Mapping from sanitized field name to original class name
    """
    from pydantic import BaseModel, Field, create_model
    from core.datatypes.el_datatypes import DATATYPE_SPECS
    from typing import Optional
    
    try:
        from store.store_singleton import get_store
        store = get_store()
        
        # Get class descriptions and data ranges
        class_info = {}
        try:
            cls_df = store.db.open_table('ontology_classes').to_pandas()
            for _, row in cls_df.iterrows():
                name = row['class_name'].replace('upo:', '')
                class_info[name] = {
                    'description': row.get('description', ''),
                    'range': 'xsd:string',  # Default
                }
        except Exception:
            pass
        
        # Get data property ranges from ontology
        try:
            dp_df = store.db.open_table('ontology_properties').to_pandas()
            dp_df = dp_df[dp_df['property_type'] == 'data']
            
            for _, row in dp_df.iterrows():
                domain = row.get('domain', [])
                if isinstance(domain, str):
                    domain = [domain]
                
                for d in domain:
                    clean_d = d.replace('upo:', '').replace('[', '').replace(']', '').strip()
                    if clean_d in class_info:
                        ranges = row.get('range', [])
                        if isinstance(ranges, str):
                            ranges = [ranges]
                        if ranges:
                            class_info[clean_d]['range'] = ranges[0].replace('[', '').replace(']', '').strip()
        except Exception:
            pass
        
        # Build detailed format specifications from DATATYPE_SPECS
        lines = []
        model_fields = {}
        field_to_class = {}  # Map sanitized field names back to class names
        
        for name in class_names:
            info = class_info.get(name, {'description': '', 'range': 'xsd:string'})
            data_range = info['range']
            
            # Get format spec from el_datatypes
            spec = DATATYPE_SPECS.get(data_range, DATATYPE_SPECS.get('xsd:string'))
            
            # Build detailed format info
            lines.append(f"### {name}")
            if info['description']:
                lines.append(f"- Description: {info['description']}")
            lines.append(f"- Data Range: {data_range}")
            lines.append(f"- Standard Format: {spec.standard_format}")
            lines.append(f"- Examples: {', '.join(spec.example_values[:3])}")
            lines.append("")
            
            # Build dynamic model field
            sanitized_name = _sanitize_field_name(name)
            field_to_class[sanitized_name] = name
            
            # Create field (format hint is in the prompt, not in Field description)
            model_fields[f"value_{sanitized_name}"] = (Optional[str], None)
        
        # Add overall field (no reasoning fields to save tokens)
        model_fields["same_row_required"] = (bool, True)  # Whether all values must appear in same row
        
        # Create dynamic model
        DynamicABoxModel = create_model('DynamicABoxConstraints', **model_fields)
        
        details_str = "\n".join(lines) if lines else "No column type details available."
        return details_str, DynamicABoxModel, field_to_class
        
    except Exception as e:
        logger.warning(f"Failed to get column type details: {e}")
        # Fallback: simple model with basic fields
        model_fields = {
            "entity_constraints": (List[Dict[str, str]], []),  # List of {value, column_type}
            "same_row_required": (bool, True)
        }
        FallbackModel = create_model('FallbackABoxConstraints', **model_fields)
        return ", ".join(class_names), FallbackModel, {}


def _sanitize_field_name(name: str) -> str:
    """Sanitize class name for use as Python field name."""
    return name.replace("-", "_").replace(" ", "_").replace(":", "_")


def _generate_output_schema(model_class: Type['BaseModel']) -> str:
    """Generate example JSON schema for the dynamic model."""
    schema = model_class.model_json_schema()
    properties = schema.get('properties', {})
    
    example = {}
    for field_name, field_info in properties.items():
        if field_name.startswith('value_'):
            example[field_name] = None  # Use null instead of angle brackets
        elif field_name == 'same_row_required':
            example[field_name] = True
    
    import json
    return f"```json\n{json.dumps(example, indent=2)}\n```"


def call_llm_for_constraints(prompt: str, llm_purpose: str = "default") -> Dict[str, Any]:
    """
    Call LLM to extract constraints from query (unstructured JSON output).
    
    Args:
        prompt: Formatted prompt
        llm_purpose: LLM purpose key (e.g., "default", "gemini")
        
    Returns:
        Parsed JSON response
    """
    from llm.manager import get_llm_by_purpose
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Get LLM
    llm = get_llm_by_purpose(llm_purpose)
    
    # Build messages
    messages = [
        SystemMessage(content="You are a precise constraint extraction assistant. Output ONLY valid JSON."),
        HumanMessage(content=prompt)
    ]
    
    # Invoke LLM
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Extract JSON from response (handle markdown code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    return json.loads(content)


def call_llm_structured(
    prompt: str,
    model_class: Type['BaseModel'],
    llm_purpose: str = "default",
) -> 'BaseModel':
    """
    Call LLM with structured output using a dynamic Pydantic model.
    
    This ensures TBox-ABox strict correspondence by having exactly one
    value field per TBox class.
    
    Args:
        prompt: Formatted prompt
        model_class: Dynamic Pydantic model class for structured output
        llm_purpose: LLM purpose key (e.g., "default", "gemini")
        
    Returns:
        Instance of the dynamic model
    """
    from llm.manager import get_llm_by_purpose
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Get LLM
    llm = get_llm_by_purpose(llm_purpose)
    
    # Use structured output
    structured_llm = llm.with_structured_output(model_class)
    
    # Build messages
    messages = [
        SystemMessage(content="You are a precise constraint extraction assistant."),
        HumanMessage(content=prompt)
    ]
    
    # Invoke with structured output
    return structured_llm.invoke(messages)


def parse_tbox_constraints(data: Dict[str, Any]) -> TBoxConstraints:
    """Parse LLM output into TBoxConstraints model.
    
    Expects format: {"primitive_class_constraints": ["YearColumn", "PersonNameColumn"]}
    """
    class_constraints = []
    
    for class_name in data.get('primitive_class_constraints', []):
        class_constraints.append(TBoxClassConstraint(class_name=class_name))
    
    return TBoxConstraints(
        primitive_class_constraints=class_constraints,
    )


def parse_dynamic_abox_constraints(
    model_instance: 'BaseModel',
    field_to_class: Dict[str, str],
) -> ABoxConstraints:
    """
    Parse dynamic ABox model output into ABoxConstraints.
    
    Converts the dynamic model with value_ClassName fields back to
    the standard EntityConstraint list format.
    
    Args:
        model_instance: Instance of dynamic ABox model
        field_to_class: Mapping from sanitized field names to original class names
        
    Returns:
        ABoxConstraints with entity constraints
    """
    entity_constraints = []
    model_dict = model_instance.model_dump()
    
    for sanitized_name, class_name in field_to_class.items():
        value_field = f"value_{sanitized_name}"
        
        value = model_dict.get(value_field)
        
        # Only add constraint if value is not None/empty
        if value is not None and str(value).strip():
            entity_constraints.append(EntityConstraint(
                value=str(value).strip(),
                column_type=class_name,
            ))
    
    return ABoxConstraints(
        entity_constraints=entity_constraints,
        same_row_required=model_dict.get('same_row_required', True),
    )


# ==================== Node Function ====================

@graph_node(on_error="continue")
def extract_constraints_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract TBox and ABox constraints from natural language query.
    
    Uses LLM to output structured constraints (JSON), NOT raw SPARQL.
    This ensures syntax correctness and enables validation.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with extracted constraints
    """
    query = state.query
    dataset_name = state.dataset_name
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Dataset: {dataset_name}")
    
    # Get available schema elements with hierarchy info
    primitive_classes = get_available_primitive_classes(dataset_name)
    
    logger.info(f"  Available primitive classes: {len(primitive_classes)}")
    
    tbox_constraints = None
    abox_constraints = None
    
    # Extract TBox constraints
    if primitive_classes:
        try:
            # Format as hierarchy tree for better LLM understanding
            from utils.tree_formatter import format_class_hierarchy
            class_tree = format_class_hierarchy(
                primitive_classes,
                include_column_root=True,
                max_desc_length=None,  # Don't truncate descriptions
            )
            
            tbox_prompt = TBOX_CONSTRAINT_PROMPT.format(
                query=query,
                primitive_classes=class_tree,
            )
            
            tbox_data = call_llm_for_constraints(tbox_prompt)
            tbox_constraints = parse_tbox_constraints(tbox_data)
            
            logger.info(f"\n  📦 TBox Constraints Extracted:")
            logger.info(f"     Primitive Class Constraints ({len(tbox_constraints.primitive_class_constraints)}):")
            for cc in tbox_constraints.primitive_class_constraints:
                logger.info(f"       - {cc.class_name}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Failed to extract TBox constraints: {e}")
    
    # Extract ABox constraints using dynamic structured output
    if tbox_constraints and tbox_constraints.primitive_class_constraints:
        try:
            column_types = [c.class_name for c in tbox_constraints.primitive_class_constraints]
            column_type_details, DynamicABoxModel, field_to_class = get_column_type_details(column_types)
            
            # Generate output schema example for prompt
            output_schema = _generate_output_schema(DynamicABoxModel)
            
            abox_prompt = ABOX_CONSTRAINT_PROMPT.format(
                query=query,
                column_type_details=column_type_details,
                output_schema=output_schema,
            )
            
            # Use structured output with dynamic model
            abox_model_instance = call_llm_structured(abox_prompt, DynamicABoxModel)
            abox_constraints = parse_dynamic_abox_constraints(abox_model_instance, field_to_class)
            
            logger.info(f"\n  📦 ABox Constraints Extracted:")
            logger.info(f"     Same row required: {abox_constraints.same_row_required}")
            logger.info(f"     Entity Constraints ({len(abox_constraints.entity_constraints)}):")
            for ec in abox_constraints.entity_constraints:
                logger.info(f"       - '{ec.value}' in {ec.column_type}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Failed to extract ABox constraints: {e}")
    
    logger.info(f"  ✓ Constraint extraction completed")
    
    return {
        'tbox_constraints': tbox_constraints,
        'abox_constraints': abox_constraints,
    }


def convert_to_path_constraints(
    tbox_constraints: TBoxConstraints,
    abox_constraints: ABoxConstraints,
) -> 'ConstraintSet':
    """
    Convert TBox/ABox constraints to PathConstraint format for PathMatcher.
    
    This bridges the LLM-extracted constraints to the unified PathMatcher input.
    
    Args:
        tbox_constraints: Extracted TBox constraints (primitive classes)
        abox_constraints: Extracted ABox constraints (entity values)
        
    Returns:
        ConstraintSet for PathMatcher
    """
    constraints = []
    
    if not tbox_constraints:
        return ConstraintSet(constraints=[])
    
    # Build value map from ABox constraints: column_type → [values]
    value_map: Dict[str, List[str]] = {}
    if abox_constraints and abox_constraints.entity_constraints:
        for ec in abox_constraints.entity_constraints:
            col_type = ec.column_type.replace('upo:', '')
            if col_type not in value_map:
                value_map[col_type] = []
            value_map[col_type].append(ec.value)
    
    # Convert each TBox class constraint to PathConstraint
    for cc in tbox_constraints.primitive_class_constraints:
        class_name = cc.class_name.replace('upo:', '')
        
        # Check if there's an ABox value for this class
        values = value_map.get(class_name, [])
        
        if values:
            # Create PathConstraint with ABox value for each value
            for value in values:
                constraints.append(PathConstraint(
                    tbox=TBoxConstraint(
                        class_name=class_name,
                        required=cc.required,
                    ),
                    abox=ABoxConstraint(value=value),
                ))
        else:
            # TBox-only constraint (no value)
            constraints.append(PathConstraint(
                tbox=TBoxConstraint(
                    class_name=class_name,
                    required=cc.required,
                ),
            ))
    
    return ConstraintSet(constraints=constraints)