"""
Transform Generator

Generates TransformContract using LLM for a specific (primitive_class, data_property) pair.

The LLM receives:
- Primitive class name and description
- Target DataProperty name, range_type, and description
- Sample values from the column
- Standard format specification for the target datatype

The LLM generates:
- pattern: regex pattern this transform handles
- transform_expr: Python expression to convert matched values TO STANDARD FORMAT
"""

import re2
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from pydantic import BaseModel, Field

import pandas as pd

from workflows.population.contract import (
    TransformContract, 
    TargetType, 
    DataPropertySpec,
    get_target_type_from_range,
)
from workflows.population.safe_regex import (
    safe_compile,
    safe_pandas_match,
)
from core.datatypes import (
    get_datatype_spec,
    get_llm_instruction,
    DATATYPE_SPECS,
    DatatypeSpec,
)

# Global flag for one-time prompt logging
_PROMPT_LOGGED = False

# Maximum length for values in error messages and prompts (prevents token overflow)
_MAX_VALUE_LENGTH_IN_PROMPT = 100


def _truncate_value(value: str, max_length: int = _MAX_VALUE_LENGTH_IN_PROMPT) -> str:
    """Truncate a value for display in prompts/errors to prevent token overflow."""
    if len(value) <= max_length:
        return value
    return value[:max_length] + "..."


class TransformContractOutput(BaseModel):
    """LLM structured output for transform contract."""
    
    pattern: str = Field(
        ...,  # Required, no default
        description=(
            "Regex pattern matching valid input values for this data property. "
            "Must include ^ and $ anchors for precise matching. "
            "We use Google RE2 engine which DOES NOT support: backreferences (\\1), "
            "lookahead (?=...), lookbehind (?<=...), possessive quantifiers, or atomic groups. "
            "NEVER use these features — the pattern will be rejected. "
            "Prefer flat character classes: [0-9.]+, [^,]+, [A-Za-z ]+. "
            "Example: '^-?[0-9,]+$' for integers, '^[^\\n]+$' for text lines."
        )
    )
    
    transform_expr: str = Field(
        ...,  # Required, no default
        description=(
            "Python expression using variable 'x' (the stripped input string) "
            "to convert to the target type. Must return the correct Python type. "
            "Example: 'int(x.replace(\",\", \"\"))' for formatted integers."
        )
    )


# Import namespace configuration
from workflows.population.transform_namespace import (
    build_namespace,
    build_prompt_section,
)

# NOTE: Static content placed FIRST to maximize prefix cache hit rate
# SYSTEM_PROMPT is invariant across all calls
# USER_PROMPT_TEMPLATE contains dynamic content (class, property, samples)

# Build execution environment prompt section from namespace config
_EXECUTION_ENV_PROMPT = build_prompt_section()

SYSTEM_PROMPT = f"""Generate a TransformContract for extracting semantic information from cell values.

A TransformContract has:
1. `pattern` - Regex to match valid input values (use ^ and $)
2. `transform_expr` - Python code using 'x' to extract/convert the value

Guidelines:
- Pattern should match the expected data format
- transform_expr must return the correct type (int/float/str)
- For string properties: consider if this property needs specific extraction
  (e.g., abbreviations, keywords) rather than just x.strip()

**CRITICAL: We use Google RE2 regex engine. The following are NOT supported and will be REJECTED:**
- Backreferences: `(\\w+)\\s+\\1` — FORBIDDEN
- Lookahead: `(?=...)` `(?!...)` — FORBIDDEN
- Lookbehind: `(?<=...)` `(?<!...)` — FORBIDDEN
- Possessive quantifiers: `a++` `a*+` — FORBIDDEN
- Atomic groups: `(?>...)` — FORBIDDEN

**Preferred patterns:**
- Flat character classes: `[0-9.]+` `[^,]+` `[A-Za-z ]+`
- Negated classes for catch-all: `^[^\\n]+$` instead of `^.+$`
- Multi-part: `[^,]+,[^,]+` instead of `(.+),(.+)`

{_EXECUTION_ENV_PROMPT}

"""


USER_PROMPT_TEMPLATE = """## All Properties for this class (context):
{sibling_properties}

## Target Property:
Class: {primitive_class}
Property: {data_property} → {target_type}
{class_description}{property_description}

Standard Format: {standard_format_requirement}

Sample values: {sample_values}
{unmatched_section}"""


# Template for retry feedback
RETRY_PROMPT_TEMPLATE = """
Previous attempts failed:
{failed_attempts}

Errors: {sample_errors}

Generate a DIFFERENT pattern/transform."""


@dataclass
class FailedAttempt:
    """Record of a failed transform attempt for feedback to LLM."""
    pattern: str
    transform_expr: str
    success_rate: float  # Unified: match + transform success
    error_count: int
    sample_errors: List[str]


@dataclass
class GenerationResult:
    """Result of transform generation."""
    contract: TransformContract
    raw_response: str = ""


class TransformGenerator:
    """
    Generates TransformContract for a (primitive_class, data_property) pair.
    """
    
    def __init__(self, llm_purpose: str = "default"):
        """
        Initialize TransformGenerator.
        
        Args:
            llm_purpose: LLM purpose key for model selection
        """
        self._llm = None
        self._llm_purpose = llm_purpose
    
    def _get_llm(self):
        """Get LLM instance."""
        if self._llm is not None:
            return self._llm
        
        from llm.manager import get_llm_by_purpose
        self._llm = get_llm_by_purpose(self._llm_purpose)
        return self._llm
    
    def _get_standard_format_requirement(self, range_type: str) -> str:
        """
        Get standard format requirement text for a range type.
        
        This provides clear instructions to the LLM about the expected output format.
        """
        spec = get_datatype_spec(range_type)
        if spec:
            return (
                f"**Target Type**: {spec.target_type.value}\n"
                f"**Standard Format**: {spec.standard_format}\n"
                f"**Validation Pattern**: `{spec.validation_pattern}`\n"
                f"**Example Values**: {', '.join(spec.example_values)}\n"
                f"**Instruction**: {spec.llm_instruction}"
            )
        
        # Default for unknown types
        target_type = get_target_type_from_range(range_type)
        defaults = {
            TargetType.INT: (
                "**Target Type**: int\n"
                "**Standard Format**: Integer without formatting: -?[0-9]+\n"
                "**Instruction**: Convert to integer using int(x.replace(',', ''))"
            ),
            TargetType.FLOAT: (
                "**Target Type**: float\n"
                "**Standard Format**: Decimal number: -?[0-9]+(\\.[0-9]+)?\n"
                "**Instruction**: Convert to float using float(x.replace(',', ''))"
            ),
            TargetType.DATETIME: (
                "**Target Type**: datetime\n"
                "**Standard Format**: ISO 8601: YYYY-MM-DDTHH:MM:SS\n"
                "**Instruction**: Parse and output in ISO format using datetime.fromisoformat(x).isoformat()"
            ),
            TargetType.STR: (
                "**Target Type**: str\n"
                "**Standard Format**: Unicode string, whitespace normalized\n"
                "**Instruction**: Strip whitespace using x.strip()"
            ),
        }
        return defaults.get(target_type, "**Target Type**: str\n**Instruction**: Return as string.")
    
    def generate(
        self,
        primitive_class: str,
        data_property_spec: DataPropertySpec,
        sample_values: List[str],
        class_description: Optional[str] = None,
        unmatched_values: Optional[List[str]] = None,
        sibling_properties: Optional[List[DataPropertySpec]] = None,
    ) -> GenerationResult:
        """
        Generate TransformContract for a (class, property) pair.
        
        Args:
            primitive_class: The primitive class name
            data_property_spec: Target DataProperty specification
            sample_values: Sample values from the column
            class_description: Optional description of the primitive class
            unmatched_values: Values that didn't match an existing contract (for refinement)
            sibling_properties: Other DataProperties for the same class (for context)
            
        Returns:
            GenerationResult with TransformContract
        """
        from workflows.population.sampling_utils import sample_for_llm_prompt, sample_values as do_sample
        
        llm = self._get_llm()
        
        # Format sample values using true random sampling
        samples_str = sample_for_llm_prompt(sample_values, n=15)
        # Keep clean samples for contract storage (also randomly sampled)
        clean_samples = do_sample(
            [str(v).strip() for v in sample_values if v and str(v).strip()],
            n=20
        )
        
        # Format unmatched values section
        unmatched_section = ""
        if unmatched_values:
            unmatched_sampled = do_sample(list(unmatched_values), n=10)
            # Truncate each value to prevent token overflow
            unmatched_str = ", ".join(f'"{_truncate_value(str(v))}"' for v in unmatched_sampled)
            unmatched_section = f"\nUnmatched: {unmatched_str}"
        
        # Format sibling properties section with rich context
        sibling_section = "(single property for this class)"
        if sibling_properties:
            sibling_lines = []
            for sp in sibling_properties:
                is_target = sp.name == data_property_spec.name
                marker = " ← GENERATING THIS" if is_target else ""
                
                # Build rich description
                parts = [f"- **{sp.name}** ({sp.range_type})"]
                if sp.comment:
                    parts.append(f"  Description: {sp.comment}")
                if sp.readout_template and not is_target:
                    # Show template for siblings to help LLM understand differentiation
                    parts.append(f"  Output template: \"{sp.readout_template}\"")
                
                sibling_lines.append("\n".join(parts) + marker)
            sibling_section = "\n".join(sibling_lines)
        
        # Get target type and standard format requirement
        target_type = get_target_type_from_range(data_property_spec.range_type)
        standard_format_req = self._get_standard_format_requirement(data_property_spec.range_type)
        
        # Build prompt (simplified format)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            primitive_class=primitive_class,
            class_description=f"\n{class_description}" if class_description else "",
            data_property=data_property_spec.name,
            target_type=target_type.value,
            property_description=f"\n{data_property_spec.comment}" if data_property_spec.comment else "",
            sample_values=samples_str,
            unmatched_section=unmatched_section,
            standard_format_requirement=standard_format_req,
            sibling_properties=sibling_section,
        )
        
        # Call LLM with structured output and retry mechanism
        from llm.invoke_with_stats import invoke_structured_llm_with_retry
        from llm.manager import get_llm_by_purpose
        
        try:
            messages_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            
            # Log the first prompt for debugging
            global _PROMPT_LOGGED
            if not _PROMPT_LOGGED:
                logger.debug("=" * 80)
                logger.debug("[TransformGenerator] FIRST PROMPT (one-time log):")
                logger.debug("=" * 80)
                logger.debug(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}")
                logger.debug("-" * 40)
                logger.debug(f"USER PROMPT:\n{user_prompt}")
                logger.debug("=" * 80)
                _PROMPT_LOGGED = True
            
            # Factory function creates LLM with specified temperature for retry
            def llm_factory(temperature: float):
                return get_llm_by_purpose(self._llm_purpose, temperature_override=temperature)
            
            result: TransformContractOutput = invoke_structured_llm_with_retry(
                llm_factory=llm_factory,
                output_schema=TransformContractOutput,
                prompt=messages_str,
                max_retries=3,
            )
            
            # Build contract
            contract = TransformContract(
                primitive_class=primitive_class,
                data_property=data_property_spec.name,
                pattern=result.pattern,
                target_type=target_type,
                transform_expr=result.transform_expr,
                sample_values=clean_samples[:5],
            )
            
            return GenerationResult(
                contract=contract,
                raw_response=str(result),
            )
            
        except Exception as e:
            logger.error(f"LLM transform generation failed: {e}")
            return GenerationResult(
                contract=self._get_fallback_contract(
                    primitive_class, 
                    data_property_spec,
                ),
                raw_response=str(e),
            )
    
    def _get_fallback_contract(
        self, 
        primitive_class: str, 
        data_property_spec: DataPropertySpec,
    ) -> TransformContract:
        """
        Get fallback contract when LLM fails.
        
        NOTE: This is a last resort. The contract may not work well for the data.
        We log a warning and use conservative defaults.
        """
        # Ensure primitive_class is not empty (use root class as fallback)
        if not primitive_class:
            primitive_class = "upo:Column"
            logger.warning(
                f"Empty primitive_class for {data_property_spec.name}, using root class 'upo:Column'"
            )
        
        target_type = get_target_type_from_range(data_property_spec.range_type)
        
        # Conservative fallback patterns - intentionally restrictive to surface issues
        # Using stricter patterns than before to avoid masking problems
        fallback_configs = {
            TargetType.INT: {
                "pattern": "^-?[0-9]+$",  # Simple integers only, no commas
                "transform_expr": "int(x)",
            },
            TargetType.FLOAT: {
                "pattern": "^-?[0-9]+\\.?[0-9]*$",  # Simple decimals only
                "transform_expr": "float(x)",
            },
            TargetType.STR: {
                "pattern": "^.+$",  # Non-empty strings
                "transform_expr": "x",
            },
            TargetType.DATETIME: {
                "pattern": "^\\d{4}-\\d{2}-\\d{2}$",  # Strict ISO date only
                "transform_expr": "x",  # Keep as string
            },
        }
        
        config = fallback_configs.get(target_type, fallback_configs[TargetType.STR])
        
        logger.warning(
            f"Using FALLBACK contract for {primitive_class}::{data_property_spec.name}. "
            f"LLM generation failed. Contract may not match data well."
        )
        
        return TransformContract(
            primitive_class=primitive_class,
            data_property=data_property_spec.name,
            pattern=config["pattern"],
            target_type=target_type,
            transform_expr=config["transform_expr"],
        )
    
    def generate_with_feedback(
        self,
        primitive_class: str,
        data_property_spec: DataPropertySpec,
        sample_values: List[str],
        class_description: Optional[str] = None,
        failed_attempts: Optional[List[FailedAttempt]] = None,
        sibling_properties: Optional[List[DataPropertySpec]] = None,
    ) -> GenerationResult:
        """
        Generate TransformContract with feedback from previous failed attempts.
        
        This method is used in the retry loop when previous contracts failed.
        It includes failed attempt information in the prompt to help LLM generate
        a better alternative.
        
        Args:
            primitive_class: The primitive class name
            data_property_spec: Target DataProperty specification
            sample_values: Sample values from the column
            class_description: Optional description of the primitive class
            failed_attempts: List of previous failed attempts for feedback
            sibling_properties: Other DataProperties for the same class (for context)
            
        Returns:
            GenerationResult with TransformContract
        """
        from workflows.population.sampling_utils import sample_for_llm_prompt, sample_values as do_sample
        
        llm = self._get_llm()
        
        # Format sample values using true random sampling
        samples_str = sample_for_llm_prompt(sample_values, n=15)
        # Keep clean samples for contract storage (also randomly sampled)
        clean_samples = do_sample(
            [str(v).strip() for v in sample_values if v and str(v).strip()],
            n=20
        )
        
        # Format sibling properties section with rich context
        sibling_section = "(single property for this class)"
        if sibling_properties:
            sibling_lines = []
            for sp in sibling_properties:
                is_target = sp.name == data_property_spec.name
                marker = " ← GENERATING THIS" if is_target else ""
                
                # Build rich description
                parts = [f"- **{sp.name}** ({sp.range_type})"]
                if sp.comment:
                    parts.append(f"  Description: {sp.comment}")
                if sp.readout_template and not is_target:
                    parts.append(f"  Output template: \"{sp.readout_template}\"")
                
                sibling_lines.append("\n".join(parts) + marker)
            sibling_section = "\n".join(sibling_lines)
        
        # Get target type and standard format requirement
        target_type = get_target_type_from_range(data_property_spec.range_type)
        standard_format_req = self._get_standard_format_requirement(data_property_spec.range_type)
        
        # Build base prompt (simplified format)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            primitive_class=primitive_class,
            class_description=f"\n{class_description}" if class_description else "",
            data_property=data_property_spec.name,
            target_type=target_type.value,
            property_description=f"\n{data_property_spec.comment}" if data_property_spec.comment else "",
            sample_values=samples_str,
            unmatched_section="",  # Already included in failed attempts
            standard_format_requirement=standard_format_req,
            sibling_properties=sibling_section,
        )
        
        # Add failed attempts feedback
        if failed_attempts:
            failed_info_lines = []
            for i, attempt in enumerate(failed_attempts, 1):
                failed_info_lines.append(
                    f"Attempt {i}:\n"
                    f"  - Pattern: `{attempt.pattern}`\n"
                    f"  - Transform: `{attempt.transform_expr}`\n"
                    f"  - Success Rate: {attempt.success_rate:.1%} (match + transform)\n"
                    f"  - Error Count: {attempt.error_count}"
                )
            failed_info = "\n".join(failed_info_lines)
            
            # Collect all sample errors from failed attempts
            all_sample_errors = []
            for attempt in failed_attempts:
                all_sample_errors.extend(attempt.sample_errors[:3])
            sample_errors_str = "\n".join(all_sample_errors[:8]) if all_sample_errors else "No specific error messages captured."
            
            feedback_section = RETRY_PROMPT_TEMPLATE.format(
                failed_attempts=failed_info,
                sample_errors=sample_errors_str,
            )
            user_prompt = user_prompt + "\n\n" + feedback_section
        
        # Call LLM with structured output and retry mechanism
        from llm.invoke_with_stats import invoke_structured_llm_with_retry
        from llm.manager import get_llm_by_purpose
        
        try:
            messages_str = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            
            # Log the first prompt for debugging (shared with generate method)
            global _PROMPT_LOGGED
            if not _PROMPT_LOGGED:
                logger.debug("=" * 80)
                logger.debug("[TransformGenerator with feedback] FIRST PROMPT (one-time log):")
                logger.debug("=" * 80)
                logger.debug(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}")
                logger.debug("-" * 40)
                logger.debug(f"USER PROMPT:\n{user_prompt}")
                logger.debug("=" * 80)
                _PROMPT_LOGGED = True
            
            # Factory function creates LLM with specified temperature for retry
            def llm_factory(temperature: float):
                return get_llm_by_purpose(self._llm_purpose, temperature_override=temperature)
            
            result: TransformContractOutput = invoke_structured_llm_with_retry(
                llm_factory=llm_factory,
                output_schema=TransformContractOutput,
                prompt=messages_str,
                max_retries=3,
            )
            
            # Build contract
            contract = TransformContract(
                primitive_class=primitive_class,
                data_property=data_property_spec.name,
                pattern=result.pattern,
                target_type=target_type,
                transform_expr=result.transform_expr,
                sample_values=clean_samples[:5],
            )
            
            return GenerationResult(
                contract=contract,
                raw_response=str(result),
            )
            
        except Exception as e:
            logger.error(f"LLM transform generation with feedback failed: {e}")
            return GenerationResult(
                contract=self._get_fallback_contract(
                    primitive_class, 
                    data_property_spec,
                ),
                raw_response=str(e),
            )


def apply_transform(
    contract: TransformContract, 
    values: List[str],
) -> Tuple[List[Any], int, int]:
    """
    Apply a TransformContract to a list of values using pandas vectorization.
    
    Args:
        contract: The transform contract to apply
        values: Raw string values
        
    Returns:
        Tuple of (transformed_values, null_count, error_count)
        
    Note:
        - Uses pandas str.match for vectorized pattern matching (~15x faster)
        - If pattern is invalid, treats all as errors
        - Uses vectorized transforms for common cases (int, float, str)
    """
    pattern_str = contract.pattern
    transform_expr = contract.transform_expr
    
    # Convert to pandas Series, handling None
    s = pd.Series(
        [str(v).strip() if v is not None else "" for v in values], 
        dtype="string"
    )
    
    # Identify null values (empty strings)
    null_mask = s == ""
    null_count = int(null_mask.sum())
    non_null_count = len(values) - null_count
    
    # Vectorized pattern matching (direct call, no subprocess overhead)
    matched, has_error = safe_pandas_match(pattern_str, s)
    
    if has_error:
        # Error indicates invalid pattern
        # Treat all non-null values as errors - caller should regenerate contract
        return [], null_count, non_null_count
    
    # Get matched values (non-null and pattern matched)
    matched_mask = matched & (~null_mask)
    error_count = int(((~matched) & (~null_mask)).sum())
    
    matched_values = s[matched_mask]
    
    if matched_values.empty:
        return [], null_count, error_count
    
    # Apply transform
    transformed, transform_errors = _apply_transform_vectorized(
        matched_values, transform_expr
    )
    
    return transformed, null_count, error_count + transform_errors


def _apply_transform_vectorized(
    matched_values: pd.Series,
    transform_expr: str,
) -> Tuple[List[Any], int]:
    """
    Apply transform expression to matched values.
    Uses vectorized operations for common cases.
    """
    transformed: List[Any] = []
    transform_errors = 0
    
    if transform_expr == "x":
        # Identity transform
        transformed = matched_values.tolist()
    elif transform_expr == "int(x)":
        # Integer conversion
        numeric = pd.to_numeric(matched_values, errors='coerce')
        valid_mask = ~numeric.isna()
        transformed = numeric[valid_mask].astype(int).tolist()
        transform_errors = int((~valid_mask).sum())
    elif transform_expr == "float(x)":
        # Float conversion
        numeric = pd.to_numeric(matched_values, errors='coerce')
        valid_mask = ~numeric.isna()
        transformed = numeric[valid_mask].tolist()
        transform_errors = int((~valid_mask).sum())
    else:
        # Complex transform - use loop
        transform_func = _build_transform_func(transform_expr)
        for v in matched_values.tolist():
            try:
                transformed.append(transform_func(v))
            except Exception:
                transform_errors += 1
    
    return transformed, transform_errors


# Cache the namespace to avoid rebuilding it for every transform
_TRANSFORM_NAMESPACE = None

def _get_transform_namespace():
    """Get the cached transform namespace."""
    global _TRANSFORM_NAMESPACE
    if _TRANSFORM_NAMESPACE is None:
        _TRANSFORM_NAMESPACE = build_namespace()
    return _TRANSFORM_NAMESPACE


def _build_transform_func(transform_expr: str):
    """Build a transform function from an expression string.
    
    The namespace is configured in transform_namespace.py and includes:
    - datetime/date/time/timedelta classes for temporal operations
    - re module for regex operations
    - Safe builtins: isinstance, type, int, float, str, len, etc.
    
    Note: SyntaxWarnings (e.g., invalid escape sequences like '\\T') are
    converted to errors so they trigger retry/regeneration logic.
    """
    import warnings
    namespace = _get_transform_namespace()
    
    def transform_func(x: str) -> Any:
        local_namespace = {"x": x, **namespace}
        # Convert SyntaxWarning to error to trigger retry on invalid escape sequences
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            return eval(transform_expr, local_namespace)
    
    return transform_func


def get_sample_errors(
    contract: TransformContract,
    values: List[str],
    max_samples: int = 5,
) -> List[str]:
    """
    Get sample error messages from applying a contract to values.
    
    Uses pandas vectorization for pattern matching (fast), then loops
    for transform testing (to get specific error messages).
    
    This is used for feedback in retry loops - to help LLM understand
    what went wrong and generate a better contract.
    
    Args:
        contract: The contract to test
        values: Values to transform
        max_samples: Maximum number of error samples to return
        
    Returns:
        List of error message strings
        
    Note:
        Uses pandas str.match for pattern matching.
    """
    pattern_str = contract.pattern
    
    # Validate pattern syntax first
    compiled = safe_compile(pattern_str)
    if compiled is None:
        return [f"Regex compilation error for pattern: {pattern_str}"]
    
    # Convert to pandas Series
    s = pd.Series(
        [str(v).strip() if v is not None else "" for v in values], 
        dtype="string"
    )
    
    # Identify non-empty values
    non_empty_mask = s != ""
    
    # Vectorized pattern matching (direct call, no subprocess overhead)
    matched, has_error = safe_pandas_match(pattern_str, s)
    
    if has_error:
        # Return error to trigger pattern regeneration
        return [f"Pattern '{pattern_str[:50]}...' has regex error"]
    
    sample_errors = []
    transform_func = _build_transform_func(contract.transform_expr)
    
    # Collect unmatched value errors
    unmatched_mask = (~matched) & non_empty_mask
    unmatched_values = s[unmatched_mask].head(max_samples).tolist()
    for v_str in unmatched_values:
        if len(sample_errors) >= max_samples:
            break
        # Truncate value to prevent token overflow in retry prompts
        truncated_v = _truncate_value(str(v_str))
        sample_errors.append(f"Value '{truncated_v}' did not match pattern '{contract.pattern}'")
    
    # If we have room, collect transform errors
    if len(sample_errors) < max_samples:
        matched_mask = matched & non_empty_mask
        matched_values = s[matched_mask].tolist()
        
        for v_str in matched_values:
            if len(sample_errors) >= max_samples:
                break
            try:
                transform_func(v_str)
            except Exception as e:
                error_type = type(e).__name__
                # Truncate value to prevent token overflow in retry prompts
                truncated_v = _truncate_value(str(v_str))
                sample_errors.append(
                    f"Value '{truncated_v}' matched pattern but transform failed: {error_type}: {e}"
                )
    
    return sample_errors
