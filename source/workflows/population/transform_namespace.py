"""
Transform Namespace Configuration

Defines the execution environment for transform expressions.
This module provides:
1. A declarative configuration of available functions/modules
2. Automatic namespace building for eval()
3. Automatic prompt generation for LLM

Usage:
    from workflows.population.transform_namespace import (
        build_namespace,
        build_prompt_section,
    )
    
    # For eval()
    namespace = build_namespace()
    
    # For LLM prompt
    prompt_section = build_prompt_section()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime as dt
import re2 as re_module


@dataclass
class NamespaceEntry:
    """Configuration entry for a namespace item.
    
    Attributes:
        name: Name exposed in transform (e.g., "datetime")
        value: Actual Python object to expose
        description: Human-readable description for LLM prompt
        usage_examples: List of example usages in transform_expr
        category: Category for grouping in prompt (e.g., "module", "builtin")
    """
    name: str
    value: Any
    description: str
    usage_examples: List[str] = field(default_factory=list)
    category: str = "builtin"


# =============================================================================
# MODULE NAMESPACE ENTRIES
# =============================================================================
# These are module-level objects exposed directly in the namespace

NAMESPACE_MODULES: List[NamespaceEntry] = [
    # datetime.datetime class (NOT the module!)
    # This allows: datetime.fromisoformat(x), datetime.strptime(x, fmt)
    NamespaceEntry(
        name="datetime",
        value=dt.datetime,
        description="datetime class for parsing and creating datetime objects",
        usage_examples=[
            "datetime.fromisoformat(x.replace(' ', 'T'))",
            "datetime.strptime(x, '%Y-%m-%d %H:%M:%S')",
        ],
        category="module",
    ),
    # datetime.date class for date-only operations
    NamespaceEntry(
        name="date",
        value=dt.date,
        description="date class for date-only operations",
        usage_examples=[
            "date.fromisoformat(x)",
        ],
        category="module",
    ),
    # datetime.time class for time-only operations
    NamespaceEntry(
        name="time",
        value=dt.time,
        description="time class for time-only operations",
        usage_examples=[
            "time.fromisoformat(x)",
        ],
        category="module",
    ),
    # datetime.timedelta for time differences
    NamespaceEntry(
        name="timedelta",
        value=dt.timedelta,
        description="timedelta for time differences",
        usage_examples=[
            "timedelta(days=1)",
        ],
        category="module",
    ),
    # re module for regex operations
    NamespaceEntry(
        name="re",
        value=re_module,
        description="Google RE2 regex module (linear-time, no backreferences/lookahead)",
        usage_examples=[
            "re.sub(r'\\s+', ' ', x)",
            "re.match(r'^\\d+$', x)",
        ],
        category="module",
    ),
]


# =============================================================================
# BUILTIN NAMESPACE ENTRIES
# =============================================================================
# These are placed in __builtins__ dict for security

NAMESPACE_BUILTINS: List[NamespaceEntry] = [
    # Type checking functions (frequently used by LLM)
    NamespaceEntry(
        name="isinstance",
        value=isinstance,
        description="check if object is instance of type",
        usage_examples=["isinstance(x, str)"],
        category="type_check",
    ),
    NamespaceEntry(
        name="type",
        value=type,
        description="get type of object",
        usage_examples=["type(x)"],
        category="type_check",
    ),
    
    # Type conversion functions (most frequently used)
    NamespaceEntry(
        name="int",
        value=int,
        description="convert to integer",
        usage_examples=["int(x)", "int(x.replace(',', ''))"],
        category="type_convert",
    ),
    NamespaceEntry(
        name="float",
        value=float,
        description="convert to float",
        usage_examples=["float(x)", "float(x.rstrip('%'))"],
        category="type_convert",
    ),
    NamespaceEntry(
        name="str",
        value=str,
        description="convert to string",
        usage_examples=["str(x)", "str(x).strip()"],
        category="type_convert",
    ),
    NamespaceEntry(
        name="bool",
        value=bool,
        description="convert to boolean",
        usage_examples=["bool(x)"],
        category="type_convert",
    ),
    
    # Sequence operations
    NamespaceEntry(
        name="len",
        value=len,
        description="get length of sequence",
        usage_examples=["len(x)"],
        category="sequence",
    ),
    NamespaceEntry(
        name="list",
        value=list,
        description="create list from iterable",
        usage_examples=["list(x)"],
        category="sequence",
    ),
    NamespaceEntry(
        name="tuple",
        value=tuple,
        description="create tuple from iterable",
        usage_examples=["tuple(x)"],
        category="sequence",
    ),
    NamespaceEntry(
        name="dict",
        value=dict,
        description="create dictionary",
        usage_examples=["dict()"],
        category="sequence",
    ),
    NamespaceEntry(
        name="set",
        value=set,
        description="create set from iterable",
        usage_examples=["set(x)"],
        category="sequence",
    ),
    
    # Math functions
    NamespaceEntry(
        name="abs",
        value=abs,
        description="absolute value",
        usage_examples=["abs(x)"],
        category="math",
    ),
    NamespaceEntry(
        name="round",
        value=round,
        description="round to n decimal places",
        usage_examples=["round(x, 2)"],
        category="math",
    ),
    NamespaceEntry(
        name="min",
        value=min,
        description="minimum value",
        usage_examples=["min(a, b)"],
        category="math",
    ),
    NamespaceEntry(
        name="max",
        value=max,
        description="maximum value",
        usage_examples=["max(a, b)"],
        category="math",
    ),
    NamespaceEntry(
        name="sum",
        value=sum,
        description="sum of iterable",
        usage_examples=["sum([1, 2, 3])"],
        category="math",
    ),
    NamespaceEntry(
        name="pow",
        value=pow,
        description="power function",
        usage_examples=["pow(x, 2)"],
        category="math",
    ),
    
    # String utilities
    NamespaceEntry(
        name="ord",
        value=ord,
        description="get Unicode code point",
        usage_examples=["ord('A')"],
        category="string",
    ),
    NamespaceEntry(
        name="chr",
        value=chr,
        description="get character from code point",
        usage_examples=["chr(65)"],
        category="string",
    ),
    NamespaceEntry(
        name="repr",
        value=repr,
        description="get string representation",
        usage_examples=["repr(x)"],
        category="string",
    ),
    
    # Constants
    NamespaceEntry(
        name="None",
        value=None,
        description="null value",
        usage_examples=["x if x is not None else ''"],
        category="constant",
    ),
    NamespaceEntry(
        name="True",
        value=True,
        description="boolean true",
        usage_examples=["True"],
        category="constant",
    ),
    NamespaceEntry(
        name="False",
        value=False,
        description="boolean false",
        usage_examples=["False"],
        category="constant",
    ),
    
    # Iteration utilities (occasionally used)
    NamespaceEntry(
        name="range",
        value=range,
        description="generate range of numbers",
        usage_examples=["range(10)"],
        category="iteration",
    ),
    NamespaceEntry(
        name="enumerate",
        value=enumerate,
        description="enumerate iterable",
        usage_examples=["enumerate(x)"],
        category="iteration",
    ),
    NamespaceEntry(
        name="zip",
        value=zip,
        description="zip iterables together",
        usage_examples=["zip(a, b)"],
        category="iteration",
    ),
    NamespaceEntry(
        name="map",
        value=map,
        description="map function over iterable",
        usage_examples=["map(int, x.split(','))"],
        category="iteration",
    ),
    NamespaceEntry(
        name="filter",
        value=filter,
        description="filter iterable",
        usage_examples=["filter(bool, x)"],
        category="iteration",
    ),
    NamespaceEntry(
        name="sorted",
        value=sorted,
        description="sort iterable",
        usage_examples=["sorted(x)"],
        category="iteration",
    ),
    NamespaceEntry(
        name="reversed",
        value=reversed,
        description="reverse iterable",
        usage_examples=["reversed(x)"],
        category="iteration",
    ),
    
    # Logic utilities
    NamespaceEntry(
        name="any",
        value=any,
        description="check if any element is truthy",
        usage_examples=["any(x)"],
        category="logic",
    ),
    NamespaceEntry(
        name="all",
        value=all,
        description="check if all elements are truthy",
        usage_examples=["all(x)"],
        category="logic",
    ),
    
    # Attribute access
    NamespaceEntry(
        name="hasattr",
        value=hasattr,
        description="check if object has attribute",
        usage_examples=["hasattr(x, 'strip')"],
        category="attribute",
    ),
    NamespaceEntry(
        name="getattr",
        value=getattr,
        description="get attribute from object",
        usage_examples=["getattr(x, 'strip', lambda: x)"],
        category="attribute",
    ),
]


# =============================================================================
# IMPORTANT RESTRICTIONS
# =============================================================================
# These are rules that MUST be communicated to the LLM

EXECUTION_RESTRICTIONS = [
    "Do NOT use `import` statements - all modules are pre-imported",
    "Do NOT define functions with `def` - use single expressions only",
    "Do NOT use multi-line statements or semicolons to separate statements",
    "The variable `x` is the input string value (already stripped)",
]


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def build_namespace() -> Dict[str, Any]:
    """Build the execution namespace for transform expressions.
    
    Returns:
        Dict containing all exposed modules and a safe __builtins__ dict.
    """
    namespace = {}
    
    # Add modules directly to namespace
    for entry in NAMESPACE_MODULES:
        namespace[entry.name] = entry.value
    
    # Build safe builtins dict
    builtins = {}
    for entry in NAMESPACE_BUILTINS:
        builtins[entry.name] = entry.value
    namespace["__builtins__"] = builtins
    
    return namespace


def build_prompt_section() -> str:
    """Generate prompt section describing available execution environment.
    
    Returns:
        String to include in LLM system prompt.
    """
    lines = []
    lines.append("**Execution Environment:**")
    lines.append("")
    
    # Restrictions first (most important)
    lines.append("**IMPORTANT RESTRICTIONS:**")
    for restriction in EXECUTION_RESTRICTIONS:
        lines.append(f"- {restriction}")
    lines.append("")
    
    # Available modules/classes
    lines.append("**Available modules/classes:**")
    for entry in NAMESPACE_MODULES:
        examples = ", ".join(f"`{ex}`" for ex in entry.usage_examples[:2])
        lines.append(f"- `{entry.name}`: {entry.description}. Examples: {examples}")
    lines.append("")
    
    # Available builtin functions (grouped by category)
    lines.append("**Available built-in functions:**")
    
    # Group by category
    categories = {}
    for entry in NAMESPACE_BUILTINS:
        if entry.category not in categories:
            categories[entry.category] = []
        categories[entry.category].append(entry.name)
    
    # Format each category
    category_order = ["type_check", "type_convert", "sequence", "math", "string", "constant"]
    for cat in category_order:
        if cat in categories:
            funcs = ", ".join(f"`{name}`" for name in categories[cat])
            cat_name = cat.replace("_", " ").title()
            lines.append(f"- {cat_name}: {funcs}")
    
    return "\n".join(lines)


def get_available_function_names() -> List[str]:
    """Get list of all available function names in the namespace.
    
    Returns:
        List of function/module names available in transform expressions.
    """
    names = [e.name for e in NAMESPACE_MODULES]
    names.extend(e.name for e in NAMESPACE_BUILTINS if callable(e.value))
    return sorted(names)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Transform Namespace Configuration")
    print("=" * 60)
    
    print("\n1. Available Functions:")
    for name in get_available_function_names():
        print(f"   - {name}")
    
    print("\n2. Prompt Section:")
    print("-" * 40)
    print(build_prompt_section())
    
    print("\n3. Test Namespace Build:")
    ns = build_namespace()
    print(f"   Namespace keys: {list(ns.keys())}")
    print(f"   Builtins count: {len(ns['__builtins__'])}")
    
    print("\n4. Test Expressions:")
    test_exprs = [
        ("x.strip()", "  hello  "),
        ("int(x)", "42"),
        ("isinstance(x, str)", "test"),
        ("datetime.fromisoformat(x.replace(' ', 'T'))", "2024-01-15 10:30:00"),
    ]
    for expr, test_val in test_exprs:
        try:
            result = eval(expr, {"x": test_val, **ns})
            print(f"   {expr}: {result}")
        except Exception as e:
            print(f"   {expr}: ERROR - {e}")
