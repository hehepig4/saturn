"""
Unified Tree Formatter for Hierarchical Data

Provides consistent tree-style formatting for:
- Classes (primitive_classes with parent_classes)
- DataProperties (with parent_properties)

All tree printing in the codebase should use this utility.
"""

from typing import Any, Callable, Dict, List, Optional, Union


def format_hierarchy_tree(
    items: List[Dict[str, Any]],
    name_key: str = "name",
    parent_key: str = "parent_classes",
    description_key: str = "description",
    extra_info_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    root_name: Optional[str] = None,
    max_desc_length: Optional[int] = 80,
    indent_size: int = 2,
    show_root: bool = True,
) -> str:
    """
    Format a list of hierarchical items as a tree structure.
    
    Args:
        items: List of dicts with hierarchical relationships
        name_key: Key for item name (default: "name")
        parent_key: Key for parent list (default: "parent_classes")
        description_key: Key for description (default: "description")
        extra_info_fn: Optional function to extract extra info (e.g., domain, range_type)
        root_name: If provided, all items without parents are children of this root
        max_desc_length: Max characters for description (default: 80, None for no truncation)
        indent_size: Number of spaces per indent level (default: 2)
        show_root: Whether to show the root node (default: True)
        
    Returns:
        Formatted tree string
        
    Example output:
        Column
        ├── TemporalColumn: Abstract category for time-related columns
        │   ├── YearColumn: Represents year values
        │   └── DateColumn: Represents date values
        ├── DescriptiveColumn: Abstract category for textual columns
        │   ├── NameColumn: Represents names
        │   │   ├── PersonNameColumn: Names of individuals
        │   │   └── OrganizationNameColumn: Names of organizations
        │   └── TitleColumn: Titles of creative works
        └── QuantitativeColumn: Abstract category for numeric columns
    """
    if not items:
        return ""
    
    def normalize_name(name: str) -> str:
        """Remove common prefixes (upo:, owl:, etc.) for matching purposes."""
        if ':' in name:
            return name.split(':', 1)[1]
        return name
    
    # Build lookup maps - use normalized names for matching, but keep original for display
    # name_to_item: normalized_name -> item (for lookup)
    # original_names: normalized_name -> original_name (for display)
    name_to_item = {}
    original_names = {}
    for item in items:
        orig_name = item.get(name_key, "")
        norm_name = normalize_name(orig_name)
        name_to_item[norm_name] = item
        original_names[norm_name] = orig_name
    
    children_map: Dict[str, List[str]] = {}  # normalized parent -> [normalized children]
    root_items: List[str] = []  # normalized names
    
    for item in items:
        name = normalize_name(item.get(name_key, ""))
        parents = item.get(parent_key, [])
        
        # Handle various parent formats
        if parents is None:
            parents = []
        elif isinstance(parents, str):
            parents = [parents] if parents else []
        elif hasattr(parents, 'tolist'):  # numpy array
            parents = parents.tolist()
        
        # Normalize parent names too
        parents = [normalize_name(p) for p in parents]
        
        if not parents:
            root_items.append(name)
        else:
            for parent in parents:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(name)
    
    def format_node(name: str, prefix: str = "", is_last: bool = True, is_root: bool = False, visited: set = None) -> List[str]:
        """Recursively format a node and its children.
        
        Args:
            name: Normalized name (without prefix) for lookup
            visited: Set of already visited nodes to detect cycles
        """
        # Initialize visited set on first call
        if visited is None:
            visited = set()
        
        # Cycle detection: skip if already visited in current path
        if name in visited:
            return [f"{prefix}└── **{name}** [CYCLE DETECTED - skipping]"]
        
        visited = visited | {name}  # Create new set to avoid cross-branch pollution
        
        lines = []
        item = name_to_item.get(name, {})
        # Use original display name if available, otherwise use normalized name
        display_name = original_names.get(name, name)
        # Strip upo: prefix for cleaner display
        if display_name.startswith('upo:'):
            display_name = display_name[4:]
        
        desc = item.get(description_key, "")
        # Only truncate if max_desc_length is set (not None)
        if desc and max_desc_length is not None and len(desc) > max_desc_length:
            desc = desc[:max_desc_length - 3] + "..."
        
        # Build the line
        if is_root:
            # Root node - no prefix
            line = f"**{display_name}**"
            if desc:
                line += f": {desc}"
        else:
            # Child node - with tree characters
            branch = "└── " if is_last else "├── "
            line = f"{prefix}{branch}**{display_name}**"
            if desc:
                line += f": {desc}"
        
        # Add extra info if provided
        if extra_info_fn:
            extra = extra_info_fn(item)
            if extra:
                lines.append(line)
                # Add extra info on next line with proper indentation
                if is_root:
                    extra_prefix = " " * indent_size
                else:
                    continuation = "    " if is_last else "│   "
                    extra_prefix = prefix + continuation
                lines.append(f"{extra_prefix}{extra}")
                line = None  # Already added
        
        if line:
            lines.append(line)
        
        # Process children
        children = children_map.get(name, [])
        children = sorted(children)
        
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            if is_root:
                child_prefix = ""
            else:
                continuation = "    " if is_last else "│   "
                child_prefix = prefix + continuation
            
            lines.extend(format_node(child, child_prefix, child_is_last, is_root=False, visited=visited))
        
        return lines
    
    result_lines = []
    
    # If root_name is provided, treat all root_items as children of root_name
    if root_name and show_root:
        # Normalize root_name for matching
        norm_root_name = normalize_name(root_name)
        result_lines.append(f"**{root_name}**")
        
        # Merge root_items with any existing children of root_name
        # (items whose parent is root_name are already in children_map[norm_root_name])
        existing_children = children_map.get(norm_root_name, [])
        
        # Exclude root_name itself from root_items to avoid duplication
        filtered_root_items = [r for r in root_items if r != norm_root_name]
        all_children = sorted(set(existing_children + filtered_root_items))
        
        # Format children of root
        for i, child in enumerate(all_children):
            child_is_last = (i == len(all_children) - 1)
            result_lines.extend(format_node(child, "", child_is_last, is_root=False))
    else:
        # Format each root item
        for i, root in enumerate(sorted(root_items)):
            is_last = (i == len(root_items) - 1)
            result_lines.extend(format_node(root, "", is_last, is_root=True))
            if not is_last:
                result_lines.append("")  # Blank line between root trees
    
    return "\n".join(result_lines)


def format_class_hierarchy(
    classes: List[Dict[str, Any]],
    include_column_root: bool = True,
    max_desc_length: Optional[int] = 80,
) -> str:
    """
    Format primitive classes as a hierarchy tree.
    
    Args:
        classes: List of class dicts with 'name', 'parent_classes', 'description'
        include_column_root: Whether to show Column as root (default: True)
        max_desc_length: Max description length (None for no truncation)
        
    Returns:
        Formatted tree string
    """
    return format_hierarchy_tree(
        items=classes,
        name_key="name",
        parent_key="parent_classes",
        description_key="description",
        root_name="Column" if include_column_root else None,
        max_desc_length=max_desc_length,
        show_root=include_column_root,
    )


# Convenience function for simple class list (no hierarchy info available)
def format_class_list(
    class_names: List[str],
    prefix: str = "- ",
) -> str:
    """
    Format a simple list of class names.
    
    Args:
        class_names: List of class name strings
        prefix: Prefix for each line (default: "- ")
        
    Returns:
        Formatted list string
    """
    return "\n".join(f"{prefix}{name}" for name in sorted(class_names))
