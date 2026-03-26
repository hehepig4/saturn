"""
TBox Validation Utilities

Provides validation functions for ontology integrity:
- Cycle detection in class hierarchy
- Parent reference validation
"""

from typing import Dict, List, Optional, Set
from loguru import logger


def detect_class_cycle(classes: List[Dict]) -> Optional[List[str]]:
    """
    Detect cycles in class hierarchy using DFS.
    
    Args:
        classes: List of class dicts with 'name' and 'parent_classes'/'parent_class'/'parent'
        
    Returns:
        None if no cycle found, otherwise a list of class names forming the cycle
        (e.g., ['A', 'B', 'C', 'A'] means A->B->C->A)
    """
    if not classes:
        return None
    
    # Build parent map: child_name -> parent_name
    parent_map: Dict[str, str] = {}
    all_class_names: Set[str] = set()
    
    for cls in classes:
        name = cls.get("name", "")
        if not name:
            continue
        
        all_class_names.add(name)
        
        # Extract parent (handle multiple formats)
        parents = cls.get("parent_classes")
        if parents is None:
            parents = cls.get("parent_class")
        if parents is None:
            parents = cls.get("parent")
        
        # Normalize to single parent
        if isinstance(parents, list):
            parent = parents[0] if parents else None
        else:
            parent = parents
        
        # Only track non-trivial parents (not Column, not empty)
        if parent and parent != "Column" and parent != "":
            parent_map[name] = parent
    
    # DFS cycle detection
    visited: Set[str] = set()
    rec_stack: Set[str] = set()  # Nodes in current recursion path
    
    def dfs(node: str, path: List[str]) -> Optional[List[str]]:
        """
        DFS from node, tracking path for cycle detection.
        
        Returns cycle path if found, None otherwise.
        """
        if node in rec_stack:
            # Found cycle - extract the cycle portion
            try:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            except ValueError:
                return [node, node]  # Fallback
        
        if node in visited:
            return None
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        # Follow parent edge
        parent = parent_map.get(node)
        if parent and parent in all_class_names:
            # Only follow if parent exists in our class set
            result = dfs(parent, path)
            if result:
                return result
        
        rec_stack.remove(node)
        path.pop()
        return None
    
    # Check each class as potential cycle start
    for cls in classes:
        name = cls.get("name", "")
        if name and name not in visited:
            cycle = dfs(name, [])
            if cycle:
                return cycle
    
    return None


def would_create_cycle(
    classes: List[Dict],
    child_name: str,
    proposed_parent: str,
) -> bool:
    """
    Check if setting child_name's parent to proposed_parent would create a cycle.
    
    This is a pre-check before applying an operation.
    
    Args:
        classes: Current class list
        child_name: Name of the class whose parent is being set
        proposed_parent: Proposed parent class name
        
    Returns:
        True if the operation would create a cycle, False otherwise
    """
    if not proposed_parent or proposed_parent == "Column":
        return False  # Column is root, cannot create cycle
    
    if child_name == proposed_parent:
        return True  # Self-loop
    
    # Build parent map from current classes
    parent_map: Dict[str, str] = {}
    
    for cls in classes:
        name = cls.get("name", "")
        if not name:
            continue
        
        parents = cls.get("parent_classes")
        if parents is None:
            parents = cls.get("parent_class")
        if parents is None:
            parents = cls.get("parent")
        
        if isinstance(parents, list):
            parent = parents[0] if parents else None
        else:
            parent = parents
        
        if parent and parent != "Column" and parent != "":
            parent_map[name] = parent
    
    # Simulate the proposed change
    parent_map[child_name] = proposed_parent
    
    # Check if proposed_parent has child_name as ancestor
    # (which would create a cycle)
    visited: Set[str] = set()
    current = proposed_parent
    
    while current and current != "Column" and current not in visited:
        visited.add(current)
        if current == child_name:
            return True  # Found child_name in ancestor chain of proposed_parent
        current = parent_map.get(current)
    
    return False


def log_cycle_detection_result(cycle: Optional[List[str]], context: str = "") -> bool:
    """
    Log cycle detection result and return whether cycle was found.
    
    Args:
        cycle: Result from detect_class_cycle()
        context: Context string for logging (e.g., "after synthesis", "after review")
        
    Returns:
        True if cycle was found (error state), False if no cycle (ok state)
    """
    if cycle:
        cycle_str = " -> ".join(cycle)
        logger.error(f"  ⚠ CYCLE DETECTED {context}: {cycle_str}")
        return True
    else:
        logger.debug(f"  ✓ No cycle detected {context}")
        return False
