"""
Logging Color Configuration for Console Output

Provides ANSI color codes and styling for different log elements
to improve readability of console output.

Originally from utilities/log_colors.py
Moved to core/formatting/colors.py
"""

from typing import Optional

class LogColors:
    """ANSI color codes for terminal output."""
    
    # Reset
    RESET = '\033[0m'
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Semantic colors for different log elements
    NODE_NAME = BOLD + BRIGHT_CYAN  # Node names
    UUID = BRIGHT_BLACK  # UUIDs and IDs
    SUCCESS = BRIGHT_GREEN  # Success messages
    ERROR = BRIGHT_RED  # Error messages
    WARNING = BRIGHT_YELLOW  # Warning messages
    INFO = BRIGHT_BLUE  # Info messages
    
    ITERATION = BRIGHT_MAGENTA  # Iteration numbers
    DECISION = BOLD + BRIGHT_YELLOW  # Agent decisions
    ACTION = BRIGHT_GREEN  # Actions taken
    CACHE_ID = CYAN  # Cache IDs
    CONCEPT = BOLD + MAGENTA  # Concepts
    HYPEREDGE = BLUE  # Hyperedge IDs
    
    METRIC = YELLOW  # Metrics and numbers
    TIMESTAMP = DIM + WHITE  # Timestamps
    DURATION = BRIGHT_CYAN  # Duration/time
    
    PLAN = BOLD + CYAN  # Plan information
    INTENT = ITALIC + MAGENTA  # Intent information
    INSIGHT = BOLD + GREEN  # Review insights
    
    SEPARATOR = DIM + WHITE  # Separators like ===
    ARROW = BRIGHT_BLUE  # Arrows and connectors
    
    # Tool-specific colors
    TOOL_CALL = CYAN  # Tool calls
    TOOL_RESULT = GREEN  # Tool results
    
    # Status colors
    STATUS_RUNNING = YELLOW
    STATUS_COMPLETED = GREEN
    STATUS_FAILED = RED

def colorize(text: str, color: str) -> str:
    """
    Colorize text with ANSI color codes.
    
    Args:
        text: Text to colorize
        color: Color code from LogColors
        
    Returns:
        Colorized text with reset at the end
    """
    return f"{color}{text}{LogColors.RESET}"

def colorize_uuid(uuid: str, show_full: bool = False) -> str:
    """
    Colorize UUID with dimmed color and optional shortening.
    
    Args:
        uuid: UUID string
        show_full: Whether to show full UUID or shortened version
        
    Returns:
        Colorized UUID string
    """
    from core.identifiers import shorten_id
    
    display_id = uuid if show_full else shorten_id(uuid)
    return colorize(display_id, LogColors.UUID)

def colorize_status(status: str, success: bool = True) -> str:
    """
    Colorize status message based on success/failure.
    
    Args:
        status: Status message
        success: Whether the status indicates success
        
    Returns:
        Colorized status string
    """
    color = LogColors.SUCCESS if success else LogColors.ERROR
    symbol = "✓" if success else "✗"
    return colorize(f"{symbol} {status}", color)

def colorize_separator(char: str = "=", length: int = 60) -> str:
    """
    Create a colorized separator line.
    
    Args:
        char: Character to use for separator
        length: Length of separator
        
    Returns:
        Colorized separator string
    """
    return colorize(char * length, LogColors.SEPARATOR)

def format_node_header(node_name: str, iteration: int, node_uuid: Optional[str] = None) -> str:
    """
    Format a colored node execution header.
    
    Args:
        node_name: Name of the node
        iteration: Current iteration number
        node_uuid: Optional UUID of the node being processed
        
    Returns:
        Formatted header string with colors
    """
    lines = [
        colorize_separator(),
        f"{colorize('NODE:', LogColors.BOLD)} {colorize(node_name, LogColors.NODE_NAME)} "
        f"({colorize('Iteration', LogColors.DIM)} {colorize(str(iteration), LogColors.ITERATION)})"
    ]
    
    if node_uuid:
        lines.append(f"{colorize('UUID:', LogColors.DIM)} {colorize_uuid(node_uuid)}")
    
    lines.append(colorize_separator())
    
    return "\n".join(lines)

def format_decision(decision: str, reasoning: Optional[str] = None) -> str:
    """
    Format a colored decision message.
    
    Args:
        decision: Decision made
        reasoning: Optional reasoning for the decision
        
    Returns:
        Formatted decision string with colors
    """
    result = f"{colorize('DECISION:', LogColors.BOLD)} {colorize(decision, LogColors.DECISION)}"
    
    if reasoning:
        # Truncate long reasoning
        if len(reasoning) > 150:
            reasoning = reasoning[:147] + "..."
        result += f"\n{colorize('→', LogColors.ARROW)} {colorize(reasoning, LogColors.DIM)}"
    
    return result

def format_duration(seconds: float) -> str:
    """
    Format a colored duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    return colorize(f"{seconds:.2f}s", LogColors.DURATION)

__all__ = [
    'LogColors',
    'colorize',
    'colorize_uuid',
    'colorize_status',
    'colorize_separator',
    'format_node_header',
    'format_decision',
    'format_duration',
]
