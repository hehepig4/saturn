"""
Node Decorators for LangGraph Workflow.

Generic decorators for graph nodes - work with any state type.
Supports both sync and async nodes.

USAGE:
    Use the unified @graph_node decorator for all nodes:
    
    @graph_node()  # Simple processing node with defaults
    def my_node(state):
        return state
    
    @graph_node(node_type="decision", on_error="raise")  # Decision node
    def decide(state):
        return Command(goto='next')
    
    @graph_node(enable_retry=True, require_cache=True)  # Critical node
    def process(state):
        return state
    
    # Async nodes are fully supported
    @graph_node()
    async def async_node(state):
        await some_async_operation()
        return state
"""
import time
import functools
import inspect
import asyncio
from typing import Callable, Optional, Literal, Any, Union, TypeVar, Dict, List
from datetime import datetime
from contextvars import ContextVar

from langgraph.types import Command
from core.formatting.colors import (
    LogColors, colorize, colorize_separator, format_node_header,
    format_decision, colorize_status, format_duration
)

from loguru import logger

# Import LLM caller context management
from llm.statistics import set_current_caller

# Generic state type
StateType = TypeVar('StateType')


# ============================= TRACE CONTEXT =============================

# Context variable to hold current trace information  
_trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('_trace_context', default=None)


class TraceContext:
    """Context manager for adding custom trace information to node execution."""
    
    def __init__(self):
        self.custom_info: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.steps: List[str] = []
        self.warnings: List[str] = []
    
    def add_info(self, key: str, value: Any) -> None:
        """Add custom information to trace."""
        self.custom_info[key] = value
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to trace."""
        self.metrics[name] = value
    
    def add_step(self, description: str) -> None:
        """Add a processing step description."""
        self.steps.append(description)
        logger.info(f"  ➜ {description}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning to trace."""
        self.warnings.append(message)
        logger.warning(f"  ⚠ {message}")


def get_trace_context() -> Optional[TraceContext]:
    """Get the current trace context for the executing node."""
    return _trace_context.get()


# ============================= UNIFIED DECORATOR =============================

def graph_node(
    node_type: Literal["decision", "processing", "finalization", "validation"] = "processing",
    log_level: str = "INFO",
    enable_timing: bool = True,
    enable_trajectory: bool = True,
    enable_state_logging: bool = False,
    enable_retry: bool = False,
    max_retries: int = 3,
    require_cache: bool = False,
    require_concepts: bool = False,
    on_error: Literal["complete", "raise", "continue"] = "complete",
    warn_threshold_seconds: float = 30.0,
):
    """
    Unified decorator for LangGraph nodes with comprehensive features.
    
    Supports both sync and async functions automatically.
    
    Args:
        node_type: Type of node for trajectory tracking
        log_level: Logging level for entry/exit messages
        enable_timing: Whether to log execution time and performance warnings
        enable_trajectory: Whether to track in trajectory log
        enable_state_logging: Whether to log detailed state information
        enable_retry: Whether to retry on failure
        max_retries: Maximum retry attempts (if enable_retry=True)
        require_cache: Whether to validate cache existence
        require_concepts: Whether to validate concepts existence
        on_error: Error handling strategy
        warn_threshold_seconds: Log warning if execution exceeds this time
    """
    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            # Create async wrapper
            @functools.wraps(func)
            async def async_wrapper(state) -> Union[Any, Command]:
                node_name = func.__name__.replace('_', ' ').title()
                start_time = time.time()
                traj_idx = None
                
                # ========== STATE VALIDATION ==========
                if require_cache or require_concepts:
                    validation_errors = []
                    
                    if require_cache and not getattr(state, 'active_cache_ids', []):
                        validation_errors.append("No active caches available")
                    
                    if require_concepts and not getattr(state, 'discovered_concepts', set()):
                        validation_errors.append("No discovered concepts available")
                    
                    if validation_errors:
                        error_msg = f"Validation failed for {node_name}: {'; '.join(validation_errors)}"
                        logger.warning(colorize(error_msg, LogColors.WARNING))
                        if on_error == "raise":
                            raise ValueError(error_msg)
                        elif on_error == "complete":
                            return Command(goto='complete')
                
                # ========== STATE LOGGING (BEFORE) ==========
                if enable_state_logging:
                    iteration = getattr(state, 'current_iteration', 0)
                    logger.log(log_level, f"[{node_name}] State at iteration {iteration}:")
                    logger.log(log_level, f"  Active caches: {len(getattr(state, 'active_cache_ids', []))}")
                    logger.log(log_level, f"  Discovered concepts: {len(getattr(state, 'discovered_concepts', set()))}")
                
                # ========== ENTRY LOGGING WITH COLORS ==========
                iteration = getattr(state, 'current_iteration', 0)
                node_uuid = getattr(state, 'node_uuid', None)
                
                if node_uuid:
                    header = format_node_header(node_name, iteration + 1, node_uuid)
                else:
                    header = format_node_header(node_name, iteration + 1)
                
                logger.log(log_level, header)
                
                # Show ready status
                completed = getattr(state, 'completed', False)
                if not completed:
                    status_msg = colorize_status("Ready", True)
                    logger.log(log_level, f"{colorize('Status:', LogColors.BOLD)} {status_msg}")
                
                # ========== START TRAJECTORY TRACKING ==========
                if enable_trajectory and hasattr(state, 'start_node_execution'):
                    traj_idx = state.start_node_execution(node_name, node_type)
                
                # ========== SETUP TRACE CONTEXT ==========
                trace_ctx = TraceContext()
                _trace_context.set(trace_ctx)
                
                # ========== SET LLM CALLER CONTEXT ==========
                # Use function name as caller for LLM statistics tracking
                set_current_caller(func.__name__)
                
                # ========== RETRY WRAPPER (ASYNC) ==========
                async def execute_with_retry_async(attempt: int = 0):
                    try:
                        # ========== EXECUTE NODE FUNCTION (ASYNC) ==========
                        result = await func(state)
                        
                        # ========== SUCCESS TRACKING ==========
                        if enable_trajectory and traj_idx is not None and hasattr(state, 'complete_node_execution'):
                            decision = None
                            action_summary = None
                            
                            if isinstance(result, Command):
                                decision = f"GOTO: {result.goto}" if result.goto else "COMMAND"
                                if hasattr(result, 'update') and result.update:
                                    plan = result.update.get('current_plan', {})
                                    if isinstance(plan, dict) and 'action' in plan:
                                        action_summary = f"Action: {plan['action']}"
                            
                            state.complete_node_execution(
                                entry_index=traj_idx,
                                success=True,
                                decision=decision,
                                action_summary=action_summary
                            )
                            
                            # If returning Command, include trajectory update
                            if isinstance(result, Command):
                                new_entries = state.trajectory_log[traj_idx:]
                                merged_update = dict(result.update) if result.update else {}
                                merged_update['trajectory_log'] = new_entries
                                result = Command(goto=result.goto, update=merged_update, graph=result.graph)
                        
                        # ========== EXIT LOGGING ==========
                        elapsed = time.time() - start_time
                        
                        if enable_timing:
                            duration_str = format_duration(elapsed)
                            
                            if elapsed > warn_threshold_seconds:
                                logger.warning(
                                    f"{node_name} took {duration_str} (exceeds threshold {warn_threshold_seconds}s)"
                                )
                            else:
                                logger.log(log_level, colorize_status(f"{node_name} completed in {duration_str}", True))
                        else:
                            logger.log(log_level, colorize_status(f"{node_name} completed", True))
                        
                        return result
                        
                    except Exception as e:
                        # ========== RETRY LOGIC ==========
                        if enable_retry and attempt < max_retries:
                            delay = 1.0 * (2.0 ** attempt)
                            logger.warning(
                                colorize(
                                    f"Retry {attempt + 1}/{max_retries} for {node_name} after {delay:.1f}s (error: {e})",
                                    LogColors.WARNING
                                )
                            )
                            await asyncio.sleep(delay)
                            return await execute_with_retry_async(attempt + 1)
                        
                        # ========== ERROR HANDLING ==========
                        elapsed = time.time() - start_time
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        
                        duration_str = format_duration(elapsed)
                        logger.error(colorize_status(f"{node_name} failed after {duration_str}", False))
                        logger.exception(colorize(f"Error in {node_name}: {e}", LogColors.ERROR))
                        
                        # Track failure in trajectory
                        if enable_trajectory and traj_idx is not None and hasattr(state, 'complete_node_execution'):
                            state.complete_node_execution(
                                entry_index=traj_idx,
                                success=False,
                                error=error_msg
                            )
                        
                        # Apply error handling strategy
                        if on_error == "complete":
                            logger.warning(colorize(f"Routing to 'complete' due to error in {node_name}", LogColors.WARNING))
                            
                            if enable_trajectory and traj_idx is not None:
                                new_entries = state.trajectory_log[traj_idx:]
                                cmd = Command(goto='complete', update={'trajectory_log': new_entries, 'error': error_msg})
                            else:
                                cmd = Command(goto='complete', update={'error': error_msg})
                            
                            return cmd
                        elif on_error == "raise":
                            raise
                        else:  # continue
                            logger.warning(colorize(f"Continuing with current state despite error in {node_name}", LogColors.WARNING))
                            return state
                
                try:
                    return await execute_with_retry_async()
                finally:
                    _trace_context.set(None)
            
            return async_wrapper
        
        else:
            # Create sync wrapper
            @functools.wraps(func)
            def sync_wrapper(state) -> Union[Any, Command]:
                node_name = func.__name__.replace('_', ' ').title()
                start_time = time.time()
                traj_idx = None
                
                # ========== STATE VALIDATION ==========
                if require_cache or require_concepts:
                    validation_errors = []
                    
                    if require_cache and not getattr(state, 'active_cache_ids', []):
                        validation_errors.append("No active caches available")
                    
                    if require_concepts and not getattr(state, 'discovered_concepts', set()):
                        validation_errors.append("No discovered concepts available")
                    
                    if validation_errors:
                        error_msg = f"Validation failed for {node_name}: {'; '.join(validation_errors)}"
                        logger.warning(colorize(error_msg, LogColors.WARNING))
                        if on_error == "raise":
                            raise ValueError(error_msg)
                        elif on_error == "complete":
                            return Command(goto='complete')
                
                # ========== STATE LOGGING (BEFORE) ==========
                if enable_state_logging:
                    iteration = getattr(state, 'current_iteration', 0)
                    logger.log(log_level, f"[{node_name}] State at iteration {iteration}:")
                    logger.log(log_level, f"  Active caches: {len(getattr(state, 'active_cache_ids', []))}")
                    logger.log(log_level, f"  Discovered concepts: {len(getattr(state, 'discovered_concepts', set()))}")
                
                # ========== ENTRY LOGGING WITH COLORS ==========
                iteration = getattr(state, 'current_iteration', 0)
                node_uuid = getattr(state, 'node_uuid', None)
                
                if node_uuid:
                    header = format_node_header(node_name, iteration + 1, node_uuid)
                else:
                    header = format_node_header(node_name, iteration + 1)
                
                logger.log(log_level, header)
                
                # Show ready status
                completed = getattr(state, 'completed', False)
                if not completed:
                    status_msg = colorize_status("Ready", True)
                    logger.log(log_level, f"{colorize('Status:', LogColors.BOLD)} {status_msg}")
                
                # ========== START TRAJECTORY TRACKING ==========
                if enable_trajectory and hasattr(state, 'start_node_execution'):
                    traj_idx = state.start_node_execution(node_name, node_type)
                
                # ========== SETUP TRACE CONTEXT ==========
                trace_ctx = TraceContext()
                _trace_context.set(trace_ctx)
                
                # ========== SET LLM CALLER CONTEXT ==========
                # Use function name as caller for LLM statistics tracking
                set_current_caller(func.__name__)
                
                # ========== RETRY WRAPPER (SYNC) ==========
                def execute_with_retry(attempt: int = 0):
                    try:
                        # ========== EXECUTE NODE FUNCTION (SYNC) ==========
                        result = func(state)
                        
                        # ========== SUCCESS TRACKING ==========
                        if enable_trajectory and traj_idx is not None and hasattr(state, 'complete_node_execution'):
                            decision = None
                            action_summary = None
                            
                            if isinstance(result, Command):
                                decision = f"GOTO: {result.goto}" if result.goto else "COMMAND"
                                if hasattr(result, 'update') and result.update:
                                    plan = result.update.get('current_plan', {})
                                    if isinstance(plan, dict) and 'action' in plan:
                                        action_summary = f"Action: {plan['action']}"
                            
                            state.complete_node_execution(
                                entry_index=traj_idx,
                                success=True,
                                decision=decision,
                                action_summary=action_summary
                            )
                            
                            # If returning Command, include trajectory update
                            if isinstance(result, Command):
                                new_entries = state.trajectory_log[traj_idx:]
                                merged_update = dict(result.update) if result.update else {}
                                merged_update['trajectory_log'] = new_entries
                                result = Command(goto=result.goto, update=merged_update, graph=result.graph)
                        
                        # ========== EXIT LOGGING ==========
                        elapsed = time.time() - start_time
                        
                        if enable_timing:
                            duration_str = format_duration(elapsed)
                            
                            if elapsed > warn_threshold_seconds:
                                logger.warning(
                                    f"{node_name} took {duration_str} (exceeds threshold {warn_threshold_seconds}s)"
                                )
                            else:
                                logger.log(log_level, colorize_status(f"{node_name} completed in {duration_str}", True))
                        else:
                            logger.log(log_level, colorize_status(f"{node_name} completed", True))
                        
                        return result
                        
                    except Exception as e:
                        # ========== RETRY LOGIC ==========
                        if enable_retry and attempt < max_retries:
                            delay = 1.0 * (2.0 ** attempt)
                            logger.warning(
                                colorize(
                                    f"Retry {attempt + 1}/{max_retries} for {node_name} after {delay:.1f}s (error: {e})",
                                    LogColors.WARNING
                                )
                            )
                            time.sleep(delay)
                            return execute_with_retry(attempt + 1)
                        
                        # ========== ERROR HANDLING ==========
                        elapsed = time.time() - start_time
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        
                        duration_str = format_duration(elapsed)
                        logger.error(colorize_status(f"{node_name} failed after {duration_str}", False))
                        logger.exception(colorize(f"Error in {node_name}: {e}", LogColors.ERROR))
                        
                        # Track failure in trajectory
                        if enable_trajectory and traj_idx is not None and hasattr(state, 'complete_node_execution'):
                            state.complete_node_execution(
                                entry_index=traj_idx,
                                success=False,
                                error=error_msg
                            )
                        
                        # Apply error handling strategy
                        if on_error == "complete":
                            logger.warning(colorize(f"Routing to 'complete' due to error in {node_name}", LogColors.WARNING))
                            
                            if enable_trajectory and traj_idx is not None:
                                new_entries = state.trajectory_log[traj_idx:]
                                cmd = Command(goto='complete', update={'trajectory_log': new_entries, 'error': error_msg})
                            else:
                                cmd = Command(goto='complete', update={'error': error_msg})
                            
                            return cmd
                        elif on_error == "raise":
                            raise
                        else:  # continue
                            logger.warning(colorize(f"Continuing with current state despite error in {node_name}", LogColors.WARNING))
                            return state
                
                try:
                    return execute_with_retry()
                finally:
                    _trace_context.set(None)
            
            return sync_wrapper
    
    return decorator


# ============================= END OF NODE DECORATORS =============================
# All nodes should use @graph_node with appropriate parameters.
