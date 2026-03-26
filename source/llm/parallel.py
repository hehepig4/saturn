"""
LLM Parallel Invocation Tools

Provides thread-based parallel execution for multiple LLM calls.
Automatically inherits caching, retry, and statistics from invoke_structured_llm.

Performance:
- Sequential: N calls × 500ms = N/2 seconds  
- Parallel (max_workers=5): N calls in ~500ms (5-10x speedup)

Usage:
    from llm.parallel import invoke_llm_parallel_with_func
    
    # Parallel LLM calls with custom function
    results = invoke_llm_parallel_with_func(process_func, tasks, max_workers=5)
"""

from typing import List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from loguru import logger
import time


class LLMServiceUnavailableError(Exception):
    """Raised when LLM service becomes unavailable (e.g., OOM, server crash)."""
    pass


def invoke_llm_parallel_with_func(
    func: Callable[[Any], Any],
    tasks: List[Any],
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    max_consecutive_errors: int = 10,
    max_error_rate: float = 0.5,
    fail_fast: bool = True,
) -> List[Any]:
    """
    Execute custom function in parallel on multiple tasks.
    
    More generic than invoke_llm_parallel - allows custom processing functions
    that internally call LLMs.
    
    Args:
        func: Function to execute on each task (should accept 1 argument)
        tasks: List of task inputs (passed to func)
        max_workers: Max parallel workers
        show_progress: Show progress updates
        max_consecutive_errors: Max consecutive errors before aborting (default: 10)
        max_error_rate: Max error rate (0-1) before aborting (default: 0.5)
        fail_fast: If True, raise exception on threshold breach; if False, return partial results
    
    Returns:
        List of results (same order as tasks)
        Exceptions are returned as Exception objects
    
    Raises:
        LLMServiceUnavailableError: When consecutive errors exceed threshold
    
    Example:
        >>> def process_cluster(cluster):
        ...     # This function internally calls invoke_structured_llm
        ...     return generate_stories_for_cluster(cluster, domain="vision")
        >>> 
        >>> clusters = [cluster1, cluster2, cluster3]
        >>> results = invoke_llm_parallel_with_func(
        ...     process_cluster,
        ...     clusters,
        ...     max_workers=3
        ... )
    """
    if not tasks:
        return []
    
    # Fixed at 256 for high throughput with GPU inference servers
    if max_workers is None:
        max_workers = 128
    
    logger.info(
        f"[Parallel Executor] Processing {len(tasks)} tasks "
        f"with {max_workers} workers..."
    )
    
    start_time = time.time()
    
    # Create indexed tasks
    indexed_tasks = list(enumerate(tasks))
    
    # Execute in parallel
    results = [None] * len(tasks)
    completed_count = 0
    error_count = 0
    consecutive_errors = 0
    should_abort = False
    abort_reason = ""
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, task): idx
            for idx, task in indexed_tasks
        }
        
        pending = set(future_to_index.keys())
        
        while pending and not should_abort:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            
            if done:
                
                for future in done:
                    idx = future_to_index[future]
                    try:
                        result = future.result(timeout=0)
                        results[idx] = result
                        completed_count += 1
                        consecutive_errors = 0
                    except Exception as e:
                        logger.warning(
                            f"[Parallel Executor] Task {idx+1} failed: "
                            f"{type(e).__name__}: {str(e)[:100]}"
                        )
                        results[idx] = e
                        error_count += 1
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            should_abort = True
                            abort_reason = f"Consecutive errors ({consecutive_errors}) exceeded threshold ({max_consecutive_errors})"
                            logger.error(f"[Parallel Executor] 🛑 ABORTING: {abort_reason}")
                            for f in pending:
                                f.cancel()
                            pending = set()
                            break
                    
                    total_processed = completed_count + error_count
                    if total_processed >= 20:
                        current_error_rate = error_count / total_processed
                        if current_error_rate > max_error_rate:
                            should_abort = True
                            abort_reason = f"Error rate ({current_error_rate:.1%}) exceeded threshold ({max_error_rate:.1%})"
                            logger.error(f"[Parallel Executor] 🛑 ABORTING: {abort_reason}")
                            for f in pending:
                                f.cancel()
                            pending = set()
                            break
                    
                    if show_progress and total_processed % 5 == 0:
                        logger.info(
                            f"[Parallel Executor] Progress: {total_processed}/{len(tasks)} "
                            f"({completed_count} succeeded, {error_count} failed)"
                        )
    finally:
        executor.shutdown(wait=True)
    
    elapsed = time.time() - start_time
    
    if should_abort:
        logger.error(
            f"[Parallel Executor] ❌ ABORTED after {elapsed:.2f}s - "
            f"{completed_count} succeeded, {error_count} failed. "
            f"Reason: {abort_reason}"
        )
        if fail_fast:
            raise LLMServiceUnavailableError(
                f"LLM service unavailable: {abort_reason}. "
                f"Processed {completed_count + error_count}/{len(tasks)} tasks before abort."
            )
    else:
        logger.info(
            f"[Parallel Executor] ✓ Completed {len(tasks)} tasks in {elapsed:.2f}s "
            f"({completed_count} succeeded, {error_count} failed)"
        )
    
    return results


__all__ = [
    'invoke_llm_parallel_with_func',
    'LLMServiceUnavailableError',
]
