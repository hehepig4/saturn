"""
Async LLM Client

Provides asynchronous LLM invocation with batching and concurrency control.
Reuses existing llm.invoke_with_stats caching and statistics.

Performance gains:
- Sequential: N calls × 500ms = N/2 seconds
- Async batch: N calls in ~500ms with asyncio.gather

Usage:
    import asyncio
    from llm.async_client import invoke_llm_async
    
    result = await invoke_llm_async(llm, prompt)
    
    # With structured output retry
    from llm.async_client import invoke_structured_llm_with_retry_async
    result = await invoke_structured_llm_with_retry_async(llm_factory, schema, prompt)
"""

import asyncio
import contextvars
from typing import Any, Optional, Callable
from langchain_core.language_models import BaseChatModel
import time

from llm.invoke_with_stats import invoke_structured_llm, invoke_structured_llm_with_retry
from llm.statistics import record_async_call
from llm.errors import LLMTimeoutError


async def invoke_llm_async(
    llm: BaseChatModel,
    prompt: str,
    max_retries: int = 3,
    timeout: float = None,
    log_io: bool = False,
    log_usage: bool = True,
    use_cache: bool = True
) -> Any:
    """
    Async wrapper for invoke_structured_llm.
    
    Runs synchronous LLM call in thread pool to avoid blocking event loop.
    Uses asyncio.wait_for to enforce timeout at async level.
    
    Args:
        llm: LLM instance
        prompt: Prompt string
        max_retries: Max retries
        timeout: Timeout in seconds (default: from config)
        log_io: Log I/O
        log_usage: Log usage
        use_cache: Use response cache
    
    Returns:
        LLM response
        
    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """
    from llm.errors import LLMTimeoutError
    from config.truncation_limits import TruncationLimits
    
    # Use unified timeout from config
    if timeout is None:
        timeout = TruncationLimits.LLM_TIMEOUT
    
    start_time = time.time()
    
    try:
        # Run sync function in thread pool (avoid blocking event loop)
        loop = asyncio.get_event_loop()
        
        # Capture current context (includes caller tracking ContextVars)
        # run_in_executor does NOT auto-propagate context in this Python build
        ctx = contextvars.copy_context()
        
        # Wrap in asyncio.wait_for to enforce timeout
        async def _invoke():
            return await loop.run_in_executor(
                None,  # Use default executor
                ctx.run, invoke_structured_llm,
                llm, prompt, max_retries, timeout, log_io, log_usage, use_cache
            )
        
        # Apply timeout at async level (adds safety layer)
        result = await asyncio.wait_for(_invoke(), timeout=timeout)
        
        # Record async call latency
        latency_ms = (time.time() - start_time) * 1000
        record_async_call(latency_ms)
        
        return result
        
    except asyncio.TimeoutError:
        # Convert asyncio.TimeoutError to LLMTimeoutError
        raise LLMTimeoutError(f"Async LLM request timed out after {timeout} seconds")


async def invoke_structured_llm_with_retry_async(
    llm_factory: Callable[[float], BaseChatModel],
    output_schema: type,
    prompt: str,
    max_retries: int = 2,
    timeout: float = None,
    log_io: bool = False,
    log_usage: bool = True,
    use_cache: bool = True,
    retry_temperature_increment: float = 0.8,
    use_ebnf: bool = True,
    ebnf_grammar: Optional[str] = None,
) -> Any:
    """
    Async version of invoke_structured_llm_with_retry.
    
    Retries structured output with increasing temperature on parsing failures.
    Runs synchronous function in thread pool to avoid blocking event loop.
    
    Args:
        llm_factory: Function that takes temperature and returns LLM instance
        output_schema: Pydantic model class for structured output
        prompt: Input prompt string
        max_retries: Maximum number of retries on parsing failure
        timeout: Timeout in seconds (default: from config)
        log_io: Whether to log input/output
        log_usage: Whether to log token usage
        use_cache: Whether to use response cache
        retry_temperature_increment: Temperature increase per retry
        use_ebnf: Whether to use EBNF constraint instead of JSON Schema
        ebnf_grammar: Optional custom EBNF grammar string
        
    Returns:
        Structured output from LLM
        
    Raises:
        LLMStructuredOutputError: If all retries fail
    """
    # Use default timeout from config if not specified
    if timeout is None:
        from config.truncation_limits import TruncationLimits
        timeout = TruncationLimits.LLM_TIMEOUT
    
    start_time = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
        # Capture current context (includes caller tracking ContextVars)
        # run_in_executor does NOT auto-propagate context in this Python build
        ctx = contextvars.copy_context()
        
        def _invoke_sync():
            return invoke_structured_llm_with_retry(
                llm_factory=llm_factory,
                output_schema=output_schema,
                prompt=prompt,
                max_retries=max_retries,
                timeout=timeout,
                log_io=log_io,
                log_usage=log_usage,
                use_cache=use_cache,
                retry_temperature_increment=retry_temperature_increment,
                use_ebnf=use_ebnf,
                ebnf_grammar=ebnf_grammar,
            )
        
        async def _invoke():
            return await loop.run_in_executor(None, ctx.run, _invoke_sync)
        
        result = await asyncio.wait_for(_invoke(), timeout=timeout * (max_retries + 1))
        
        latency_ms = (time.time() - start_time) * 1000
        record_async_call(latency_ms)
        
        return result
        
    except asyncio.TimeoutError:
        raise LLMTimeoutError(
            f"Async structured LLM request timed out after {timeout * (max_retries + 1)} seconds"
        )


__all__ = [
    'invoke_llm_async',
    'invoke_structured_llm_with_retry_async',
]
