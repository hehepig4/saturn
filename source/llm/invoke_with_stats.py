"""
LLM Invocation with Statistics

Wrapper functions for invoking LLMs with automatic usage tracking and retry.
"""

from typing import Any, Optional, Dict, Callable
import hashlib
import os
import json
from datetime import datetime
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from .callbacks import TokenUsageCallback
from .retry import retry_with_exponential_backoff
from .errors import LLMTimeoutError, LLMError, LLMStructuredOutputError
from .statistics import get_usage_stats, record_usage

# LLM response cache (in-memory)
_llm_response_cache: Dict[str, Any] = {}
_cache_stats = {"hits": 0, "misses": 0}


def clear_llm_response_cache():
    """Clear the in-memory LLM response cache and reset cache stats.

    Must be called between experiment variants to prevent cross-contamination
    of cached responses across different experimental conditions.
    """
    _llm_response_cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


# Environment variable for LLM call logging
# NOTE: These are read at module load time. For runtime override, use
# functions like enable_llm_logging() / disable_llm_logging()
_DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / 'logs' / 'llm_calls'
_LLM_LOG_CALLS = os.getenv('LLM_LOG_CALLS', '0') == '1'
_LLM_LOG_DIR = Path(os.getenv('LLM_LOG_DIR', '')) if os.getenv('LLM_LOG_DIR') else _DEFAULT_LOG_DIR

def _get_llm_log_config() -> tuple:
    """Get current LLM logging configuration.
    
    Returns:
        (enabled: bool, log_dir: Path)
    
    This reads environment variables at call time, allowing runtime changes.
    """
    enabled = os.getenv('LLM_LOG_CALLS', '0') == '1'
    log_dir_str = os.getenv('LLM_LOG_DIR', '')
    log_dir = Path(log_dir_str) if log_dir_str else _DEFAULT_LOG_DIR
    return enabled, log_dir

def _log_llm_call_to_file(
    model_name: str,
    input_data: Any,
    output_data: Any,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log LLM call to file if LLM_LOG_CALLS=1.
    
    Args:
        model_name: Name of the LLM model
        input_data: Input prompt or messages
        output_data: Response from LLM (or None for failed calls)
        success: Whether the call succeeded
        error_message: Error message if call failed
        metadata: Additional metadata to log
    """
    enabled, log_dir = _get_llm_log_config()
    if not enabled:
        return
    
    try:
        # Create logs directory if not exists
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp and success/failed suffix
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        status_suffix = 'success' if success else 'failed'
        filename = f"{timestamp}_{model_name.replace('/', '_')}_{status_suffix}.json"
        filepath = log_dir / filename
        
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'success': success,
            'input': str(input_data),
            'output': str(output_data) if output_data is not None else None,
            'error_message': error_message,
            'metadata': metadata or {}
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        log_level = "debug" if success else "warning"
        getattr(logger, log_level)(f"[LLM Call Log] {'Success' if success else 'Failed'} - Saved to {filepath}")
    
    except Exception as e:
        logger.warning(f"Failed to log LLM call to file: {e}")

def _compute_cache_key(prompt: str, model_name: str) -> str:
    """Compute cache key for LLM prompt."""
    key_str = f"{model_name}:{prompt}"
    return hashlib.md5(key_str.encode()).hexdigest()


def invoke_structured_llm(
    llm: BaseChatModel,
    prompt: str,
    max_retries: int = 3,
    timeout: float = None,
    log_io: bool = False,
    log_usage: bool = True,
    use_cache: bool = True
) -> Any:
    """
    Invoke a structured LLM with automatic retry, timeout, and usage tracking.
    
    Args:
        llm: LangChain LLM instance (should have with_structured_output)
        prompt: Input prompt string
        max_retries: Maximum number of retries on failure
        timeout: Timeout in seconds
        log_io: Whether to log input/output
        log_usage: Whether to log token usage
        use_cache: Whether to use response cache
        
    Returns:
        Structured output from LLM
    """
    # Use unified timeout from config
    from config.truncation_limits import TruncationLimits
    if timeout is None:
        timeout = TruncationLimits.LLM_TIMEOUT
    
    # Check cache first
    model_name = getattr(llm, "model_name", "unknown")
    cache_key = _compute_cache_key(prompt, model_name)
    
    if use_cache and cache_key in _llm_response_cache:
        _cache_stats["hits"] += 1
        if log_io:
            logger.debug(f"[LLM Cache Hit] Key: {cache_key[:16]}...")
        return _llm_response_cache[cache_key]
    
    _cache_stats["misses"] += 1
    
    # Track usage before call
    stats_before = get_usage_stats()
    tokens_before = stats_before['total_tokens']
    
    # Create callback for usage tracking
    callback = TokenUsageCallback(log_usage=log_usage)
    config = RunnableConfig(callbacks=[callback])
    
    # Create messages
    messages = [HumanMessage(content=prompt)]
    
    # Define the invoke function with retry logic
    @retry_with_exponential_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        exponential_base=2.0,
        max_delay=180.0,
        retry_on=(LLMTimeoutError, LLMError)
    )
    def _invoke_with_retry():
        try:
            if log_io or _LLM_LOG_CALLS:
                input_preview = prompt[:200] if len(prompt) > 200 else prompt
                if log_io:
                    logger.debug(f"[LLM Input] {input_preview}...")
            
            # Use invoke() instead of stream() to ensure on_llm_end callback
            # fires properly with token usage. Streaming bypasses the callback's
            # llm_output, causing token counts to be 0.
            import time
            start_time = time.time()
            
            response = llm.invoke(messages, config)
            
            if response is None:
                raise LLMError("LLM returned no response")
            
            # Post-call timeout warning (actual timeout handled by HTTP client)
            if timeout and timeout > 0:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning(f"LLM call took {elapsed:.1f}s (exceeded timeout={timeout}s)")
            
            # Log to file if enabled
            if _LLM_LOG_CALLS:
                _log_llm_call_to_file(
                    model_name=model_name,
                    input_data=prompt,
                    output_data=response,
                    success=True,
                    metadata={
                        'cache_key': cache_key[:16],
                        'use_cache': use_cache,
                        'max_retries': max_retries,
                        'timeout': timeout
                    }
                )
            
            if log_io:
                output_preview = str(response)[:200] if len(str(response)) > 200 else str(response)
                logger.debug(f"[LLM Output] {output_preview}...")
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            error_full = str(e)
            
            # Log failed call to file
            if _LLM_LOG_CALLS:
                _log_llm_call_to_file(
                    model_name=model_name,
                    input_data=prompt,
                    output_data=None,
                    success=False,
                    error_message=error_full,
                    metadata={
                        'cache_key': cache_key[:16],
                        'use_cache': use_cache,
                        'max_retries': max_retries,
                        'timeout': timeout,
                        'error_type': type(e).__name__
                    }
                )
            
            # Classify error types
            if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                raise LLMError(f"Rate limit exceeded: {e}") from e
            elif 'timeout' in error_msg or 'timed out' in error_msg:
                raise LLMTimeoutError(f"Request timed out: {e}") from e
            elif 'api' in error_msg or 'server' in error_msg or '500' in error_msg or '503' in error_msg:
                raise LLMError(f"API error: {e}") from e
            else:
                raise LLMError(f"LLM error: {e}") from e
    
    # Execute with retry
    try:
        response = _invoke_with_retry()
        
        # Cache successful response
        if use_cache:
            _llm_response_cache[cache_key] = response
            # Limit cache size to 1000 entries (FIFO)
            if len(_llm_response_cache) > 1000:
                oldest_key = next(iter(_llm_response_cache))
                del _llm_response_cache[oldest_key]
        
        # Log usage if requested
        if log_usage:
            stats_after = get_usage_stats()
            tokens_used = stats_after['total_tokens'] - tokens_before
            input_tokens = stats_after['total_input_tokens'] - stats_before['total_input_tokens']
            output_tokens = stats_after['total_output_tokens'] - stats_before['total_output_tokens']
            
            if tokens_used > 0:
                # logger.debug(
                #     f"[LLM Stats] This call: "
                #     f"in={input_tokens}, out={output_tokens}, total={tokens_used}"
                # )
                pass
        
        return response
        
    except Exception as e:
        logger.error(f"[LLM Error] {type(e).__name__}: {e}")
        raise


def invoke_structured_llm_with_retry(
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
    Invoke structured LLM with retry on parsing failures.
    
    When structured output parsing fails (e.g., JSON format error, missing fields),
    this function will retry with increased temperature to get different outputs.
    
    This solves the problem of deterministic failures when temperature=0.
    
    Args:
        llm_factory: Function that takes temperature and returns LLM instance
                    Example: lambda t: get_llm_by_purpose("default", temperature_override=t)
        output_schema: Pydantic model class for structured output
        prompt: Input prompt string
        max_retries: Maximum number of retries on parsing failure (default 3)
        timeout: Timeout in seconds (default 60)
        log_io: Whether to log input/output
        log_usage: Whether to log token usage
        use_cache: Whether to use response cache (will be disabled on retries)
        retry_temperature_increment: Temperature increase per retry (default 0.3)
        use_ebnf: Whether to use EBNF constraint instead of JSON Schema (default False)
                  EBNF is more efficient for large schemas with repeated Literal types.
        ebnf_grammar: Optional custom EBNF grammar string. If None and use_ebnf=True,
                      will auto-generate from output_schema using pydantic_to_ebnf().
        
    Returns:
        Structured output from LLM
        
    Raises:
        LLMStructuredOutputError: If all retries fail
        
    Example:
        >>> from llm.manager import get_llm_by_purpose
        >>> result = invoke_structured_llm_with_retry(
        ...     llm_factory=lambda t: get_llm_by_purpose("default"),
        ...     output_schema=TableAnnotation,
        ...     prompt="Annotate this table...",
        ... )
    """
    # Use unified timeout from config
    from config.truncation_limits import TruncationLimits
    if timeout is None:
        timeout = TruncationLimits.LLM_TIMEOUT
    
    last_error = None
    attempt_temperatures = []
    
    # Helper function to check if base_url supports EBNF constraint
    def _supports_ebnf(url: str) -> bool:
        """Check if the API endpoint supports EBNF constraint.
        
        EBNF constraint is only supported by local SGLang/vLLM servers.
        Remote APIs (OpenRouter, OpenAI, Gemini, etc.) do not support it.
        """
        if not url:
            return False
        url_lower = url.lower()
        # Local servers that support EBNF
        local_patterns = ['localhost', '127.0.0.1', '10.120.', '192.168.', '/v1']
        # Remote APIs that don't support EBNF
        remote_patterns = ['openrouter.ai', 'openai.com', 'googleapis.com', 'anthropic.com']
        
        # Check if it's a remote API
        for pattern in remote_patterns:
            if pattern in url_lower:
                return False
        
        # Check if it looks like a local server
        for pattern in local_patterns:
            if pattern in url_lower:
                return True
        
        # Default to False for unknown URLs (safer)
        return False
    
    # EBNF mode: use direct OpenAI API with EBNF grammar constraint
    if use_ebnf:
        from .ebnf_constraint import pydantic_to_ebnf, invoke_with_ebnf_constraint
        
        # Get LLM to check base_url first (before generating grammar)
        test_llm = llm_factory(0)
        test_base_url = getattr(test_llm, 'openai_api_base', None) or getattr(test_llm, 'base_url', '')
        
        if not _supports_ebnf(test_base_url):
            logger.debug(f"EBNF not supported for {test_base_url}, falling back to JSON Schema mode")
            use_ebnf = False  # Fall back to standard mode
        else:
            # Generate EBNF grammar if not provided
            grammar = ebnf_grammar if ebnf_grammar else pydantic_to_ebnf(output_schema)
    
    # Helper function to classify error types
    def _classify_error(error_str: str) -> str:
        """Classify error type: 'infra', 'parsing', or 'other'."""
        error_lower = error_str.lower()
        
        # Infra errors (transient, should retry with wait)
        infra_keywords = [
            'timeout', 'timed out', 'rate limit', 'too many requests',
            '429', '503', '502', '500', 'overloaded', 'connection',
            'network', 'unavailable', 'service temporarily',
        ]
        if any(kw in error_lower for kw in infra_keywords):
            return 'infra'
        
        # Parsing errors (should retry with temp increase)
        parsing_keywords = [
            'parse', 'parsing', 'json', 'validation', 'invalid',
            'missing', 'required', 'field', 'schema', 'pydantic',
            'decode', 'format', 'output',
        ]
        if any(kw in error_lower for kw in parsing_keywords):
            return 'parsing'
        
        return 'other'
    
    # EBNF mode: use direct OpenAI API with EBNF grammar constraint
    if use_ebnf:
        import time
        infra_retry_delay = 3.0  # Base delay for infra errors
        
        for attempt in range(max_retries + 1):
            temperature = attempt * retry_temperature_increment
            attempt_temperatures.append(temperature)
            
            try:
                # Get LLM to extract config (base_url, api_key, model)
                llm = llm_factory(temperature)
                
                # Extract config from LangChain LLM
                base_url = getattr(llm, 'openai_api_base', None) or getattr(llm, 'base_url', 'http://127.0.0.1:8000/v1')
                api_key = getattr(llm, 'openai_api_key', None) or 'token-abcd123'
                model = getattr(llm, 'model_name', None) or 'default'
                max_tokens = getattr(llm, 'max_tokens', None)
                
                # Handle SecretStr
                if hasattr(api_key, 'get_secret_value'):
                    api_key = api_key.get_secret_value()
                
                # Use EBNF constraint invocation
                result = invoke_with_ebnf_constraint(
                    prompt=prompt,
                    output_schema=output_schema,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    ebnf_grammar=grammar,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                if result is None:
                    raise LLMStructuredOutputError("LLM returned None")
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = _classify_error(str(e))
                
                if error_type == 'infra':
                    # Infra errors (timeout often caused by repetitive output): retry with wait + temperature increase
                    if attempt < max_retries:
                        wait_time = infra_retry_delay * (2 ** attempt)
                        next_temp = (attempt + 1) * retry_temperature_increment
                        logger.warning(
                            f"[EBNF Retry] Attempt {attempt + 1}/{max_retries + 1} infra error at temp={temperature:.2f}: {e}. "
                            f"Waiting {wait_time:.0f}s, then retry with temp={next_temp:.2f}..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"[EBNF Retry] All {max_retries + 1} attempts failed (infra). Last error: {e}")
                
                elif error_type == 'parsing':
                    # Parsing errors: retry with temperature increase
                    if attempt < max_retries:
                        logger.warning(
                            f"[EBNF Retry] Attempt {attempt + 1}/{max_retries + 1} failed at temp={temperature:.2f}: "
                            f"{str(e)[:100]}. Retrying with temp={temperature + retry_temperature_increment:.2f}..."
                        )
                        continue
                    else:
                        logger.error(f"[EBNF Retry] All {max_retries + 1} attempts failed. Last error: {e}")
                
                else:
                    # Unknown errors: raise immediately
                    logger.warning(f"[EBNF Retry] Attempt {attempt + 1}/{max_retries + 1} failed (unknown): {e}")
                    raise
        
        raise LLMStructuredOutputError(
            f"EBNF structured output failed after {max_retries + 1} attempts. Last error: {last_error}"
        )
    
    # Standard mode: use LangChain with_structured_output
    import time
    infra_retry_delay = 3.0  # Base delay for infra errors
    
    for attempt in range(max_retries + 1):
        # Calculate temperature: start with 0, then increment on each retry
        temperature = attempt * retry_temperature_increment
        attempt_temperatures.append(temperature)
        
        try:
            # Get LLM with current temperature
            llm = llm_factory(temperature)
            structured_llm = llm.with_structured_output(output_schema)
            
            # Use cache only on first attempt (temperature=0)
            # Retries with higher temperature should not use cache
            use_cache_this_attempt = use_cache and (attempt == 0)
            
            result = invoke_structured_llm(
                structured_llm,
                prompt,
                max_retries=1,  # Don't use internal retry (we handle it here)
                timeout=timeout,
                log_io=log_io,
                log_usage=log_usage,
                use_cache=use_cache_this_attempt,
            )
            
            # Validate result is not None and has expected type
            if result is None:
                raise LLMStructuredOutputError("LLM returned None")
            
            return result
            
        except Exception as e:
            last_error = e
            error_type = _classify_error(str(e))
            
            if error_type == 'infra':
                # Infra errors: retry with wait (exponential backoff)
                if attempt < max_retries:
                    wait_time = infra_retry_delay * (2 ** attempt)
                    logger.warning(
                        f"[Structured Retry] Attempt {attempt + 1}/{max_retries + 1} infra error: {e}. "
                        f"Waiting {wait_time:.0f}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"[Structured Retry] All {max_retries + 1} attempts failed (infra). "
                        f"Last error: {e}"
                    )
            
            elif error_type == 'parsing':
                # Parsing errors: retry with temperature increase
                if attempt < max_retries:
                    logger.warning(
                        f"[Structured Retry] Attempt {attempt + 1}/{max_retries + 1} failed at temp={temperature:.2f}: "
                        f"{str(e)[:100]}. Retrying with temp={temperature + retry_temperature_increment:.2f}..."
                    )
                    continue
                else:
                    logger.error(
                        f"[Structured Retry] All {max_retries + 1} attempts failed. "
                        f"Temperatures tried: {attempt_temperatures}. "
                        f"Last error: {e}"
                    )
            
            else:
                # Unknown errors: raise immediately
                logger.warning(f"[Structured Retry] Attempt {attempt + 1}/{max_retries + 1} failed (unknown): {e}")
                raise
    
    # All retries exhausted
    raise LLMStructuredOutputError(
        f"Structured output parsing failed after {max_retries + 1} attempts "
        f"(temperatures: {attempt_temperatures}). Last error: {last_error}"
    )


__all__ = [
    'invoke_structured_llm',
    'invoke_structured_llm_with_retry',
    'LLMStructuredOutputError',
]
