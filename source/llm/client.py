"""
LLM Client

High-level client for calling LLMs with retry and error handling.
"""

from typing import List, Optional, Any
import os
import json
from datetime import datetime
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from loguru import logger

from .errors import LLMTimeoutError, LLMError
from .retry import retry_with_exponential_backoff
from .statistics import record_usage

# Environment variable for LLM call logging
_LLM_LOG_CALLS = os.getenv('LLM_LOG_CALLS', '0') == '1'
_LLM_LOG_DIR = Path(os.getenv('LLM_LOG_DIR', 'source/logs/calls'))

def _log_llm_call_to_file(
    model_name: str,
    input_data: Any,
    output_data: Any,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None
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
    if not _LLM_LOG_CALLS:
        return
    
    try:
        # Create logs directory if not exists
        _LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp and success/failed suffix
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        status_suffix = 'success' if success else 'failed'
        filename = f"{timestamp}_{model_name.replace('/', '_')}_{status_suffix}.json"
        filepath = _LLM_LOG_DIR / filename
        
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


def call_llm_with_retry(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    max_retries: int = 3,
    timeout: Optional[float] = None,
    log_io: bool = False
) -> Any:
    """
    Call LLM with automatic retry and error handling.
    
    This function wraps LLM invocation with:
    - Automatic retry with exponential backoff
    - Timeout protection
    - Comprehensive error classification
    - Token usage tracking
    - Detailed logging
    
    Args:
        llm: LLM instance to invoke
        messages: List of messages to send
        max_retries: Maximum retry attempts (default: 3)
        timeout: Request timeout in seconds (default: from config)
        log_io: Whether to log input/output details (default: False)
    
    Returns:
        LLM response
    
    Raises:
        LLMTimeoutError: If request times out
        LLMError: For LLM-related errors
    """
    # Use unified timeout from config
    from config.truncation_limits import TruncationLimits
    if timeout is None:
        timeout = TruncationLimits.LLM_TIMEOUT
    
    @retry_with_exponential_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        exponential_base=2.0,
        max_delay=180.0,
        retry_on=(LLMTimeoutError, LLMError)
    )
    def _invoke():
        try:
            # Log input if requested
            if log_io or _LLM_LOG_CALLS:
                input_preview = str(messages)[:200] if messages else "No messages"
                if log_io:
                    logger.debug(f"[LLM Input] {input_preview}...")
            
            # Use streaming to enable early disconnect detection
            # Server can detect client disconnect and stop generation
            import time
            start_time = time.time()
            response = None
            
            # For non-structured output, we need to collect all chunks
            from langchain_core.messages import AIMessageChunk, AIMessage
            collected_content = []
            
            for chunk in llm.stream(messages):
                # Check timeout during streaming
                if timeout and timeout > 0:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        raise LLMTimeoutError(f"LLM request timed out after {timeout} seconds")
                
                # Collect content from chunks
                if hasattr(chunk, 'content') and chunk.content:
                    collected_content.append(chunk.content)
                response = chunk  # Keep last chunk for metadata
            
            # Build final response with collected content
            if collected_content and response is not None:
                # Create a message-like response with full content
                full_content = ''.join(collected_content)
                # Preserve metadata from last chunk
                if hasattr(response, 'response_metadata'):
                    response = AIMessage(
                        content=full_content,
                        response_metadata=response.response_metadata,
                        usage_metadata=getattr(response, 'usage_metadata', None)
                    )
                else:
                    response = AIMessage(content=full_content)
            elif response is None:
                raise LLMError("LLM returned no response")
            
            # Extract token usage if available
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            input_tokens = 0
            output_tokens = 0
            found_usage = False
            
            # Try multiple possible locations for usage information
            # 1. New style: usage_metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0)
                if input_tokens > 0 or output_tokens > 0:
                    found_usage = True
            
            # 2. Response metadata (OpenAI/OpenRouter style)
            if not found_usage and hasattr(response, 'response_metadata') and response.response_metadata:
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)
                    if input_tokens > 0 or output_tokens > 0:
                        found_usage = True
                elif 'usage' in metadata:
                    usage = metadata['usage']
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    if input_tokens > 0 or output_tokens > 0:
                        found_usage = True
            
            # 3. Additional kwargs (some providers)
            if not found_usage and hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                kwargs = response.additional_kwargs
                if 'usage' in kwargs:
                    usage = kwargs['usage']
                    input_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    output_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                    if input_tokens > 0 or output_tokens > 0:
                        found_usage = True
            
            # Record usage if any tokens were found
            if found_usage and (input_tokens > 0 or output_tokens > 0):
                record_usage(model_name, input_tokens, output_tokens)
                logger.debug(
                    f"[LLM Usage] {model_name}: "
                    f"in={input_tokens}, out={output_tokens}, total={input_tokens+output_tokens}"
                )
            else:
                logger.warning(f"[LLM Usage] No token usage found in response for {model_name}")
                logger.debug(f"[LLM Usage] Response attributes: {dir(response)}")
                if hasattr(response, 'response_metadata'):
                    logger.debug(f"[LLM Usage] response_metadata: {response.response_metadata}")
                if hasattr(response, 'usage_metadata'):
                    logger.debug(f"[LLM Usage] usage_metadata: {response.usage_metadata}")
                if hasattr(response, 'additional_kwargs'):
                    logger.debug(f"[LLM Usage] additional_kwargs: {response.additional_kwargs}")
            
            # Log to file if enabled
            if _LLM_LOG_CALLS:
                _log_llm_call_to_file(
                    model_name=model_name,
                    input_data=messages,
                    output_data=response,
                    success=True,
                    metadata={
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'max_retries': max_retries,
                        'timeout': timeout
                    }
                )
            
            # Log output if requested
            if log_io:
                output_preview = str(response.content)[:200] if hasattr(response, 'content') else str(response)[:200]
                logger.debug(f"[LLM Output] {output_preview}...")
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            error_full = str(e)
            
            # Get model name for logging
            model_name_for_log = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            
            # Log failed call to file
            if _LLM_LOG_CALLS:
                _log_llm_call_to_file(
                    model_name=model_name_for_log,
                    input_data=messages,
                    output_data=None,
                    success=False,
                    error_message=error_full,
                    metadata={
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
                # Unknown error - wrap in generic LLMError
                raise LLMError(f"LLM error: {e}") from e
    
    return _invoke()


__all__ = ['call_llm_with_retry']
