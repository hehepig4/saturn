"""
LLM Retry Logic

Exponential backoff retry decorator for LLM calls.
"""

import time
from typing import Callable
from functools import wraps
from loguru import logger


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 180.0,
    retry_on: tuple = (Exception,)
):
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        retry_on: Tuple of exception types to retry on
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retry_on as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    current_delay = min(delay * (exponential_base ** attempt), max_delay)
                    
                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)[:100]}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    time.sleep(current_delay)
                    continue
            
            # All retries exhausted
            logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


__all__ = ['retry_with_exponential_backoff']
