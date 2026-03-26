"""
LLM Package

LLM management, retry logic, and usage statistics.

Originally from utilities/llm_manager.py, now reorganized into:
- errors.py: LLM exception classes
- statistics.py: Usage tracking
- manager.py: LLM singleton manager
- retry.py: Retry with exponential backoff
- client.py: High-level LLM client
"""

from .errors import (
    LLMError,
    LLMTimeoutError,
)

from .statistics import (
    UsageStats,
    get_usage_stats,
    record_usage,
)

from .manager import (
    LLMManager,
    get_llm,
    clear_llm_cache,
)

from .retry import (
    retry_with_exponential_backoff,
)

from .client import (
    call_llm_with_retry,
)

from .invoke_with_stats import (
    invoke_structured_llm,
    invoke_structured_llm_with_retry,
)

from .ebnf_constraint import (
    pydantic_to_ebnf,
    invoke_with_ebnf_constraint,
)

from .async_client import (
    invoke_llm_async,
    invoke_structured_llm_with_retry_async,
)

__all__ = [
    # Errors
    'LLMError',
    'LLMTimeoutError',
    
    # Statistics
    'UsageStats',
    'get_usage_stats',
    'record_usage',
    
    # Manager
    'LLMManager',
    'get_llm',
    'clear_llm_cache',
    
    # Retry
    'retry_with_exponential_backoff',
    
    # Client
    'call_llm_with_retry',
    
    # Invoke with stats
    'invoke_structured_llm',
    'invoke_structured_llm_with_retry',
    
    # EBNF Constraint
    'pydantic_to_ebnf',
    'invoke_with_ebnf_constraint',
    
    # Async Client
    'invoke_llm_async',
    'invoke_structured_llm_with_retry_async',
]
