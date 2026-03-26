"""
LLM Error Classes

Custom exceptions for LLM operations.
"""


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass


class LLMStructuredOutputError(LLMError):
    """Raised when structured output parsing fails."""
    pass


__all__ = [
    'LLMError',
    'LLMTimeoutError',
    'LLMStructuredOutputError',
]
