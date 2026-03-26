"""
LLM Usage Statistics

Token usage tracking and reporting with caller attribution.
"""

from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from contextvars import ContextVar
from loguru import logger


# Context variable for current caller (set by graph_node decorator or manually)
_current_caller: ContextVar[str] = ContextVar('llm_caller', default='unknown')


def set_current_caller(caller: str):
    """Set the current caller context for LLM tracking."""
    _current_caller.set(caller)


def get_current_caller() -> str:
    """Get the current caller context."""
    return _current_caller.get()


class UsageStats:
    """Token usage statistics tracker with performance metrics and caller attribution."""
    
    def __init__(self):
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        })
        # Track by caller (node/component)
        self.by_caller: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        })
        self.start_time = datetime.now()
        self.last_reset_time = datetime.now()
        # Performance metrics
        self.async_calls = 0
        self.async_latencies = []

    
    def record(self, model_name: str, input_tokens: int, output_tokens: int, caller: Optional[str] = None):
        """Record a single LLM invocation."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        
        model_stats = self.by_model[model_name]
        model_stats["requests"] += 1
        model_stats["input_tokens"] += input_tokens
        model_stats["output_tokens"] += output_tokens
        model_stats["total_tokens"] += (input_tokens + output_tokens)
        
        # Record by caller (use context var if not explicitly provided)
        if caller is None:
            caller = get_current_caller()
        caller_stats = self.by_caller[caller]
        caller_stats["requests"] += 1
        caller_stats["input_tokens"] += input_tokens
        caller_stats["output_tokens"] += output_tokens
        caller_stats["total_tokens"] += (input_tokens + output_tokens)
    
    def record_async(self, latency_ms: float):
        """Record async LLM call performance."""
        self.async_calls += 1
        self.async_latencies.append(latency_ms)
        # Keep only last 1000 entries
        if len(self.async_latencies) > 1000:
            self.async_latencies = self.async_latencies[-1000:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage statistics summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        since_reset = (datetime.now() - self.last_reset_time).total_seconds()
        
        summary = {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": elapsed,
            "since_reset_seconds": since_reset,
            "by_model": dict(self.by_model),
            "by_caller": dict(self.by_caller),  # Per-caller statistics
            # Performance metrics
            "async_calls": self.async_calls,
            "async_latencies": self.async_latencies,
        }
        return summary
    
    def reset(self):
        """Reset statistics."""
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.by_model.clear()
        self.by_caller.clear()
        self.last_reset_time = datetime.now()
        # Reset performance metrics
        self.async_calls = 0
        self.async_latencies = []


# Global instance
_usage_stats = UsageStats()


def get_usage_stats() -> Dict[str, Any]:
    """
    Get token usage statistics.
    
    Returns:
        Dictionary with usage statistics
    """
    return _usage_stats.get_summary()


def record_async_call(latency_ms: float):
    """Record async LLM call performance."""
    _usage_stats.record_async(latency_ms)


def record_usage(model_name: str, input_tokens: int, output_tokens: int, caller: Optional[str] = None):
    """Record token usage for an LLM invocation.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        caller: Optional caller identifier (uses context var if not provided)
    """
    _usage_stats.record(model_name, input_tokens, output_tokens, caller)


def reset_usage_stats():
    """Reset all usage statistics. Call before starting a new measurement period."""
    _usage_stats.reset()


__all__ = [
    'UsageStats',
    'get_usage_stats',
    'record_usage',
    'record_async_call',
    'reset_usage_stats',
    # Caller context management
    'set_current_caller',
    'get_current_caller',
]