"""
LLM Callbacks - Token Usage Tracking

Custom callbacks to capture token usage even when using with_structured_output().
"""

from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger

from .statistics import record_usage


class TokenUsageCallback(BaseCallbackHandler):
    """
    Callback handler to capture and record token usage from LLM calls.
    
    This callback intercepts LLM responses to extract token usage information
    even when using with_structured_output(), which normally strips metadata.
    """
    
    def __init__(self, log_usage: bool = True):
        """
        Initialize token usage callback.
        
        Args:
            log_usage: Whether to log usage information
        """
        super().__init__()
        self.log_usage = log_usage
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_total_tokens = 0
        self.model_name = "unknown"
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Called when LLM finishes running.
        
        Args:
            response: LLM result containing token usage
            **kwargs: Additional arguments
        """
        if not response.llm_output:
            return
        
        # Try to extract model name
        if "model_name" in response.llm_output:
            self.model_name = response.llm_output["model_name"]
        elif "model" in response.llm_output:
            self.model_name = response.llm_output["model"]
        
        # Extract token usage from different possible locations
        input_tokens = 0
        output_tokens = 0
        
        # Location 1: token_usage in llm_output
        if "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        
        # Location 2: usage in llm_output
        elif "usage" in response.llm_output:
            usage = response.llm_output["usage"]
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        
        # Location 3: Direct fields in llm_output
        elif "prompt_tokens" in response.llm_output:
            input_tokens = response.llm_output.get("prompt_tokens", 0)
            output_tokens = response.llm_output.get("completion_tokens", 0)
        
        # Store for retrieval
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens
        self.last_total_tokens = input_tokens + output_tokens
        
        # Record usage
        if input_tokens > 0 or output_tokens > 0:
            record_usage(self.model_name, input_tokens, output_tokens)
            
        else:
            if self.log_usage:
                logger.warning(
                    f"[Callback] No token usage found in response. "
                    f"llm_output keys: {list(response.llm_output.keys())}"
                )
    

__all__ = ['TokenUsageCallback']
