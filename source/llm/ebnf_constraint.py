"""
EBNF Grammar Generator for Structured LLM Output.

This module provides utilities to generate EBNF grammars from Pydantic models,
enabling efficient FSM-based constrained decoding with XGrammar/SGLang.

Key Benefits over JSON Schema:
- EBNF allows rule reuse (shared rules compiled once)
- Significantly smaller grammar size for models with repeated Literal/Enum types
- Faster FSM compilation (e.g., 17s vs timeout for 142 fields × 41 enum options)

Usage:
    from llm.ebnf_constraint import invoke_with_ebnf_constraint
    
    result = invoke_with_ebnf_constraint(
        prompt="Your prompt here",
        output_schema=YourPydanticModel,
        base_url="http://localhost:8000/v1",
        api_key="your-key",
    )
"""
import json
from typing import Optional, Type
from pydantic import BaseModel
from loguru import logger


def pydantic_to_ebnf(model: Type[BaseModel], compact: bool = True) -> str:
    """
    Convert a Pydantic model to EBNF grammar string using XGrammar.
    
    This uses XGrammar's from_json_schema for reliable grammar generation.
    XGrammar handles all the escaping and format requirements correctly.
    
    Args:
        model: Pydantic model class
        compact: If True, replace unbounded whitespace rules with bounded
                 whitespace (0-4 chars) to prevent excessive output while
                 still allowing formatted JSON (default: True)
        
    Returns:
        EBNF grammar string compatible with XGrammar/SGLang
    """
    import xgrammar  # type: ignore[import-not-found]
    
    # Get JSON schema from Pydantic model
    json_schema = model.model_json_schema()
    json_schema_str = json.dumps(json_schema)
    
    # Use XGrammar to convert JSON schema to grammar
    grammar = xgrammar.Grammar.from_json_schema(json_schema_str)
    ebnf_str = str(grammar)
    
    if compact:
        # Replace unbounded whitespace [ \n\t]* with bounded (0-4 chars).
        # Removing whitespace entirely causes degenerate output (", ") on
        # Qwen3 models. Bounded whitespace allows formatted JSON while
        # preventing infinite whitespace loops.
        _WS_BOUNDED = r'([ \n\t] ([ \n\t] ([ \n\t] ([ \n\t])?)?)?)?'
        ebnf_str = ebnf_str.replace(r'[ \n\t]*', _WS_BOUNDED)
    
    return ebnf_str


def invoke_with_ebnf_constraint(
    prompt: str,
    output_schema: Type[BaseModel],
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "token-abcd123",
    model: str = "default",
    timeout: float = None,
    system_message: Optional[str] = None,
    ebnf_grammar: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> BaseModel:
    """
    Invoke LLM with EBNF constraint and return parsed Pydantic model.
    
    This is a drop-in replacement for `llm.with_structured_output(schema).invoke(prompt)`.
    
    Args:
        prompt: User prompt
        output_schema: Pydantic model class for output
        base_url: SGLang/vLLM API base URL
        api_key: API key
        model: Model name
        timeout: Request timeout in seconds (default: from config)
        system_message: Optional system message
        ebnf_grammar: Optional pre-generated EBNF grammar (if None, auto-generate)
        temperature: Sampling temperature (default 0.0)
        max_tokens: Maximum tokens in response (None for no limit)
        **kwargs: Additional parameters passed to the API
        
    Returns:
        Parsed Pydantic model instance
        
    Raises:
        ValueError: If parsing fails
        TimeoutError: If request times out
    """
    import openai
    
    # Use unified timeout from config
    from config.truncation_limits import TruncationLimits
    if timeout is None:
        timeout = TruncationLimits.LLM_TIMEOUT
    
    # Generate EBNF grammar if not provided
    grammar = ebnf_grammar if ebnf_grammar else pydantic_to_ebnf(output_schema)
    
    logger.debug(f"Using EBNF grammar ({len(grammar)} chars)")
    
    # Build messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    # Call API with EBNF constraint
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    
    # Build API call parameters
    api_params = {
        "model": model,
        "messages": messages,
        "extra_body": {"ebnf": grammar},
        "timeout": timeout,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    
    try:
        response = client.chat.completions.create(**api_params)
    except openai.APITimeoutError as e:
        raise TimeoutError(f"LLM request timed out after {timeout}s") from e
    
    # Record token usage to global UsageStats
    from .statistics import record_usage
    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0
        record_usage(model, input_tokens, output_tokens)
    
    # Parse response
    content = response.choices[0].message.content
    
    # Remove control characters (except \n, \r, \t which are valid in JSON)
    def remove_control_characters(s: str) -> str:
        """Remove control characters that are invalid in JSON strings."""
        import re
        # Remove \u0000-\u001F except \t(\u0009), \n(\u000A), \r(\u000D)
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
    
    # Try to parse JSON, with fallback to fix invalid escapes
    def fix_invalid_json_escapes(s: str) -> str:
        r"""
        Fix invalid JSON escape sequences by double-escaping the backslash.
        
        JSON only allows: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        LLMs often output regex escapes like \d, \s, \w which are invalid.
        We convert \d -> \\d etc.
        """
        valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_char = s[i + 1]
                if next_char in valid_escapes:
                    # Valid escape - keep as is
                    result.append(s[i])
                    result.append(next_char)
                    i += 2
                    # Handle \uXXXX
                    if next_char == 'u' and i + 4 <= len(s):
                        result.extend(s[i:i+4])
                        i += 4
                else:
                    # Invalid escape like \d - double the backslash
                    result.append('\\\\')
                    result.append(next_char)
                    i += 2
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)
    
    # First, remove control characters
    content = remove_control_characters(content)
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try fixing invalid escapes (common with regex patterns)
        try:
            fixed_content = fix_invalid_json_escapes(content)
            data = json.loads(fixed_content)
            logger.debug("Fixed invalid JSON escapes in LLM response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {content[:500]}...")
            raise ValueError(f"Invalid JSON response: {e}") from e
    
    try:
        return output_schema.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to validate model: {e}")
        raise ValueError(f"Failed to validate response: {e}") from e
