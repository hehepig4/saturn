"""
Query Generator for BIRDIE - Using TLlama via vLLM API (Faithful to Official Implementation)

This script is a FAITHFUL reimplementation of the official BIRDIE query_g.py,
adapted to use TLlama via vLLM's OpenAI-compatible API.

Key Features (matching official implementation):
1. TSA (Table Sampling Algorithm): Each LLM call uses a DIFFERENT random sample 
   of table rows to increase query diversity
2. Uses n=20 parameter to generate 20 responses per LLM call (if supported)
3. Iteratively samples until enough valid questions are collected
4. Maintains a sample_list pool that gets depleted and reset

The vLLM server should be started separately with LoRA support:
    CUDA_VISIBLE_DEVICES=2 conda run -n vllm2 python -m vllm.entrypoints.openai.api_server \
        --model /path/to/Llama-3-8B-table-base \
        --port 8001 --host 0.0.0.0 \
        --enable-lora --max-loras 1 --max-lora-rank 64 \
        --lora-modules "fetaqa=/path/to/lora_adapter/fetaqa"

Usage:
    python query_g_tllama_vllm_v2.py --dataset_name "fetaqa" \
                                     --tableid_path "path/to/id_map.json" \
                                     --table_data_path "path/to/table_data.json" \
                                     --num 20 \
                                     --out_train_path "output/dir" \
                                     --vllm_url "http://localhost:8001" \
                                     --model_name "fetaqa"
"""

import json
from tqdm import tqdm
import random
import os
import argparse
import re
import time
from typing import List, Tuple, Optional
from openai import OpenAI

# Default vLLM API settings
DEFAULT_VLLM_URL = "http://localhost:8001"
DEFAULT_MODEL_NAME = "fetaqa"


def json_to_markdown(json_data: dict) -> str:
    """Convert table JSON to markdown format (faithful to official implementation)."""
    data = json_data
    columns = [col['text'] for col in data['columns']]
    rows = [row['cells'] for row in data['rows']]
    
    # Header
    markdown_table = '|'
    for col in columns:
        if col == '':
            col = ' '
        markdown_table += col + '|'
    
    # Separator
    markdown_table += '\n|' + '|'.join(['---'] * len(columns)) + '|'
    
    # Rows
    for row in rows:
        markdown_table += '\n|'
        for cell in row:
            markdown_table += cell['text'] + '|'
    
    return markdown_table


def estimate_token_count(text: str) -> int:
    """Estimate token count (simple heuristic: ~4 chars per token for English)."""
    return len(text) // 4


def get_limit_prompt(table_prompt: str, token_count: int, max_tokens: int = 2048) -> Tuple[List[str], List[str], int]:
    """
    Truncate table to fit token limit (faithful to official implementation).
    
    Returns:
        header: First two lines (column headers and separator)
        rows: All original rows (for sampling pool)
        row_num: Number of rows that fit within token limit
    """
    split_prompt = table_prompt.split("\n")
    header = split_prompt[:2]
    rows = split_prompt[2:]
    
    if not rows:
        return header, rows, 0
    
    excess_tokens = token_count - max_tokens
    row_tokens = estimate_token_count("\n".join(rows))
    row_token_count = row_tokens / len(rows)
    
    num_rows_to_remove = int(excess_tokens / row_token_count) + 1
    num_rows_to_remove = min(num_rows_to_remove, len(rows))
    
    # Return the original rows list (for sampling), and how many can fit
    return header, rows, len(rows) - num_rows_to_remove


def is_valid_format(answer: str) -> bool:
    """Check if answer is in valid JSON format with 'question' key."""
    try:
        answer_dict = json.loads(answer)
        return isinstance(answer_dict, dict) and 'question' in answer_dict
    except (json.JSONDecodeError, TypeError):
        return False


def extract_question_from_response(response_text: str) -> Optional[str]:
    """Extract question from LLM response."""
    if not response_text:
        return None
    
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith('```'):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    
    # Try to parse as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'question' in data:
            q = data['question']
            if q and isinstance(q, str) and len(q.strip()) > 5:
                return q.strip()
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract question using regex patterns
    # Pattern 1: "question": "..."
    pattern1 = r'"question"\s*:\s*"([^"]+)"'
    match = re.search(pattern1, text)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Lines ending with "?"
    for line in text.split('\n'):
        line = line.strip()
        if line.endswith('?') and len(line) > 15:
            return line
    
    return None


class TLlamaQueryGenerator:
    """
    Query generator using TLlama via vLLM API.
    Faithful to the official BIRDIE implementation with TSA (Table Sampling Algorithm).
    """
    
    def __init__(self, vllm_url: str, model_name: str, timeout: int = 120, api_key: str = "token-abc123"):
        """Initialize the vLLM API client."""
        self.vllm_url = vllm_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        
        # Initialize OpenAI client
        # Note: vllm_url should be base URL like http://localhost:8001/v1
        # OpenAI client expects base_url to end with /v1
        base_url = self.vllm_url if self.vllm_url.endswith('/v1') else f"{self.vllm_url}/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Test connection
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"Connected to vLLM server. Available models: {model_ids}")
            
            if model_name not in model_ids:
                print(f"Warning: Model '{model_name}' not in available models. Will try anyway.")
        except Exception as e:
            print(f"Warning: Could not connect to vLLM server: {e}")
    
    def call_gpt_multiple(
        self,
        instruction: str,
        document_title: str,
        table_prompt_info: Tuple[List[str], List[str], int],
        n: int = 20,
        temperature: float = 1.0,
        max_retries: int = None  # Will default to n + 10 if not specified
    ) -> List[str]:
        """
        Generate multiple queries for a table using TSA (Table Sampling Algorithm).
        
        This is FAITHFUL to the official BIRDIE implementation:
        - Each LLM call uses a DIFFERENT random sample of rows
        - sample_list is depleted after each call and reset when exhausted
        - Continues until n valid questions are collected or max_retries reached
        
        Args:
            instruction: The prompt instruction
            document_title: Table caption/title
            table_prompt_info: Tuple of (header_lines, all_rows, row_num_to_sample)
            n: Number of questions to generate
            temperature: LLM sampling temperature
            max_retries: Maximum retry attempts
        
        Returns:
            List of generated questions
        """
        header_prompt, all_rows, row_num = table_prompt_info
        
        # Default max_retries - with batch_n=10, we need fewer retries
        # Each retry can generate up to 10 unique questions
        if max_retries is None:
            max_retries = (n // 10) + 5  # Much fewer retries needed with batch generation
        
        # Initialize sample_list with all rows (this is the TSA pool)
        sample_list = list(all_rows)
        
        def get_prompt(sample_list_ref: List[str]) -> Tuple[str, List[str]]:
            """
            Build prompt with a random sample of rows.
            TSA: Each call uses different rows, depleting the pool.
            """
            # Determine how many rows to sample
            sample_num = min(5, row_num) if row_num > 0 else 0
            
            if len(sample_list_ref) >= sample_num and sample_num > 0:
                # Sample rows and remove them from the pool
                result_sample = random.sample(sample_list_ref, sample_num)
                sample_list_ref = [s for s in sample_list_ref if s not in result_sample]
            else:
                # Pool exhausted, reset it (TSA behavior)
                sample_list_ref = list(all_rows)
                if sample_num > 0 and sample_list_ref:
                    result_sample = random.sample(sample_list_ref, min(sample_num, len(sample_list_ref)))
                    sample_list_ref = [s for s in sample_list_ref if s not in result_sample]
                else:
                    result_sample = []
            
            # Build the table prompt with sampled rows
            table_prompt = "\n".join(header_prompt + result_sample)
            prompt_base = f" tableCaption: {document_title}\ntable: {table_prompt}"
            prompt = instruction + "\n\n" + prompt_base
            
            return prompt, sample_list_ref
        
        # First call
        prompt, sample_list = get_prompt(sample_list)
        
        unique_answers = []
        retries = 0
        
        # Batch size for n parameter - vLLM supports n>1 for parallel generation
        batch_n = min(20, n)  # Generate up to 20 responses per API call for efficiency
        
        try:
            # First call with batch generation
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=batch_n,  # Use batch generation for efficiency
                top_p=1.0,
                max_tokens=512
            )
            
            # Extract questions from all responses
            for choice in response.choices:
                content = choice.message.content
                question = extract_question_from_response(content)
                if question and question not in unique_answers:
                    unique_answers.append(question)
        except Exception as e:
            print(f"First call error: {e}")
        
        # Continue until we have enough questions (TSA: each iteration uses different rows)
        while len(unique_answers) < n and retries < max_retries:
            # TSA: Get new prompt with DIFFERENT sampled rows
            prompt, sample_list = get_prompt(sample_list)
            
            # Calculate remaining queries needed
            remaining = n - len(unique_answers)
            current_batch = min(batch_n, max(1, remaining + 3))  # Request a few extra to account for duplicates
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    n=current_batch,
                    top_p=1.0,
                    max_tokens=512
                )
                
                for choice in response.choices:
                    content = choice.message.content
                    question = extract_question_from_response(content)
                    if question and question not in unique_answers:
                        unique_answers.append(question)
                        
            except Exception as e:
                print(f"Retry {retries} error: {e}")
            
            retries += 1
        
        # Return exactly n questions (or fewer if not enough generated)
        return random.sample(unique_answers, n) if len(unique_answers) > n else unique_answers


def get_origin_id(table_path: str) -> dict:
    """Get original table ID mapping (faithful to official implementation)."""
    with open(table_path, 'r') as table_f:
        content = table_f.read()
    
    # Try to parse as single JSON object (dict format)
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            table_ids = list(data.keys())
            return {tid: i for i, tid in enumerate(table_ids)}
    except json.JSONDecodeError:
        pass
    
    # Fall back to JSONL format
    table_ids = []
    for line in content.strip().split('\n'):
        if line.strip():
            table_data_f = json.loads(line)
            table_ids.append(table_data_f.get('tableId', table_data_f.get('tableID', '')))
    return {tid: i for i, tid in enumerate(table_ids)}


def get_semantic_id(tableid_path: str) -> dict:
    """Get semantic ID mapping (faithful to official implementation)."""
    with open(tableid_path, 'r') as file:
        content = file.read()
    
    table_id_semantic_map = {}
    
    # Try to parse as JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                table_id_semantic_map[item['tableID']] = item['semantic_id']
            return table_id_semantic_map
    except json.JSONDecodeError:
        pass
    
    # Fall back to JSONL format
    for line in content.strip().split('\n'):
        if line.strip():
            try:
                data = json.loads(line.strip())
                table_id_semantic_map[data['tableID']] = data['semantic_id']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return table_id_semantic_map


def create_args():
    parser = argparse.ArgumentParser(
        description='Generate synthetic queries for tables using TLlama via vLLM API (Faithful to BIRDIE)'
    )
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--tableid_path', type=str, required=True, help='Path to semantic ID mapping file')
    parser.add_argument('--table_data_path', type=str, required=True, help='Path to table data file')
    parser.add_argument('--num', type=int, default=20, help='Number of queries per table')
    parser.add_argument('--out_train_path', type=str, required=True, help='Output directory')
    parser.add_argument('--run_tag', type=str, default='', help='Tag for output file name')
    parser.add_argument('--table_max_token', type=int, default=2048, help='Max tokens for table')
    
    # vLLM API configuration
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL,
                        help='Base URL of vLLM server')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Model name to use (base model or LoRA adapter name)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default 1.0 matches official)')
    parser.add_argument('--timeout', type=int, default=120, help='Request timeout in seconds')
    parser.add_argument('--api_key', type=str, default='token-abc123', help='API key for vLLM server')
    
    return parser.parse_args()


def main():
    args = create_args()
    
    # Instruction (faithful to official: uses the last instruction)
    instruction = '''Please generate the probable question a user would ask based on this table, ensuring the answer can be found in some cells in this table or via an aggregation operator such as (Max,Min,Avg,Count). The generated question should contain the explicit information of this table (such as the information of the caption) so that given the question and a repository of tables, this table can be successfully found.

Return the final result in JSON format as {"question":""}.'''

    # Create output directory
    output_dir = os.path.join(args.out_train_path, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = TLlamaQueryGenerator(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        timeout=args.timeout,
        api_key=args.api_key
    )
    
    # Load mappings
    origin_id_map = get_origin_id(args.table_data_path)
    semantic_id_map = get_semantic_id(args.tableid_path)
    
    # Output file
    run_tag = f"_{args.run_tag}" if args.run_tag else ""
    train_data_out_path = os.path.join(output_dir, f"train_query{run_tag}.json")
    
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {train_data_out_path}")
    print(f"Queries per table: {args.num}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Max table tokens: {args.table_max_token}")
    print()
    
    # Load table data
    with open(args.table_data_path, 'r') as table_f:
        content = table_f.read()
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            tables = [(tid, tdata) for tid, tdata in data.items()]
        elif isinstance(data, list):
            tables = [(t.get('tableId', t.get('tableID', '')), t) for t in data]
        else:
            raise ValueError(f"Unexpected table data format: {type(data)}")
    except json.JSONDecodeError:
        tables = []
        for line in content.strip().split('\n'):
            if line.strip():
                t = json.loads(line)
                tables.append((t.get('tableId', t.get('tableID', '')), t))
    
    print(f"Loaded {len(tables)} tables", flush=True)
    
    # Process tables
    total_queries = 0
    start_time = time.time()
    
    with open(train_data_out_path, 'w') as output_f:
        for idx, (table_id, table_data) in enumerate(tqdm(tables, desc="Generating queries")):
            
            # Skip if table not in semantic ID map
            if table_id not in semantic_id_map:
                print(f"Warning: Table {table_id} not in semantic ID map, skipping", flush=True)
                continue
            
            # Get IDs
            origin_id = origin_id_map.get(table_id, idx)
            semantic_id = semantic_id_map[table_id]
            
            # Convert table to markdown
            table_prompt = json_to_markdown(table_data)
            document_title = table_data.get('documentTitle', '') or table_data.get('tableId', '')
            
            # Get table info with potential truncation (TSA setup)
            token_count = estimate_token_count(table_prompt)
            if token_count > args.table_max_token:
                header_prompt, rows, row_num = get_limit_prompt(
                    table_prompt, token_count, max_tokens=args.table_max_token
                )
            else:
                split_prompt = table_prompt.split("\n")
                header_prompt = split_prompt[:2]
                rows = split_prompt[2:]
                row_num = len(rows)
            
            table_prompt_info = (header_prompt, rows, row_num)
            
            table_start = time.time()
            
            # Generate queries using TSA (faithful to official implementation)
            try:
                queries = generator.call_gpt_multiple(
                    instruction=instruction,
                    document_title=document_title,
                    table_prompt_info=table_prompt_info,
                    n=args.num,
                    temperature=args.temperature
                )
            except Exception as e:
                print(f"Error generating queries for {table_id}: {e}")
                queries = []
            
            table_time = time.time() - table_start
            
            # Write results in BIRDIE format (faithful to official)
            for query in queries:
                record = {
                    "text_id": semantic_id,
                    "question": query,
                    "tableId": table_id,
                    "origin_id": origin_id
                }
                output_f.write(json.dumps(record) + '\n')
            output_f.flush()
            
            total_queries += len(queries)
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(tables) - idx - 1)
                print(f"[{idx+1}/{len(tables)}] {table_id}: {len(queries)} queries in {table_time:.1f}s "
                      f"(total: {total_queries}, ETA: {remaining/60:.1f}min)", flush=True)
    
    total_time = time.time() - start_time
    print(f"\nDone! Generated {total_queries} queries in {total_time:.1f}s")
    print(f"Output saved to: {train_data_out_path}")


if __name__ == "__main__":
    main()
