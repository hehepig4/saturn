#!/usr/bin/env python3
"""
Query Generator LoRA Training Script for BIRDIE

This script trains a LoRA adapter on TLlama3 (Llama-3.1-8B-table-base) to generate
synthetic queries for a specific dataset.

Paper Parameters (Section 4.2):
- Base model: TLlama3 (Llama-3.1-8b fine-tuned on 7 table tasks)
- LoRA: rank=8, target=attention layers
- Training data: Dataset training split with human-annotated queries

Usage:
    python train_query_generator.py \
        --model_path /path/to/Llama-3-8B-table-base \
        --train_data /path/to/train_data.json \
        --output_dir /path/to/output \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Instruction template - Using JSON output format for consistency with inference
# Format: Instruction + Input + Output (JSON)
INSTRUCTION_TEMPLATE = """Please generate the probable question a user would ask based on this table, ensuring the answer can be found in some cells in this table or via an aggregation operator such as (Max,Min,Avg,Count). The generated question should contain the explicit information of this table (such as the information of the caption) so that given the question and a repository of tables, this table can be successfully found.

Return the final result in JSON format as {{"question":""}}.

 tableCaption: {caption}
table: {table_markdown}"""

# Expected output format for training (appended after INSTRUCTION_TEMPLATE)
# The model should learn to generate: '{"question": "..."}'


def json_to_markdown(table_data: Dict) -> str:
    """Convert table JSON to markdown format."""
    columns = [col.get('text', '') for col in table_data.get('columns', [])]
    rows = table_data.get('rows', [])
    
    # Header
    header = '|' + '|'.join(col if col else ' ' for col in columns) + '|'
    separator = '|' + '|'.join(['---'] * len(columns)) + '|'
    
    # Rows
    row_lines = []
    for row in rows:
        cells = [cell.get('text', '') for cell in row.get('cells', [])]
        row_lines.append('|' + '|'.join(cells) + '|')
    
    return '\n'.join([header, separator] + row_lines)


def truncate_table_markdown(table_markdown: str, max_rows: int = 20) -> str:
    """Truncate table to max_rows to fit context window."""
    lines = table_markdown.split('\n')
    if len(lines) <= max_rows + 2:  # header + separator + rows
        return table_markdown
    
    header = lines[:2]
    rows = lines[2:]
    
    # Sample rows if too many
    if len(rows) > max_rows:
        sampled_rows = random.sample(rows, max_rows)
    else:
        sampled_rows = rows
    
    return '\n'.join(header + sampled_rows)


class QueryGeneratorDataset(Dataset):
    """Dataset for training query generator.
    
    Each sample contains:
    - Input: Table in markdown format with instruction
    - Output: Generated question in JSON format
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 256,
        table_data_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load training data
        self.samples = []
        self.tables = {}
        
        # Load table data if provided
        if table_data_path:
            with open(table_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    table = json.loads(line.strip())
                    self.tables[table['tableId']] = table
        
        # Load question-table pairs
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get table data
        table_id = sample.get('tableId') or sample.get('tableID')
        question = sample.get('text') or sample.get('question')
        
        if table_id in self.tables:
            table = self.tables[table_id]
            caption = table.get('documentTitle', table.get('title', 'Unknown'))
            table_markdown = json_to_markdown(table)
        else:
            # Use embedded table data if available
            if 'table' in sample:
                caption = sample.get('caption', sample.get('title', 'Unknown'))
                if isinstance(sample['table'], dict):
                    table_markdown = json_to_markdown(sample['table'])
                else:
                    table_markdown = sample['table']
            else:
                caption = sample.get('caption', sample.get('title', 'Unknown'))
                table_markdown = sample.get('table_markdown', '')
        
        # Truncate table
        table_markdown = truncate_table_markdown(table_markdown, max_rows=20)
        
        # Build input (matches paper Figure 5 format)
        input_text = INSTRUCTION_TEMPLATE.format(
            caption=caption,
            table_markdown=table_markdown
        )
        
        # Build output (JSON format for consistency with inference)
        # Escape quotes in question for valid JSON
        escaped_question = question.replace('"', '\\"')
        output_text = '{"question": "' + escaped_question + '"}'
        
        # Tokenize
        input_ids = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )['input_ids']
        
        output_ids = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )['input_ids']
        
        # For causal LM, we concatenate input and output
        # Labels: -100 for input tokens (ignored in loss), actual token ids for output
        full_input_ids = input_ids + output_ids
        labels = [-100] * len(input_ids) + output_ids
        
        return {
            'input_ids': full_input_ids,
            'attention_mask': [1] * len(full_input_ids),
            'labels': labels,
        }


@dataclass
class DataCollatorForCausalLM:
    """Data collator for causal language modeling with variable length."""
    
    tokenizer: Any
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length
        max_len = max(len(f['input_ids']) for f in features)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of:
            max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }
        
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            
            # Pad input_ids
            batch['input_ids'].append(
                f['input_ids'] + [self.tokenizer.pad_token_id] * pad_len
            )
            
            # Pad attention_mask
            batch['attention_mask'].append(
                f['attention_mask'] + [0] * pad_len
            )
            
            # Pad labels with -100 (ignored in loss)
            batch['labels'].append(
                f['labels'] + [-100] * pad_len
            )
        
        return {k: torch.tensor(v) for k, v in batch.items()}


def create_lora_config(
    rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create LoRA config matching paper parameters.
    
    Paper: "adding LoRA to the attention layers with a rank of 8"
    """
    if target_modules is None:
        # Target attention layers for Llama models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    parser = argparse.ArgumentParser(description='Train Query Generator LoRA')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to TLlama3 base model')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (JSONL with question-table pairs)')
    parser.add_argument('--table_data', type=str, default=None,
                        help='Path to table_data.json (optional, for looking up tables)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for LoRA adapter')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Warmup ratio')
    parser.add_argument('--max_input_length', type=int, default=1024,
                        help='Maximum input sequence length')
    parser.add_argument('--max_output_length', type=int, default=256,
                        help='Maximum output sequence length')
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank (paper: 8)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 training')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 training')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Query Generator LoRA Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Training data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Create LoRA config
    print("Applying LoRA...")
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\nLoading training data...")
    train_dataset = QueryGeneratorDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        table_data_path=args.table_data,
    )
    print(f"Training samples: {len(train_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=args.max_input_length + args.max_output_length,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
