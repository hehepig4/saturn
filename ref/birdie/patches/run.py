import argparse
import warnings
from data import IndexingTrainDataset, IndexingCollator, MultiAnswerQueryDataset, MultiAnswerQueryEvalCollator
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from inspect import signature
from transformers import (
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    IntervalStrategy,
    set_seed,
)
from trainer import DSITrainer
import numpy as np
import torch
import wandb
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
wandb.init(mode='offline')


@dataclass
class RunArguments:
    model_name: str = field(default="google/mt5-base")
    base_model_path: str = field(default="./model/mt5-base")
    max_length: Optional[int] = field(default=64)
    id_max_length: Optional[int] = field(default=20)
    train_file: str = field(default="./dataset/fetaqa/train.json")
    valid_file: str = field(default="./dataset/fetaqa/test.json")
    task: str = field(default="Index",  metadata={"help": "Index, Search"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    remove_prompt: Optional[bool] = field(default=True)
    peft: Optional[bool] = field(default=False)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=0.0005)
    warmup_steps: float = field(default=10000)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=32)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    max_steps: int = field(default=800)
    save_strategy: str = field(default="steps")
    dataloader_num_workers: int = field(default=6)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    gradient_accumulation_steps : int = field(default=6)
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=100)
    metric_for_best_model: str = field(default="Hits@5")
    dataloader_drop_last: bool = field(default=False)
    greater_is_better: bool = field(default=True)
    run_name: str = field(default="feta")
    output_dir: str = field(default="./model/feta")


set_seed(313)
class PrintGradientsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        print(f"Epoch {state.epoch} ended.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter Name: {name}")
                print("Parameter Value:")
                print(param.data)
                if param.grad is not None:
                    print("Parameter Gradient:")
                    print(param.grad)
                else:
                    print("Parameter Gradient: None")


def make_compute_metrics(tokenizer, valid_ids, all_answer_tables_map=None):
    """Create compute_metrics function for evaluation.
    
    Args:
        tokenizer: MT5 tokenizer
        valid_ids: Set of valid table IDs
        all_answer_tables_map: Optional dict mapping query_index -> list of all correct table IDs
                              for multi-answer evaluation. If None, uses single label_id.
    """
    def compute_metrics(eval_preds):
        # Extended k values: 1, 3, 5, 10, 20, 100
        k_values = [1, 3, 5, 10, 20, 100]
        hits = {k: 0 for k in k_values}
        
        i = 0
        gt2err = dict()
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)

            label_id = tokenizer.decode(label, skip_special_tokens=True)
            
            # Get all valid answer tables for this query
            if all_answer_tables_map is not None and i in all_answer_tables_map:
                answer_tables = set(all_answer_tables_map[i])
            else:
                answer_tables = {label_id}
            
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)
            
            # Multi-answer evaluation: check if ANY prediction matches ANY answer table
            def check_hit_at_k(rank_list, answer_tables, k):
                """Check if any of the top-k predictions matches any answer table."""
                top_k = rank_list[:k]
                return len(set(top_k) & answer_tables) > 0
            
            # Check hits for all k values
            for k in k_values:
                if check_hit_at_k(filtered_rank_list, answer_tables, k):
                    hits[k] += 1
            i += 1
        
        total = len(eval_preds.predictions)
        return {
            f"Hits@{k}": hits[k] / total for k in k_values
        }
    return compute_metrics

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = HfArgumentParser((CustomTrainingArguments, RunArguments))
    args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI2Table", name=args.run_name, entity="DSITable")

    tokenizer = MT5Tokenizer.from_pretrained(run_args.base_model_path, cache_dir='cache')
    fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.base_model_path, cache_dir='cache')

    # legal tokens
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    if run_args.task == "Index":
        training_args = args
        model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')

        if run_args.peft:
            print("**************PEFT*****************")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                target_modules='(decoder\.block\.\d+\.layer\.2|encoder\.block\.\d+\.layer\.1)\.DenseReluDense\.(wi_0|wi_1|wo)',
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            print("**************FULL*****************")

        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics = make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab = restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
        )
        trainer.train()


    elif run_args.task == 'Search':
        test_args = args
        test_args.do_train = False
        test_args.do_predict = True
        if run_args.peft == True:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')
            lora_model_path = run_args.lora_model_path
            model = PeftModel.from_pretrained(model, lora_model_path)
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')
        
        # Use MultiAnswerQueryDataset for test data to support multi-answer evaluation
        test_dataset = MultiAnswerQueryDataset(path_to_data=run_args.valid_file,
                                               max_length=run_args.max_length,
                                               cache_dir='cache',
                                               remove_prompt=run_args.remove_prompt,
                                               tokenizer=tokenizer)

        all_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           remove_prompt=run_args.remove_prompt,
                                           tokenizer=tokenizer)
        
        # Build all_answer_tables_map from test_dataset
        all_answer_tables_map = {}
        print("Building multi-answer evaluation map...")
        for i in range(len(test_dataset)):
            _, _, all_answers = test_dataset[i]
            all_answer_tables_map[i] = all_answers
        
        multi_answer_count = sum(1 for answers in all_answer_tables_map.values() if len(answers) > 1)
        print(f"Multi-answer queries: {multi_answer_count}/{len(test_dataset)}")
        
        # init trainer with multi-answer support
        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=test_args,
            compute_metrics=make_compute_metrics(fast_tokenizer, all_dataset.valid_ids, all_answer_tables_map),
            data_collator=MultiAnswerQueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length)

        results = trainer.evaluate(test_dataset)
        
        # Print results
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        for k, v in sorted(results.items()):
            if 'Hits@' in k:
                print(f"{k}: {v:.4f}")
        print("="*50)

    else:

        raise NotImplementedError("--task should be in 'DSI' or 'inference'")


if __name__ == "__main__":
    main()
