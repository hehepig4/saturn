from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For backward compatibility, export new classes
__all__ = [
    'IndexingTrainDataset', 'IndexingCLDataset', 'GenerateDataset',
    'IndexingCollator', 'IndexingCollator_Los', 'QueryEvalCollator',
    'MultiAnswerQueryDataset', 'MultiAnswerQueryEvalCollator'
]


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            trust_remote_code=True,
            cache_dir=cache_dir
        )['train']
        # print(self.train_data[0]) #{"text_id":x, "text":str}
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]

        return input_ids, str(data['text_id'])


class IndexingCLDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            trust_remote_code=True,
            cache_dir=cache_dir
        )['train']
        # print(self.train_data[0]) #{"text_id":x, "text":str}
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation=True,
                                   padding="max_length",
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


class MultiAnswerQueryDataset(Dataset):
    """Dataset for multi-answer query evaluation.
    
    Supports both BIRDIE format (text, text_id, all_answer_tables) and 
    original format (question, table_id, all_answer_tables).
    Returns (input_ids, label_id, all_answer_tables) for proper multi-answer evaluation.
    """
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            trust_remote_code=True,
            cache_dir=cache_dir
        )['train']
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        
        # Build valid_ids set (semantic IDs)
        self.valid_ids = set()
        for data in self.train_data:
            # Support both formats
            label_id = str(data.get('text_id', data.get('table_id', '')))
            self.valid_ids.add(label_id)
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        # Support both BIRDIE format (text) and original format (question)
        text = data.get('text', data.get('question', ''))
        if self.remove_prompt:
            text = text[9:] if text.startswith('Passage: ') else text
            text = text[10:] if text.startswith('Question: ') else text
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        
        # Primary label - support both formats (text_id for BIRDIE, table_id for original)
        label_id = str(data.get('text_id', data.get('table_id', '')))
        
        # All answer semantic IDs for multi-answer evaluation
        # IMPORTANT: Use all_semantic_ids (not all_answer_tables which contains tableIDs)
        all_answers = data.get('all_semantic_ids', [label_id])
        if not all_answers:
            all_answers = [label_id]
        all_answers = [str(a) for a in all_answers]
        
        return input_ids, label_id, all_answers


class GenerateDataset(Dataset):
    lang2mT5 = dict(
        ar='Arabic',
        bn='Bengali',
        fi='Finnish',
        ja='Japanese',
        ko='Korean',
        ru='Russian',
        te='Telugu'
    )

    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                if 'xorqa' in path_to_data:
                    docid, passage, title = data.split('\t')
                    for lang in self.lang2mT5.values():
                        self.data.append((docid, f'Generate a {lang} question for this passage: {title} {passage}'))
                elif 'msmarco' in path_to_data:
                    docid, passage = data.split('\t')
                    self.data.append((docid, f'{passage}'))
                else:
                    raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(docid)


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # inputs['text_ids'] = [x[1] for x in features]
        return inputs

class IndexingCollator_Los(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['text_ids'] = [x[1] for x in features]
        return inputs

@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels


@dataclass
class MultiAnswerQueryEvalCollator(DataCollatorWithPadding):
    """Collator for multi-answer query evaluation.
    
    Returns inputs dict with tokenized labels for trainer compatibility.
    """
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]  # Primary label IDs (semantic IDs)
        # all_answers = [x[2] for x in features]  # Not used directly here, handled in compute_metrics
        inputs = super().__call__(input_ids)
        
        # Tokenize labels (semantic IDs) for prediction_step compatibility
        label_tensors = self.tokenizer(
            labels, padding="longest", return_tensors="pt"
        ).input_ids
        
        # Replace padding token id's with -100
        label_tensors[label_tensors == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = label_tensors
        
        return inputs
