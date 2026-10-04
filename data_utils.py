from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from collections import defaultdict
import random


MAX_LENGTH = 512


class BaseDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        el = self.samples[idx]
        input_ids = self.tokenizer(el)['input_ids']
        input_ids = input_ids[:self.max_length]
        length = len(input_ids)
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.int64), length


class StandardDataset(Dataset):
    """
    See task desciption
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        ### YOUR CODE HERE
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        ### YOUR CODE HERE
        return len(self.samples)

    def __getitem__(self, idx: int):
        ### YOUR CODE HERE
        text = self.samples[idx]

        input_ids = self.tokenizer(text)["input_ids"]
        input_ids = input_ids[:self.max_length]

        length = len(input_ids)

        return torch.tensor(input_ids, dtype=torch.int64), length


class SequencedDataset(IterableDataset):
    """
    See task desciption
    """
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        batch_size: int, 
        max_length: int = MAX_LENGTH
    ):
        ### YOUR CODE HERE
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.samples = []

        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __iter__(self):
        ### YOUR CODE HERE
        batch = []
        batch_len = 0

        for sample in self.samples:
            input_ids = self.tokenizer(sample)["input_ids"]
            input_ids = input_ids[:self.max_length]

            length = len(input_ids)

            tokens = torch.tensor(input_ids, dtype=torch.int64)

            if batch and batch_len + length > self.max_length:
                yield self.collate_data(batch)

                batch = []
                batch_len = 0

            batch.append((tokens, length))
            batch_len += length

            if len(batch) == self.batch_size:
                yield self.collate_data(batch)

                batch = []
                batch_len = 0

        if batch:
            yield self.collate_data(batch)




    def collate_data(self, batch):
        # Return a loss_mask in addition to the block-causal attention mask.
        # It must exclude transitions between adjacent packed samples.
        ### YOUR CODE HERE
        lengths = [el[1] for el in batch]

        tokens = torch.cat([el[0] for el in batch], dim=0)

        T = tokens.size(0)

        blocks = []
        for length in lengths:
            block = torch.tril(torch.ones((length, length), dtype=torch.bool))
            blocks.append(block)

        attention_mask = torch.block_diag(*blocks)


        loss_mask = torch.ones((1, T - 1), dtype=torch.bool)

        boundary = 0
        for length in lengths[:-1]:
            boundary += length
            loss_mask[0, boundary - 1] = False

        tokens = tokens.unsqueeze(0)

        return {
        "tokens": tokens,
        "lengthes": torch.tensor([T]),
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }

def base_collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_processed = torch.stack([el[0] for el in batch])
    lengthes = torch.tensor([el[1] for el in batch])
    return {
        'tokens': batch_processed, 
        'lengthes': lengthes, 
        'attention_mask': None,
        'loss_mask': None,
    }


def collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    See task desciption
    """
    ### YOUR CODE HERE
    tokens = [el[0] for el in batch]

    lengthes = torch.tensor(
        [el[1] for el in batch]
    )

    batch_processed = torch.nn.utils.rnn.pad_sequence(
        tokens,
        batch_first=True,
        padding_value=pad_token_id
    )

    return {
        "tokens": batch_processed,
        "lengthes": lengthes,
        "attention_mask": None,
        "loss_mask": None,
    }
    


class BalancedBatchSampler(Sampler):
    """
    See task desciption
    """
    def __init__(self, dataset, k: int, batch_size: int):
        ### YOUR CODE HERE
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size

        self.len_to_idx = defaultdict(list)
        max_len = 0

        for i in range(len(dataset)):
            _, length = dataset[i]

            self.len_to_idx[length].append(i)
            max_len = max(max_len, length)

        self.buckets = []

        curr_bucket = []
        min_len = None

        for l in range(max_len + 1):
            idxs = self.len_to_idx.get(l)

            if not idxs:
                continue

            if min_len is None:
                min_len = l

            if l - min_len <= k:
                curr_bucket.extend(idxs)
            else:
                self.buckets.append(curr_bucket)

                curr_bucket = list(idxs)
                min_len = l

        if curr_bucket:
            self.buckets.append(curr_bucket)

        self.num_batches = sum((len(bucket) + batch_size - 1)//batch_size  for bucket in self.buckets)

        

    def __len__(self):
        ### YOUR CODE HERE
        return self.num_batches

    def __iter__(self):
        ### YOUR CODE HERE
        order = list(range(len(self.buckets)))
        random.shuffle(order)

        for i in order:
            idxs = self.buckets[i].copy()
            random.shuffle(idxs)

            for s in range(0, len(idxs), self.batch_size):
                yield idxs[s:s + self.batch_size]
