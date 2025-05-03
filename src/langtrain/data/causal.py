import os
import torch
import numpy as np
from typing import Generator, Tuple, List
from platformdirs import user_cache_dir
from langtrain.utils import seed_everything
from langtrain.tokenizer.base import Tokenizer
from langtrain.data.base import TokenDumpper, DocumentLoader, IterableDocumentLoader


seed_everything(42)


class SimpleCausalDataset(torch.utils.data.Dataset, DocumentLoader):
    """
    A simple dataset for causal language modeling.
    This dataset is designed for small to medium-sized text files.
    """
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, n_ctx: int) -> None:
        self.n_ctx = n_ctx
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data)

    def __len__(self) -> int:
        return len(self.data) - self.n_ctx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load only the required part into memory
        block = self.data[idx: idx + self.n_ctx + 1]
        input_ids, target_ids = block[:-1], block[1:]
        return (
            torch.tensor(input_ids, dtype=torch.long), 
            torch.tensor(target_ids, dtype=torch.long)
            )
    

class IterableCausalDataset(torch.utils.data.IterableDataset, IterableDocumentLoader):
    """
    An IterableDataset for causal language modeling.
    This dataset is designed to handle large text files efficiently.
    """
    def __init__(self, dir_or_path: str, tokenizer, n_ctx: int, stride: int = 1) -> None:
        self.dir_or_path = dir_or_path
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.stride = stride

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        files, text_generator = self.load(self.dir_or_path)
        # If using multiple workers, split the files among them
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            all_files = files
            # Subsample files based on worker ID and total number of workers
            all_files = all_files[worker_info.id::worker_info.num_workers]
            text_generator = (open(file, "r", encoding="utf-8").read() for file in all_files)

        for text in text_generator:
            token_ids = self.tokenizer.encode(text)
            n_tokens = len(token_ids)
            # Use a sliding window to generate examples
            for i in range(0, n_tokens - self.n_ctx, self.stride):
                block = token_ids[i: i + self.n_ctx + 1]
                input_ids, target_ids = block[:-1], block[1:]
                yield torch.tensor(input_ids), torch.tensor(target_ids)



class ExperimentalIterableCausalDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset for causal language modeling.
    This dataset is still in the experimental phase.
    """
    def __init__(self, directory: str, tokenizer, n_ctx: int, batch: int) -> None:
        self.directory = directory
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.batch = batch

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        # Iterate over grouped windows of sentences
        for sentences in IterableCausalDataset.batch_stream(self.directory, self.batch):
            text = " ".join(sentences)
            ids = self.tokenizer.encode(text)

            # Slide over token IDs in non-overlapping blocks
            for i in range(0, len(ids) - 1, self.n_ctx):
                block = ids[i : i + self.n_ctx + 1]
                if len(block) < 2:
                    continue
                input_ids = block[:-1]
                target_ids = block[1:]
                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )

    @staticmethod
    def sentence_stream(directory: str) -> Generator[str, None, None]:
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith('.txt'):
                continue
            path = os.path.join(directory, fn)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s:
                        yield s

    @staticmethod
    def batch_stream(directory: str, num_sentences: int) -> Generator[List[str], None, None]:
        sentences: List[str] = []
        for sentence in IterableCausalDataset.sentence_stream(directory):
            sentences.append(sentence)
            if len(sentences) == num_sentences:
                yield sentences
                sentences = []
        if sentences:
            yield sentences



class LazyCausalDataset(torch.utils.data.Dataset):
    """
    A lazy dataset for causal language modeling.
    This dataset uses memory mapping to handle very large files efficiently.
    """
    def __init__(self, file_path: str, tokenizer: Tokenizer, n_ctx: int, token_caching: bool = True, cache_dir: str = None, **kwargs) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.token_caching = token_caching
        self.cache_dir = cache_dir
        if cache_dir is None: cache_dir = user_cache_dir("langtrain")
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, "TOKENS.bin")
        if not os.path.exists(cache_path) or not token_caching:
            if os.path.exists(cache_path) and not token_caching: 
                os.remove(cache_path)
            dumpper = TokenDumpper(file_path, tokenizer, cache_path, **kwargs)
            dumpper.start()
        else:
            print(f"Using cached tokens from {cache_path}")

        self.tokens = np.memmap(cache_path, dtype=np.uint16, mode='r')

    def __len__(self) -> int:
        return len(self.tokens) - self.n_ctx
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.tokens[idx: idx + self.n_ctx + 1]
        input_ids = torch.from_numpy(block[:-1].copy()).long()
        target_ids = torch.from_numpy(block[1:].copy()).long()
        return input_ids, target_ids