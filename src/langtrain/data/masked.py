import random
import torch
from typing import Tuple
from langtrain.data.base import DocumentLoader
from langtrain.utils import seed_everything
from langtrain.tokenizer.sentencepiece import Tokenizer


seed_everything(42)


class MaskedDataset(torch.utils.data.Dataset, DocumentLoader):
    """
    A dataset for masked language modeling.
    """
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, n_ctx: int, mask_prob: float = 0.15) -> None:
        self.mask_prob = mask_prob
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data).tolist()

    def __len__(self):
        # We subtract 2 because we will add [CLS] and [SEP] tokens.
        return len(self.data) - (self.n_ctx - 2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx: idx + self.n_ctx - 2]
        input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
        target_ids = [-100] * len(input_ids)

        for i in range(1, len(input_ids) - 1):
            if random.random() < self.mask_prob:
                target_ids[i] = input_ids[i]
                rand = random.random()
                if rand < 0.8:
                    # 80% of the time, replace with [MASK]
                    input_ids[i] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    # 10% of the time, replace with a random token from the vocabulary.
                    input_ids[i] = random.choice(range(self.tokenizer.vocab_size))
                else:
                    # 10% of the time, leave the token unchanged.
                    pass

        return torch.tensor(input_ids), torch.tensor(target_ids)