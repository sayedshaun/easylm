import os
import random
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import List, NamedTuple, Tuple, Union


def trainable_parameters(model: torch.nn.Module)->str:
    trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    return f"Total Trainable Params: {trainable_params/1e6:.2f} M"


class CausalModelOutput(NamedTuple):
    loss : Union[torch.Tensor, None] = None
    logits : Union[torch.Tensor, None] = None
    last_hidden_state: Union[torch.Tensor, None] = None


class MaskedModelOutput(NamedTuple):
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None
    hidden_states: Union[torch.Tensor, None] = None
    pooler_output: Union[torch.Tensor, None] = None


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)  # Unpack the batch into inputs and targets
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs, targets


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False