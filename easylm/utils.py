from typing import List, NamedTuple, Union
import numpy as np
import torch
import torch.nn.functional as F


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