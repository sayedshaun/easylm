import torch
from typing import NamedTuple, Union


class CausalModelOutput(NamedTuple):
    loss : Union[torch.Tensor, None] = None
    logits : Union[torch.Tensor, None] = None
    last_hidden_state: Union[torch.Tensor, None] = None


class MaskedModelOutput(NamedTuple):
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None
    hidden_states: Union[torch.Tensor, None] = None
    pooler_output: Union[torch.Tensor, None] = None