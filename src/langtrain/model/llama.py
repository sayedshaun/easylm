import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from langtrain.nn.nn import LlamaBlock
from langtrain.config.config import LlamaConfig
from torch.nn import Dropout, Embedding, Linear, RMSNorm
from langtrain.model.modeling_utils import CausalModelOutput, CausalGenerationMixin, LoadFromPretrainedMixin


class LlamaModel(nn.Module, CausalGenerationMixin, LoadFromPretrainedMixin):
    """
    The Llama model class for causal language modeling for predicting the next token in a sequence.
    """
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig) -> None:
        super(LlamaModel, self).__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [LlamaBlock(config.hidden_size, config.num_heads,config.dropout, config.norm_epsilon)
            for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)
        self.apply(self._init_weights)


    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, input_ids: torch.Tensor, target_ids: Union[torch.Tensor, None] = None, 
            causal_mask: bool = True) -> CausalModelOutput:
        
        mask = self._make_triangle_mask(input_ids) if causal_mask else None 
        position_ids = self._make_position_ids(input_ids)
        input_ids = self.embedding(input_ids)
        for block in self.blocks:
            input_ids = block(input_ids, position_ids, mask)
        input_ids = self.dropout(input_ids)
        input_ids = self.norm(input_ids)
        logits = self.linear(input_ids)
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            return CausalModelOutput(loss=loss, logits=logits)
        else:
            return CausalModelOutput(logits=logits)
    

    def _make_position_ids(self, X: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(X.shape[1], device=X.device)
        return position_ids

    def _make_triangle_mask(self, X: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1], device=X.device))
        return mask
