import torch
from typing import Union
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from langtrain.config.config import GPTConfig
from langtrain.nn.nn import TransformerDecoderBlock, PositionalEmbeddings
from langtrain.model.modeling_utils import CausalModelOutput, CausalGenerationMixin, LoadFromPretrainedMixin


class GPTModel(torch.nn.Module, CausalGenerationMixin, LoadFromPretrainedMixin):
    """
    The GPT model class for causal language modeling for predicting the next token in a sequence.
    """
    config_class = GPTConfig
    
    def __init__(self, config: GPTConfig) -> None: 
        super(GPTModel, self).__init__()
        self.config = config
        self.embedding = PositionalEmbeddings(
            config.vocab_size,config.hidden_size,config.max_seq_len,config.dropout)
        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(config.hidden_size,config.num_heads,config.norm_epsilon,config.dropout) 
            for _ in range(config.num_layers)])
        self.norm = LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

    def forward(self, input_ids: torch.Tensor, target_ids: Union[torch.Tensor, None] = None, 
                causal_mask: bool = True)-> CausalModelOutput:
        
        mask = self._make_causal_mask(input_ids) if causal_mask else None
        input_ids = self.embedding(input_ids)
        for block in self.blocks:
            input_ids = block(input_ids, mask)
        input_ids = self.dropout(self.norm(input_ids))
        logits = self.linear(input_ids)
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            return CausalModelOutput(loss=loss, logits=logits)
        else:
            return CausalModelOutput(logits=logits)
    
    def _make_causal_mask(self, input_ids: torch.Tensor)-> torch.Tensor:
        N, S = input_ids.shape # batch, seq_len
        mask = torch.ones(S, S, dtype=torch.bool, device=input_ids.device).tril(diagonal=0)
        return mask