from typing import Union
import torch
import torch.nn.functional as F
from easylm.config import GPTConfig
from easylm.nn import Linear, TransformerDecoderBlock, PositionalEmbeddings
from easylm.utils import CausalModelOutput


class GPTModel(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None: 
        super(GPTModel, self).__init__()
        self.config = config
        self.embedding = PositionalEmbeddings(
            config.vocab_size, 
            config.hidden_size, 
            config.max_seq_len, 
            config.dropout
        )
        self.blocks = torch.nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_size, 
                    config.num_heads,
                    config.norm_epsilon,
                    config.dropout, 
                    ) 
                for _ in range(config.num_layers)
            ]
        )
        self.linear = Linear(config.hidden_size, config.vocab_size)

    def forward(
            self, 
            input_ids: torch.Tensor, 
            target_ids: Union[torch.Tensor, None] = None, 
            causal_mask: bool = True
            )-> CausalModelOutput:
        mask = self._make_causal_mask(input_ids) if causal_mask else None
        input_ids = self.embedding(input_ids)
        for block in self.blocks:
            input_ids = block(input_ids, mask)
        logits = self.linear(input_ids)
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            return CausalModelOutput(loss=loss, logits=logits)
        else:
            return CausalModelOutput(logits=logits)
    
    def _make_causal_mask(self, X: torch.Tensor)-> torch.Tensor:
        N, S = X.shape # batch, seq_len
        mask = torch.ones(S, S, dtype=torch.bool).tril(diagonal=0)
        return mask

    def generate(self, input_ids: torch.Tensor, eos_token_id: int, max_length: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(max_length):
                outputs = self(generated, causal_mask=False) # (batch_size, seq_len, vocab_size)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :] # last token's logits
                next_token = next_token_logits.argmax(dim=-1, keepdim=True) # Greedy decoding
                generated = torch.cat([generated, next_token], dim=1) # Append new token to sequence
                if next_token == eos_token_id:
                    break
            return generated

