from typing import Union
import torch
from torch import nn
from easylm.config import GPTConfig
from easylm.tokenizer import Tokenizer
from easylm.nn import Linear, TransformerDecoderBlock, PositionalEmbeddings


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None: 
        super(GPTModel, self).__init__()
        self.embedding = PositionalEmbeddings(
            config.vocab_size, 
            config.hidden_size, 
            config.max_seq_len, 
            config.dropout
        )
        self.blocks = nn.ModuleList(
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

    def forward(self, X: torch.Tensor, causal_mask: bool = True)-> torch.Tensor:
        mask = self._make_causal_mask(X) if causal_mask else None
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X, mask)
        logits = self.linear(X)
        return logits
    
    def _make_causal_mask(self, X: torch.Tensor)-> torch.Tensor:
        N, S = X.shape # batch, seq_len
        mask = torch.ones(S, S, dtype=torch.bool).tril(diagonal=0)
        return mask

    def generate(self, input_ids: torch.Tensor, eos_token_id: int, max_length: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(max_length):
                logits = self(generated, causal_mask=False) # (batch_size, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :] # last token's logits
                next_token = next_token_logits.argmax(dim=-1, keepdim=True) # Greedy decoding
                generated = torch.cat([generated, next_token], dim=1) # Append new token to sequence
                if next_token == eos_token_id:
                    break
            return generated

