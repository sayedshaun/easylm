import torch
import torch.nn as nn
from easylm.config import LlamaConfig
from easylm.nn import Dropout, Embedding, Linear, LlamaBlock


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super(LlamaModel, self).__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                LlamaBlock(
                    config.hidden_size, 
                    config.num_heads,
                    config.dropout, 
                    config.norm_epsilon
                    )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)
    

    def forward(self, X: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        mask = self._make_triangle_mask(X) if causal_mask else None 
        position_ids = self._make_position_ids(X)
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X, position_ids, mask)
        X = self.dropout(X)
        logits = self.linear(X)
        return logits
    
    def _make_position_ids(self, X: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(X.shape[1], device=X.device)
        return position_ids
    
    def _make_triangle_mask(self, X: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1], device=X.device))
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


    