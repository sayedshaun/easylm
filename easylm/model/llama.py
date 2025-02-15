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
                LlamaBlock(config.hidden_size, config.num_heads,config.dropout, config.norm_epsilon)
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        position_ids = self._make_position_ids(X)
        mask = self._make_triangle_mask(X)
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
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            # Loop for max_length steps
            for _ in range(max_length):
                # Get model output: shape (batch_size, seq_len, vocab_size)
                logits = self(generated)
                # Focus on the last token's logits
                next_token_logits = logits[:, -1, :]
                # Greedy decoding: choose token with highest probability
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                # Append new token to sequence
                generated = torch.cat([generated, next_token], dim=1)
                # If all sequences generated an EOS token, stop early
                if (next_token == self.config.eos_token_id).all():
                    break
            return generated


    