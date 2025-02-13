import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easylm.config import LlamaConfig
from easylm.nn import Dropout, Embeddings, Linear, LlamaBlock


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super(LlamaModel, self).__init__()
        self.config = config
        self.embedding = Embeddings(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
            LlamaBlock(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads, 
                dropout=config.dropout, 
                norm_epsilon=config.norm_epsilon
            )
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
        position_ids = torch.arange(X.shape[1])
        return position_ids
    
    def _make_triangle_mask(self, X: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1]))
        return mask
    