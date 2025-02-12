import torch
from easylm.nn import (
    PositionalEmbeddings, 
    TransformerEncoderBlock, 
    LayerNormalization, 
    Dropout, 
    Linear
)

class BertModel(torch.nn.Module):
    def __init__(self, config: object) -> None:
        super(BertModel, self).__init__()
        self.embedding = PositionalEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        # Stacking transformer encoder blocks
        self.blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads, 
                norm_epsilon=config.norm_epsilon, 
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        self.norm = LayerNormalization(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(config.dropout)
        # Linear head to produce logits over vocabulary for each token.
        self.linear = Linear(config.hidden_size, config.vocab_size)

    def _make_attention_mask(self, X: torch.Tensor) -> torch.Tensor:
        mask = (X != 0).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mask = self._make_attention_mask(X)  # (B, L)
        X = self.embedding(X)  # (B, L, hidden_size)
        for block in self.blocks:
            X = block(X, mask) 
        
        X = self.norm(X)
        X = self.dropout(X)
        logits = self.linear(X)
        return logits


