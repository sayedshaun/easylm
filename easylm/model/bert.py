from typing import Tuple, Union
import torch
from easylm.config import BERTConfig
from easylm.nn import (
    PositionalEmbeddings, 
    TransformerEncoderBlock, 
    LayerNormalization, 
    Dropout, 
    Linear
)


class BertModel(torch.nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super(BertModel, self).__init__()
        # Embedding layer (includes token and positional embeddings)
        self.embedding = PositionalEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        # Stacking transformer encoder blocks
        self.blocks = torch.nn.ModuleList(
            [
            TransformerEncoderBlock(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads, 
                norm_epsilon=config.norm_epsilon, 
                dropout=config.dropout
            ) for _ in range(config.num_layers)
            ]
        )
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.norm = LayerNormalization(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)

    def _make_attention_mask(self, X: torch.Tensor) -> torch.Tensor:
        # Create mask from input tokens (assumes token id 0 is padding)
        mask = (X != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
        return mask

    def forward(self, X: torch.Tensor, return_last_state: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            X (torch.Tensor): Input token IDs of shape (B, L). 
                              Note: For compatibility with BERT, X is expected NOT to include a [CLS] token.
            return_last_state (bool): Whether to return the last state of the transformer.
        Returns:
            logits (torch.Tensor): Logits over vocabulary for each position, shape (B, L+1, vocab_size)
            pooled_output (torch.Tensor): Pooled output from the [CLS] token, shape (B, hidden_size)
        """
        # Create attention mask for the input tokens.
        mask = self._make_attention_mask(X)  # (B, 1, 1, L)

        # Compute token embeddings.
        X = self.embedding(X)  # (B, L, hidden_size)
        batch_size = X.size(0)

        # Expand the learnable [CLS] token to the batch dimension.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_size)

        # Prepend the CLS token to the input embeddings.
        X = torch.cat((cls_tokens, X), dim=1)  # (B, L+1, hidden_size)

        # Update the attention mask to include the CLS token (assumed valid, so mask value 1).
        cls_mask = torch.ones(batch_size, 1, 1, 1, device=X.device, dtype=mask.dtype)
        mask = torch.cat((cls_mask, mask), dim=-1)  # (B, 1, 1, L+1)

        # Pass the entire sequence (including the CLS token) through the transformer blocks.
        for block in self.blocks:
            X = block(X, mask)

        # Apply normalization and dropout.
        X = self.norm(X)
        X = self.dropout(X)

        # Compute logits over vocabulary for each token position.
        logits = self.linear(X)  # (B, L+1, vocab_size)

        # Pooler: take the hidden state corresponding to the first token ([CLS]),
        # pass it through a linear layer and tanh activation.
        pooled_output = torch.tanh(self.pooler(X[:, 0, :]))  # (B, hidden_size)
        if return_last_state:
            return logits, pooled_output
        return logits


    @torch.no_grad()
    def fill_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, L).

        Returns:
            logits (torch.Tensor): Logits over vocabulary for each position,
                                shape (B, L, vocab_size)
        """
        # Run the forward pass which prepends the CLS token
        self.eval()
        logits = self.forward(input_ids)  # shape: (B, L+1, vocab_size)
        
        # Remove the logits for the [CLS] token (first token)
        return logits[:, 1:, :]