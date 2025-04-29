from typing import Tuple, Union
import torch
import torch.nn.functional as F
import yaml
from src.config import BertConfig
from src.nn import (
    PositionalEmbeddings, 
    TransformerEncoderBlock, 
    LayerNorm, 
    Dropout, 
    Linear
)
from src.utils import MaskedModelOutput


class BertModel(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertModel, self).__init__()
        self.config = config
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
        self.norm = LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Create mask from input tokens (assumes token id 0 is padding)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
        return mask.to(input_ids.device)

    def forward(
            self, 
            input_ids: torch.Tensor, 
            target_ids: Union[torch.Tensor, None] = None, 
            ) -> MaskedModelOutput:
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
        mask = self._make_attention_mask(input_ids)  # (B, 1, 1, L)

        # Compute token embeddings.
        input_ids = self.embedding(input_ids)  # (B, L, hidden_size)
        batch_size = input_ids.size(0)

        # Expand the learnable [CLS] token to the batch dimension.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_size)

        # Prepend the CLS token to the input embeddings.
        input_ids = torch.cat((cls_tokens, input_ids), dim=1)  # (B, L+1, hidden_size)

        # Update the attention mask to include the CLS token (assumed valid, so mask value 1).
        cls_mask = torch.ones(batch_size, 1, 1, 1, device=input_ids.device, dtype=mask.dtype)
        mask = torch.cat((cls_mask, mask), dim=-1)  # (B, 1, 1, L+1)

        # Pass the entire sequence (including the CLS token) through the transformer blocks.
        for block in self.blocks:
            input_ids = block(input_ids, mask)

        # Apply normalization and dropout.
        input_ids = self.norm(input_ids)
        input_ids = self.dropout(input_ids)
        logits = self.linear(input_ids)  # (B, L+1, vocab_size)
        pooled_output = torch.tanh(self.pooler(input_ids[:, 0, :]))  # (B, hidden_size)

        if target_ids is not None:
            # Remove the CLS token logits for computing the loss.
            reshaped_logits = logits[:, 1:].reshape(-1, logits.shape[-1])  # (B * L, vocab_size)
            loss = F.cross_entropy(reshaped_logits, target_ids.view(-1))
            return MaskedModelOutput(loss=loss, logits=logits, pooler_output=pooled_output)
        else:
            return MaskedModelOutput(logits=logits, pooler_output=pooled_output)



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
        outputs = self.forward(input_ids)  # shape: (B, L+1, vocab_size)
        logits = outputs.logits
        # Remove the logits for the [CLS] token (first token)
        return logits[:, 1:, :]
    

    @staticmethod
    def from_pretrained(preprained_path: str, device: Union[torch.device, str] = "cpu"):
        with open(f"{preprained_path}/model_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = BertConfig(**config_dict)
        model = BertModel(config)
        model.load_state_dict(
            torch.load(
                f"{preprained_path}/pytorch_model.pt", 
                weights_only=True, 
                map_location=device), 
            strict=True
        )
        return model