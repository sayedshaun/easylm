import torch
from torch import nn
from typing import Union
import torch.nn.functional as F
from langtrain.config import BertConfig
from langtrain.nn.nn import PositionalEmbeddings, TransformerEncoderBlock
from langtrain.model.modeling_utils import MaskedModelOutput, MaskGenerationMixin, LoadFromPretrainedMixin


class BertModel(torch.nn.Module, MaskGenerationMixin, LoadFromPretrainedMixin):
    """
    The BERT model class for masked language modeling for predicting masked tokens in a sequence.
    """
    config_class = BertConfig

    def __init__(self, config: BertConfig) -> None:
        super(BertModel, self).__init__()
        self.config = config
        self.embedding = PositionalEmbeddings(
            config.vocab_size,config.hidden_size,config.max_seq_len,config.dropout)
        self.blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(config.hidden_size, config.num_heads, config.norm_epsilon, config.dropout) 
            for _ in range(config.num_layers)])
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Create mask from input tokens (assumes token id 0 is padding)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
        return mask.to(input_ids.device)

    def forward(self, input_ids: torch.Tensor, target_ids: Union[torch.Tensor, None] = None, ) -> MaskedModelOutput:
        mask = self._make_attention_mask(input_ids)  # (B, 1, 1, L)
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