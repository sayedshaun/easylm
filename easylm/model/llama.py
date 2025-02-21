from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easylm.config import LlamaConfig
from easylm.utils import CausalModelOutput
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
    

    def forward(
            self, 
            input_ids: torch.Tensor, 
            target_ids: Union[torch.Tensor, None] = None, 
            causal_mask: bool = True
            ) -> CausalModelOutput:
        mask = self._make_triangle_mask(input_ids) if causal_mask else None 
        position_ids = self._make_position_ids(input_ids)
        input_ids = self.embedding(input_ids)
        for block in self.blocks:
            input_ids = block(input_ids, position_ids, mask)
        input_ids = self.dropout(input_ids)
        logits = self.linear(input_ids)
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            return CausalModelOutput(loss=loss, logits=logits)
        else:
            return CausalModelOutput(logits=logits)
    

    def _make_position_ids(self, X: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(X.shape[1], device=X.device)
        return position_ids
    

    def _make_triangle_mask(self, X: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1], device=X.device))
        return mask
    

    def generate(self, input_ids: torch.Tensor, eos_token_id: int, max_seq_len: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(max_seq_len):
                outputs = self(generated, causal_mask=False) # (batch_size, seq_len, vocab_size)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :] # last token's logits
                next_token = next_token_logits.argmax(dim=-1, keepdim=True) # Greedy decoding
                generated = torch.cat([generated, next_token], dim=1) # Append new token to sequence
                if next_token == eos_token_id:
                    break
            return generated


    @staticmethod
    def from_pretrained(preprained_path: str):
        with open(f"{preprained_path}/model_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = LlamaConfig(**config_dict)
        model = LlamaModel(config)
        model.load_state_dict(
            torch.load(f"{preprained_path}/pytorch_model.pt", weights_only=True), strict=True)
        return model
