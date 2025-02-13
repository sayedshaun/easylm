from typing import Union
import torch
from torch import nn
from easylm.config import GPTConfig
from easylm.data import Tokenizer
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
            [TransformerDecoderBlock(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads,
                norm_epsilon=config.norm_epsilon,
                dropout=config.dropout, 
            ) for _ in range(config.num_layers)
            ]
        )
        self.linear = Linear(config.hidden_size, config.vocab_size)

        
    def forward(self, X: torch.Tensor,  return_last_state: bool = False)-> torch.Tensor:
        mask = self._make_causal_mask(X)
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X, mask)
        logits = self.linear(X)
        if return_last_state:
            return X
        return logits
    
    def _make_causal_mask(self, X: torch.Tensor)-> torch.Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1]))
        return mask


    @torch.no_grad()
    def generate(self, start:str, max_length:int=50, temperature:float=0.5, device:str="cpu")->str:
        outputs = start
        for _ in range(max_length):
            long = torch.LongTensor(outputs).unsqueeze(0).to(device)
            logits = self(long)[:, -1, :]/temperature
            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs, num_samples=1)
            top_p = index[0, -1].item()
            outputs.append(top_p)
        return "".join(Tokenizer.decode(outputs))

