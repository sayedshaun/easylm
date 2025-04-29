from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from src.config import LlamaConfig
from src.utils import CausalModelOutput
from src.nn import LlamaBlock
from torch.nn import Dropout, Embedding, Linear, RMSNorm


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
        self.norm = RMSNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)
    
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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
        input_ids = self.norm(input_ids)
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
    

    def generate(
            self, 
            input_ids: torch.Tensor, 
            max_new_tokens: int = 20, 
            context_size: int = 1, 
            temperature: float = 1.0, 
            top_k: Union[int, None] = None, 
            eos_id: Union[int, None] = None
        ) -> torch.Tensor:  
        for _ in range(max_new_tokens):           
            idx_cond = input_ids[:, -context_size:] 
            with torch.no_grad():    
                outputs = self.forward(idx_cond, causal_mask=False)
                logits = outputs.logits 
                logits = logits[:, -1, :] 

                if top_k is not None:                   
                    top_logits, _ = torch.topk(logits, top_k)    
                    min_val = top_logits[:, -1]    
                    logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits) 
                
                if temperature > 0.0:                     
                    logits = logits / temperature    
                    probs = torch.softmax(logits, dim=-1)    
                    idx_next = torch.multinomial(probs, num_samples=1) 
                else:       
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True) 

                if eos_id is not None and (idx_next == eos_id).all():                 
                    break  
                
                input_ids = torch.cat((input_ids, idx_next), dim=1)    

        return input_ids


    @staticmethod
    def from_pretrained(preprained_path: str, device: Union[torch.device, str] = "cpu"):
        with open(f"{preprained_path}/model_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config = LlamaConfig(**config_dict)
        model = LlamaModel(config)
        model.load_state_dict(
            torch.load(
                f"{preprained_path}/pytorch_model.pt", 
                weights_only=True, 
                map_location=device), 
            strict=True
        )
        return model
