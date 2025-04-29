from typing import Union
import torch
import torch.nn.functional as F
import yaml
from src.config import GPTConfig
from src.nn import TransformerDecoderBlock, PositionalEmbeddings
from torch.nn import LayerNorm, Linear
from src.utils import CausalModelOutput


class GPTModel(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None: 
        super(GPTModel, self).__init__()
        self.config = config
        self.embedding = PositionalEmbeddings(
            config.vocab_size, 
            config.hidden_size, 
            config.max_seq_len, 
            config.dropout
        )
        self.blocks = torch.nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_size, 
                    config.num_heads,
                    config.norm_epsilon,
                    config.dropout, 
                    ) 
                for _ in range(config.num_layers)
            ]
        )
        self.norm = LayerNorm(config.hidden_size, config.norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.linear = Linear(config.hidden_size, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

    def forward(
            self: "GPTModel", 
            input_ids: torch.Tensor, 
            target_ids: Union[torch.Tensor, None] = None, 
            causal_mask: bool = True
            )-> CausalModelOutput:
        mask = self._make_causal_mask(input_ids) if causal_mask else None
        input_ids = self.embedding(input_ids)
        for block in self.blocks:
            input_ids = block(input_ids, mask)
        input_ids = self.dropout(self.norm(input_ids))
        logits = self.linear(input_ids)
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            return CausalModelOutput(loss=loss, logits=logits)
        else:
            return CausalModelOutput(logits=logits)
    
    def _make_causal_mask(self, input_ids: torch.Tensor)-> torch.Tensor:
        N, S = input_ids.shape # batch, seq_len
        mask = torch.ones(S, S, dtype=torch.bool, device=input_ids.device).tril(diagonal=0)
        return mask

    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 20, 
        context_size: int = 1, 
        temperature: float = 0.7, 
        top_k: Union[int, None] = None, 
        eos_id: Union[int, None] = None) -> torch.Tensor:  
          
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
        config = GPTConfig(**config_dict)
        model =GPTModel(config)
        model.load_state_dict(
            torch.load(
                f"{preprained_path}/pytorch_model.pt", 
                weights_only=True, 
                map_location=device), 
            strict=True
        )
        return model