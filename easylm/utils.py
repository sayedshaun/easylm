from typing import List, Union
import numpy as np
import torch
from easylm.data import Tokenizer


def get_embeddings(model, text: Union[str, List[str]], return_type:str="np")->Union[torch.Tensor, List, np.ndarray]:
    input_ids = Tokenizer.encode(text)
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    hidden_states = model(input_ids, return_hidden_states=True)
    out_embed = hidden_states.mean(1).detach().cpu().numpy()

    valid_types = ["torch", "np", "list"]
    if return_type not in valid_types:
        raise ValueError(f"return_type must be one of {valid_types}")
    
    match return_type:
        case "torch":
            return torch.tensor(out_embed)
        case "list":
            return out_embed.tolist()[0]
    return out_embed 


def trainable_parameters(model: torch.nn.Module)->str:
    trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    return f"Total Trainable Params: {trainable_params/1e6:.2f} M"