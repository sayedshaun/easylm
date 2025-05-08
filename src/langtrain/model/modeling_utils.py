import os
import yaml
import torch
from typing import NamedTuple, Union


class CausalModelOutput(NamedTuple):
    """
    Output type for Causal Models (e.g. GPT, Llama)
    """
    loss : Union[torch.Tensor, None] = None
    logits : Union[torch.Tensor, None] = None
    last_hidden_state: Union[torch.Tensor, None] = None


class MaskedModelOutput(NamedTuple):
    """
    Output type for Masked Models (e.g. BERT, RoBERTa)
    """
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None
    hidden_states: Union[torch.Tensor, None] = None
    pooler_output: Union[torch.Tensor, None] = None


class CausalGenerationMixin:
    """
    This class provides `generate` method for generating sequences of tokens
    """
    def generate(self: "CausalGenerationMixin", 
                 input_ids: torch.Tensor, 
                 max_new_tokens: int = 20, 
                 context_size: int = 1, 
                 temperature: float = 0.7, 
                 top_k: Union[int, None] = None, 
                 eos_id: Union[int, None] = None
        ) -> torch.Tensor:  
        """
        Generates a sequence of tokens based on the input IDs and model parameters.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 20.
            context_size (int, optional): The size of the context window. Defaults to 1.
            temperature (float, optional): The temperature for sampling. Defaults to 0.7.
            top_k (Union[int, None], optional): The number of top tokens to consider. Defaults to None.
            eos_id (Union[int, None], optional): The end-of-sequence token ID. Defaults to None.

        Returns:
            torch.Tensor: The generated sequence of token IDs.
        """
          
        for _ in range(max_new_tokens):           
            idx_cond = input_ids[:, -context_size:] 
            with torch.no_grad():    
                outputs = self.forward(idx_cond, causal_mask=False)
                logits = outputs.logits 
                logits = logits[:, -1, :] 

                if top_k is not None:                   
                    top_logits, _ = torch.topk(logits, top_k)    
                    min_val = top_logits[:, -1]    
                    logits = torch.where(
                        logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits
                        ) 
                
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
    

class MaskGenerationMixin:
    """
    This provides `fill_mask` method for filling masked input IDs with predicted token IDs
    """
    @torch.inference_mode(mode=True)
    def fill_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Fills the masked input IDs with predicted token IDs.

        Args:
            input_ids (torch.Tensor): The input token IDs with masked positions.

        Returns:
            torch.Tensor: The predicted token IDs for the masked positions.
        """
    
        self.eval()
        outputs = self.forward(input_ids)  # shape: (B, L+1, vocab_size)
        logits = outputs.logits
        # Remove the logits for the [CLS] token (first token)
        return logits[:, 1:, :]
    

class LoadFromPretrainedMixin:
    """
    This mixin provides a `from_pretrained` method for loading pretrained models.
    """
    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: Union[torch.device, str] = "cpu"):
        """
        Loads a pretrained model from the specified path.
        """
        config_path = os.path.join(pretrained_path, "model_config.yaml")
        model_path = os.path.join(pretrained_path, "pytorch_model.pt")

        # Ensure the paths exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the model configuration
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = cls.config_class(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.to(device)
        return model
