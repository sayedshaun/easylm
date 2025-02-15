from typing import List, Union
import numpy as np
import torch
from easylm.data import Tokenizer


def trainable_parameters(model: torch.nn.Module)->str:
    trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    return f"Total Trainable Params: {trainable_params/1e6:.2f} M"


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    This implementation is adapted from HuggingFace's transformers library.
    
    Args:
        logits (torch.Tensor): Logits distribution of shape (1, vocab_size).
        top_k (int): Keep only top_k tokens with highest probability (0 means no filtering).
        top_p (float): Keep the top tokens with cumulative probability >= top_p (0.0 means no filtering).
        filter_value (float): The value to assign to filtered logits.
    
    Returns:
        torch.Tensor: The filtered logits.
    """
    top_k = max(top_k, 0)
    if top_k > 0:
        # Remove all tokens with a probability less than the top_k tokens.
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        # Sort the logits in descending order.
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to retain at least one token.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set filtered tokens to the filter value.
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value
    return logits


def generate(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50, 
    temperature: float = 1.0,
    top_k: int = 0, 
    top_p: float = 0.0
) -> str:
    """
    Generate text using the trained autoregressive model.
    
    Args:
        model: The trained language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompt (str): The input prompt to condition generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature; lower values make the predictions more deterministic.
        top_k (int): If >0, keep only top_k tokens for sampling.
        top_p (float): If >0.0, apply nucleus (top-p) sampling.
        
    Returns:
        str: The generated text (which includes the prompt).
    """
    # Encode the prompt into token IDs
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Start with the prompt tokens
    generated = input_ids
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass: obtain logits for the last token in the sequence.
            outputs = model(generated)
            logits = outputs[0][:, -1, :]  # shape: [1, vocab_size]
            logits = logits / temperature  # Adjust for temperature
            
            # Optionally, filter logits using top_k and/or top_p
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            
            # Convert logits to probabilities and sample the next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the predicted token to the sequence
            generated = torch.cat((generated, next_token), dim=1)
            
            # If an end-of-sequence token is generated, stop early.
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decode the tokens back to text
    generated_text = tokenizer.decode(generated.squeeze().tolist())
    return generated_text
