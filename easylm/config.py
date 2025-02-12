import torch
from dataclasses import dataclass
from typing import Union
from torch.utils.data import DataLoader


@dataclass
class GPTConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None

@dataclass
class LlamaConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None

@dataclass
class BERTConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None


@dataclass
class VITConfig:
    image_size: Union[int, None] = None
    patch_size: Union[int, None] = None
    color_channels: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None


@dataclass
class TrainingConfig:
    model: Union[torch.nn.Module, None] = None
    epochs: Union[int, None] = None
    batch_size: Union[int, None] = None
    learning_rate: Union[float, None] = None
    train_dataloader: Union[DataLoader, None] = None
    val_dataloader: Union[DataLoader, None] = None
    test_dataloader: Union[DataLoader, None] = None
    device: Union[str, None] = None
    gradient_accumulation_steps: Union[int, None] = None
    gradient_clipping: Union[float, None] = None
    precision: Union[str, None] = None
    seed: Union[int, None] = None
    validation_steps: Union[int, None] = None
    logging_steps: Union[int, None] = None
    save_steps: Union[int, None] = None
    optimizer: Union[torch.optim.Optimizer, None] = None



__all__ = ["GPTConfig", "LlamaConfig", "BERTConfig", "VITConfig", "TrainingConfig"]