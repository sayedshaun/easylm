import torch
from dataclasses import dataclass
from typing import Union
from torch.utils.data import DataLoader, Dataset


@dataclass
class GPTConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None
    device: Union[str, torch.device, None] = None

@dataclass
class LlamaConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None
    device: Union[str, torch.device, None] = None

@dataclass
class BERTConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None
    device: Union[str, torch.device, None] = None


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
    device: Union[str, torch.device, None] = None


@dataclass
class TrainingConfig:
    model: Union[torch.nn.Module, None] = None
    epochs: Union[int, None] = 1
    batch_size: Union[int, None] = 8
    learning_rate: Union[float, None] = 1e-5
    train_data: Union[DataLoader, Dataset, None] = None
    val_data: Union[DataLoader, Dataset, None] = None
    test_data: Union[DataLoader, Dataset, None] = None
    device: Union[str, torch.device, None] = None
    gradient_accumulation_steps: Union[int, None] = 1
    gradient_clipping: Union[float, None] = 1
    precision: Union[str, None] = torch.float16
    seed: Union[int, None] = 42
    validation_steps: Union[int, None] = 500
    logging_steps: Union[int, None] = 500
    save_steps: Union[int, None] = 500
    optimizer: Union[torch.optim.Optimizer, None] = None



__all__ = ["GPTConfig", "LlamaConfig", "BERTConfig", "VITConfig", "TrainingConfig"]