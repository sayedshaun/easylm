import torch
from dataclasses import dataclass
from typing import Union, NamedTuple
from easylm.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset


@dataclass
class GPTConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = 768
    num_heads: Union[int, None] = 12
    num_layers: Union[int, None] = 12
    norm_epsilon: Union[float, None] = 1e-5 
    dropout: Union[float, None] = 0.1
    max_seq_len: Union[int, None] = 512


@dataclass
class LlamaConfig:
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = 1e-5
    dropout: Union[float, None] = 0.1
    max_seq_len: Union[int, None] = 512


@dataclass
class BertConfig:
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
    epochs: Union[int, None] = 1
    batch_size: Union[int, None] = 8
    learning_rate: Union[float, None] = 1e-5
    train_data: Union[DataLoader, Dataset, None] = None
    val_data: Union[DataLoader, Dataset, None] = None
    test_data: Union[DataLoader, Dataset, None] = None
    device: Union[str, torch.device, None] = None
    gradient_accumulation_steps: Union[int, None] = 1
    gradient_clipping: Union[float, None] = 1
    precision: str = "fp32"
    seed: Union[int, None] = 42
    validation_steps: Union[int, None] = 500
    logging_steps: Union[int, None] = 500
    save_steps: Union[int, None] = 500



__all__ = ["GPTConfig", "LlamaConfig", "BertConfig", "VITConfig", "TrainingConfig"]