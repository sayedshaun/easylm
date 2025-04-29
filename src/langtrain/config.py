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
    epochs: Union[int, None] =  None
    batch_size: Union[int, None] = None
    learning_rate: float = None
    weight_decay: float = 0
    lr_epsilon: float = 1e-8
    train_data: Union[DataLoader, Dataset, None] = None
    val_data: Union[DataLoader, Dataset, None] = None
    test_data: Union[DataLoader, Dataset, None] = None
    device: Union[str, torch.device, None] = None
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    precision: str = "fp16"
    seed: Union[int, None] = None
    validation_steps: Union[int, None] = 500
    logging_steps: Union[int, None] = 500
    save_steps: Union[int, None] = 500
    num_workers: Union[int, None] = 0
    shuffle_data: bool = False
    pin_memory: bool = False
    num_checkpoints: int = 1
    report_to_wandb: Union[bool, None] = None
    wandb_project: Union[str, None] = "easylm"
    distributed_backend: Union[str, None] = None



__all__ = ["GPTConfig", "LlamaConfig", "BertConfig", "VITConfig", "TrainingConfig"]
