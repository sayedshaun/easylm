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
    epochs: Union[int, None] =  1
    overwrite_output_dir: bool = False
    batch_size: Union[int, None] = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0
    lr_epsilon: float = 1e-8
    device: Union[str, torch.device, None] = None
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    precision: str = "fp16"
    seed: Union[int, None] = 42
    validation_steps: Union[int, None] = 500
    logging_steps: Union[int, None] = 500
    save_steps: Union[int, None] = 500
    num_workers: Union[int, None] = 0
    shuffle_data: bool = False
    pin_memory: bool = False
    num_checkpoints: int = 1
    early_stopping: bool = False
    patience: int = 5
    report_to_wandb: Union[bool, None] = None
    wandb_project: Union[str, None] = "langtrain"
    distributed_training: Union[str, None] = None
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = True
    drop_dataloader_last: bool = False
    monitor_loss_for: str = "train_loss"




__all__ = ["GPTConfig", "LlamaConfig", "BertConfig", "VITConfig", "TrainingConfig"]
