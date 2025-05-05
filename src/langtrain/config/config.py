import torch
from dataclasses import dataclass, field
from typing import Union
from torch.utils.data import DataLoader, Dataset


@dataclass
class GPTConfig:
    vocab_size: Union[int, None] = field(default=None, metadata={"help": "Number of tokens in the vocabulary. Typically the tokenizer's vocab size."})
    hidden_size: Union[int, None] = field(default=None, metadata={"help": "Dimensionality of the hidden layers and embeddings."})
    num_heads: Union[int, None] = field(default=None, metadata={"help": "Number of attention heads in the multi-head attention mechanism."})
    num_layers: Union[int, None] = field(default=None, metadata={"help": "Number of transformer blocks (layers) in the model."})
    norm_epsilon: Union[float, None] = field(default=None, metadata={"help": "Epsilon value for layer normalization to avoid division by zero."})
    dropout: Union[float, None] = field(default=None, metadata={"help": "Dropout probability applied in various layers to prevent overfitting."})
    max_seq_len: Union[int, None] = field(default=None, metadata={"help": "Maximum sequence length (context window) the model can handle."})



@dataclass
class LlamaConfig:
    vocab_size: Union[int, None] = field(default=None, metadata={"help": "Number of tokens in the vocabulary. Typically the tokenizer's vocab size."})
    hidden_size: Union[int, None] = field(default=None, metadata={"help": "Dimensionality of the hidden layers and embeddings."})
    num_heads: Union[int, None] = field(default=None, metadata={"help": "Number of attention heads in the multi-head attention mechanism."})
    num_layers: Union[int, None] = field(default=None, metadata={"help": "Number of transformer blocks (layers) in the model."})
    norm_epsilon: Union[float, None] = field(default=None, metadata={"help": "Epsilon value for layer normalization to avoid division by zero."})
    dropout: Union[float, None] = field(default=None, metadata={"help": "Dropout probability applied in various layers to prevent overfitting."})
    max_seq_len: Union[int, None] = field(default=None, metadata={"help": "Maximum sequence length (context window) the model can handle."})


@dataclass
class BertConfig:
    vocab_size: Union[int, None] = field(default=None, metadata={"help": "Number of tokens in the vocabulary. Typically the tokenizer's vocab size."})
    hidden_size: Union[int, None] = field(default=None, metadata={"help": "Dimensionality of the hidden layers and embeddings."})
    num_heads: Union[int, None] = field(default=None, metadata={"help": "Number of attention heads in the multi-head attention mechanism."})
    num_layers: Union[int, None] = field(default=None, metadata={"help": "Number of transformer blocks (layers) in the model."})
    norm_epsilon: Union[float, None] = field(default=None, metadata={"help": "Epsilon value for layer normalization to avoid division by zero."})
    dropout: Union[float, None] = field(default=None, metadata={"help": "Dropout probability applied in various layers to prevent overfitting."})
    max_seq_len: Union[int, None] = field(default=None, metadata={"help": "Maximum sequence length (context window) the model can handle."})


@dataclass
class VITConfig:
    image_size: Union[int, None] = field(default=None, metadata={"help": "Input image size (height/width in pixels)."})
    patch_size: Union[int, None] = field(default=None, metadata={"help": "Size of each image patch (e.g., 16x16)."})
    color_channels: Union[int, None] = field(default=None, metadata={"help": "Number of color channels (e.g., 3 for RGB)."})
    hidden_size: Union[int, None] = field(default=None, metadata={"help": "Dimensionality of hidden embeddings."})
    num_heads: Union[int, None] = field(default=None, metadata={"help": "Number of attention heads in the transformer."})
    num_layers: Union[int, None] = field(default=None, metadata={"help": "Number of transformer encoder layers."})
    norm_epsilon: Union[float, None] = field(default=None, metadata={"help": "Epsilon value for layer normalization."})
    dropout: Union[float, None] = field(default=None, metadata={"help": "Dropout probability throughout the model."})


@dataclass
class TrainingConfig:
    epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the contents of the output directory."})
    batch_size: Union[int, None] = field(default=2, metadata={"help": "Batch size per device during training."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Initial learning rate for optimizer."})
    weight_decay: float = field(default=0, metadata={"help": "Weight decay to apply (if any)."})
    lr_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for numerical stability in optimizer."})
    device: Union[str, torch.device, None] = field(default=None, metadata={"help": "Device to train on (e.g., 'cuda', 'cpu')."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before backpropagation."})
    gradient_clipping: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping."})
    precision: str = field(default="fp16", metadata={"help": "Floating point precision to use (e.g., 'bf16' 'fp16', 'fp32')."})
    seed: Union[int, None] = field(default=42, metadata={"help": "Random seed for reproducibility."})
    validation_steps: Union[int, None] = field(default=500, metadata={"help": "Number of steps between validation runs."})
    logging_steps: Union[int, None] = field(default=500, metadata={"help": "Number of steps between logging metrics."})
    save_steps: Union[int, None] = field(default=500, metadata={"help": "Number of steps between saving model checkpoints."})
    num_workers: Union[int, None] = field(default=0, metadata={"help": "Number of subprocesses for data loading."})
    shuffle_data: bool = field(default=False, metadata={"help": "Whether to shuffle training data."})
    pin_memory: bool = field(default=False, metadata={"help": "Whether to pin memory in DataLoader."})
    num_checkpoints: int = field(default=1, metadata={"help": "Number of checkpoints to keep."})
    early_stopping: bool = field(default=False, metadata={"help": "Enable early stopping based on validation loss."})
    patience: int = field(default=5, metadata={"help": "Number of evaluations to wait before early stopping."})
    report_to_wandb: Union[bool, None] = field(default=None, metadata={"help": "Report metrics to Weights & Biases."})
    wandb_project: Union[str, None] = field(default="langtrain", metadata={"help": "WandB project name."})
    distributed_training: Union[str, None] = field(default=None, metadata={"help": "Type of distributed training to use (e.g., 'ddp', 'dp')."})
    distributed_backend: str = field(default="nccl", metadata={"help": "Backend for distributed training (e.g., 'nccl', 'gloo')."})
    find_unused_parameters: bool = field(default=True, metadata={"help": "Find unused parameters in DDP training."})
    drop_dataloader_last: bool = field(default=False, metadata={"help": "Drop last batch if it is smaller than batch size."})
    monitor_loss_for: str = field(default="train_loss", metadata={"help": "Which loss to monitor (e.g., 'train_loss', 'val_loss')."})

