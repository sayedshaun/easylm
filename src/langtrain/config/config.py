import torch
from typing import Union
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for GPT (Generative Pre-trained Transformer).

    Attributes:
        vocab_size (int): Number of tokens in the vocabulary. Typically the tokenizer's vocab size.
        hidden_size (int): Dimensionality of the hidden layers and embeddings.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        num_layers (int): Number of transformer blocks (layers) in the model.
        norm_epsilon (float or None): Epsilon value for layer normalization to avoid division by zero.
        dropout (float): Dropout probability applied in various layers to prevent overfitting.
        max_seq_len (int): Maximum sequence length (context window) the model can handle.
    """
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None


@dataclass
class LlamaConfig:
    """
    Configuration for LLaMA (Large Language Model from Meta).

    Attributes:
        vocab_size (int): Number of tokens in the vocabulary. Typically the tokenizer's vocab size.
        hidden_size (int): Dimensionality of the hidden layers and embeddings.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        num_layers (int): Number of transformer blocks (layers) in the model.
        norm_epsilon (float): Epsilon value for layer normalization to avoid division by zero.
        dropout (float or None): Dropout probability applied in various layers to prevent overfitting.
        max_seq_len (int): Maximum sequence length (context window) the model can handle.
    """
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None


@dataclass
class BertConfig:
    """
    Configuration for BERT (Bidirectional Encoder Representations from Transformers).

    Attributes:
        vocab_size (int): Number of tokens in the vocabulary. Typically the tokenizer's vocab size.
        hidden_size (int): Dimensionality of the hidden layers and embeddings.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        num_layers (int): Number of transformer blocks (layers) in the model.
        norm_epsilon (float): Epsilon value for layer normalization to avoid division by zero.
        dropout (float): Dropout probability applied in various layers to prevent overfitting.
        max_seq_len (int): Maximum sequence length (context window) the model can handle.
    """
    vocab_size: Union[int, None] = None
    hidden_size: Union[int, None] = None
    num_heads: Union[int, None] = None
    num_layers: Union[int, None] = None
    norm_epsilon: Union[float, None] = None
    dropout: Union[float, None] = None
    max_seq_len: Union[int, None] = None



@dataclass
class VITConfig:
    """
    Configuration for Vision Transformer (ViT).

    Attributes:
        image_size (int): Input image size (height/width in pixels).
        patch_size (int): Size of each image patch (e.g., 16x16).
        color_channels (int): Number of color channels (e.g., 3 for RGB).
        hidden_size (int): Dimensionality of hidden embeddings.
        num_heads (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer encoder layers.
        norm_epsilon (float or None): Epsilon value for layer normalization.
        dropout (float): Dropout probability throughout the model.
    """
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
    """
    Configuration for training.

    Attributes:
        epochs (int): Number of training epochs.
        overwrite_output_dir (bool): Overwrite the contents of the output directory.
        batch_size (int): Batch size per device during training.
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight decay to apply (if any).
        lr_epsilon (float): Epsilon for numerical stability in optimizer.
        device (str, torch.device, or None): Device to train on (e.g., 'cuda', 'cpu').
        gradient_accumulation_steps (int): Steps to accumulate gradients before backpropagation.
        gradient_clipping (float): Maximum gradient norm for clipping.
        precision (str): Floating point precision (e.g., 'bf16', 'fp16', 'fp32').
        seed (int): Random seed for reproducibility.
        validation_steps (int): Steps between validation runs.
        logging_steps (int): Steps between logging metrics.
        save_steps (int): Steps between saving model checkpoints.
        num_workers (int): Number of subprocesses for data loading.
        shuffle_data (bool): Whether to shuffle training data.
        pin_memory (bool): Whether to pin memory in DataLoader.
        num_checkpoints (int): Number of checkpoints to keep.
        early_stopping (bool): Enable early stopping based on validation loss.
        patience (int): Evaluations to wait before early stopping.
        report_to_wandb (bool or None): Report metrics to Weights & Biases.
        wandb_project (str or None): WandB project name.
        distributed_training (str or None): Type of distributed training (e.g., 'ddp', 'dp').
        distributed_backend (str): Backend for distributed training (e.g., 'nccl', 'gloo').
        find_unused_parameters (bool): Find unused parameters in DDP training.
        drop_dataloader_last (bool): Drop last batch if it is smaller than batch size.
        monitor_loss_for (str): Which loss to monitor (e.g., 'train_loss', 'val_loss').
    """
    epochs: int = 1
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
    monitor_loss_for: str = "train"
