# Training Configuration

This document describes the configurable parameters used during training.

## 🔧 Model & Training Basics

| Parameter                  | Type                        | Default     | Description |
|---------------------------|-----------------------------|-------------|-------------|
| `model`                   | `torch.nn.Module` or `None` | `None`      | The model to be trained. |
| `epochs`                  | `int` or `None`             | `None`      | Total number of training epochs. |
| `batch_size`              | `int` or `None`             | `None`      | Number of samples per training batch. |
| `learning_rate`           | `float`                     | `None`      | Initial learning rate. |
| `weight_decay`            | `float`                     | `0`         | Weight decay for optimizer (L2 regularization). |
| `lr_epsilon`              | `float`                     | `1e-8`      | Minimum learning rate epsilon for optimizers like Adam. |

## 📦 Data Configuration

| Parameter       | Type                                     | Default | Description |
|----------------|------------------------------------------|---------|-------------|
| `train_data`   | `DataLoader`, `Dataset`, or `None`       | `None`  | Training dataset or dataloader. |
| `val_data`     | `DataLoader`, `Dataset`, or `None`       | `None`  | Validation dataset or dataloader. |
| `test_data`    | `DataLoader`, `Dataset`, or `None`       | `None`  | Test dataset or dataloader. |
| `shuffle_data` | `bool`                                   | `False` | Whether to shuffle data each epoch. |
| `num_workers`  | `int` or `None`                          | `0`     | Number of subprocesses used for data loading. |
| `pin_memory`   | `bool`                                   | `False` | Use pinned memory during data loading (recommended for GPU training). |

## 🧠 Device & Precision

| Parameter      | Type                           | Default | Description |
|----------------|--------------------------------|---------|-------------|
| `device`       | `str`, `torch.device`, or `None` | `None`  | Device to use (`"cpu"` or `"cuda"`). |
| `precision`    | `str`                          | `"fp16"`| Precision setting (`"fp16"`, `"fp32"`). |

## 🔄 Optimization Settings

| Parameter                     | Type     | Default | Description |
|------------------------------|----------|---------|-------------|
| `gradient_accumulation_steps`| `int`    | `1`     | Number of steps to accumulate gradients before updating. |
| `gradient_clipping`          | `float`  | `1.0`   | Maximum norm for gradient clipping. |
| `seed`                       | `int` or `None` | `None` | Random seed for reproducibility. |

## 📉 Evaluation & Logging

| Parameter        | Type          | Default | Description |
|------------------|---------------|---------|-------------|
| `validation_steps` | `int` or `None` | `500`   | Perform validation every N steps. |
| `logging_steps`    | `int` or `None` | `500`   | Log training info every N steps. |
| `save_steps`       | `int` or `None` | `500`   | Save checkpoints every N steps. |
| `num_checkpoints`  | `int`         | `1`     | Maximum number of checkpoints to keep. |

## 📊 Logging & Monitoring

| Parameter         | Type         | Default   | Description |
|-------------------|--------------|-----------|-------------|
| `report_to_wandb` | `bool` or `None` | `None`  | Whether to report metrics to Weights & Biases. |
| `wandb_project`   | `str` or `None`  | `"easylm"` | W&B project name. |

## 🧪 Distributed Training

| Parameter             | Type            | Default | Description |
|-----------------------|-----------------|---------|-------------|
| `distributed_backend` | `str` or `None` | `None`  | Backend for distributed training (e.g., `"ddp"`, `"dp"`). |