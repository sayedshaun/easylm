This guide explains how to install the `langtrain` package.

## Installation

To install the stable version of `langtrain`, run the following command:

```bash
pip install langtrain

# Development Version
pip install git+https://github.com/sayedshaun/langtrain.git
```


---


This tutorial demonstrates how to train a language model using the `langtrain` package.

## Tokenizer
### Sentence Piece Tokenizer:
```python
tokenizer = lt.tokenizer.SentencePieceTokenizer(
    "your_data_directory",           # Directory with training data (text files) for tokenizer
    vocab_size=5000,     # Number of tokens in the vocabulary. Higher values capture more unique tokens.
    retrain=True,        # If True, the tokenizer will be retrained from scratch even if a model already exists
    model_type="bpe"     # Tokenization algorithm type: "bpe" (Byte-Pair Encoding), "unigram", "char", or "word"
)
```

## Dataset
### SimpleCausalDataset
```python
dataset = lt.dataset.SimpleCausalDataset(
    data_path,            # Directory with text files to tokenize and prepare as training samples
    tokenizer=tokenizer,  # The pretrained SentencePiece tokenizer used to tokenize the dataset
    n_ctx=512             # Maximum context length (number of tokens) for each training sample
)
```
### IterableCausalDataset
```python
dataset = lt.dataset.IterableCausalDataset(
    data_path,            # Directory with text files to tokenize on the fly
    tokenizer=tokenizer,  # The tokenizer used to process text into tokens
    n_ctx=512,            # Length of each sample in number of tokens
    stride=1              # Step size between consecutive samples; a stride of 1 means maximum overlap (efficient for training)
)
```
### LazyCausalDataset
```python
dataset = lt.dataset.LazyCausalDataset(
    data_path,             # Directory with raw text files
    tokenizer=tokenizer,   # Tokenizer used to preprocess the dataset
    n_ctx=512,             # Context window size (number of tokens per sample)
    token_caching=True,    # Enables token caching to avoid re-tokenizing text files repeatedly
    cache_dir="cache"      # Directory to store cached tokenized data; if None, uses default location
)
```

## Model
Go to [Model Argumenets](model_config.md) for more details

### LlamaModel
```python
model = lt.model.LlamaModel(
    lt.model.LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        norm_epsilon=1e-5,
        max_seq_len=512
    )
)
```
Configs for Causal Models are almost similar.

### Vision Transformer
```python
model = lt.model.VisionTransformer(
    lt.model.VisionTransformerConfig(
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        norm_epsilon=1e-5,
        max_seq_len=512
    )
)
```


## Trainer
### Training Arguments
Go to [Training Arguments](training_config.md) for more details

```python
train_config = lt.config.TrainingConfig(
    train_data=dataset,
    val_data=dataset,
    epochs=10,
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    report_to_wandb=True,
    wandb_project="langtrain-tutorial",
    save_steps=500,
    save_total_limit=2,
    distributed_backend="ddp"
)
```

### Trainer
```python
trainer = lt.trainer.Trainer(
    model=model,                        # The model to train
    tokenizer=tokenizer,                # The tokenizer used by the model
    model_name="nano-llama",            # Name of the model to save
    collate_fn=lt.utils.collate_fn,     # Function to collate training samples into batches
    config=train_config                 # Training configuration
)
```
#### Start Training
```python
trainer.train()
```
If you have checkpoints and you want to continue training from there, use
```python
trainer.train(resume_from_checkpoint=True)
trainer.train()
```