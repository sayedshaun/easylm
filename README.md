![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic&logo=PyTorch&logoColor=white)

![alt text](assets/logo.png)

## A python package for training Language Models from scratch with few lines of code

EasyLM is a python package for training Language Models from scratch. It provides a simple interface to train large Language Models from scratch with few lines of code.

## Installation

### Stable Version
```bash
pip install langtrain
```

### Development Version
```bash
pip install git+https://github.com/sayedshaun/langtrain.git
```

## Usage

### Training

```python
from langtrain.model import LlamaModel
from langtrain.data import IterableCausalDataset
from langtrain.tokenizer import Tokenizer
from langtrain.config import TrainingConfig, LlamaConfig
from langtrain.trainer import Trainer
from langtrain.utils import trainable_parameters


data_path = "data_directory"
tokenizer = Tokenizer(data_path, vocab_size=5000)
dataset = IterableCausalDataset(data_path, tokenizer, n_ctx=50, batch=10000)
model = LlamaModel(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_seq_len=50,
        norm_epsilon=1e-5
    )
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    model_name="nano-llama",
    collate_fn=IterableCausalDataset.collate_fn,
    config=TrainingConfig(
        train_data=dataset,
        learning_rate=1e-4,
        epochs=5,
        batch_size=8,
        device="cuda",
        logging_steps=100,
        num_checkpoints=3,
        report_to_wandb=True,
    )
)
print(trainable_parameters(model))
trainer.from_checkpoint("nano-llama/checkpoint-200")
trainer.train()
```


### Pretrained Detailes:
Once the model is trained the pretrained dicretory will looks like this:
```
nano-llama/
    ├── /checkpoint-200
    ├── train_config.yaml
    ├── model_config.yaml
    ├── pytorch_model.pt
    ├── VOCAB.model
    └── VOCAB.vocab
```

### Inference

```python
from langtrain.model import LlamaModel
from langtrain.tokenizer import Tokenizer

tokenizer = Tokenizer.from_pretrained("nano-llama")
model = LlamaModel.from_pretrained("nano-llama")
inputs = tokenizer.encode("Sherlock Holmes")
output = model.generate(inputs, eos_id=tokenizer.eos_token_id, max_new_tokens=50)
tokenizer.decode(output)
```