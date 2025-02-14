# EasyLM, A python package for training Language Models from scratch

EasyLM is a python package for training Language Models from scratch. It provides a simple interface to train and evaluate Language Models on various tasks.

## Installation

```bash
pip install easylm
```

For development previews, you can install the development version from GitHub:

```bash
pip install git+https://github.com/sayedshaun/easylm.git
```

## Usage

```python
from easylm.models import GPTModel
from easylm.data import NextWordPredDataset
from easylm.tokenizer import Tokenizer
from easylm.config import GPTConfig, TrainingConfig
from easylm.trainer import Trainer


data_path = "data.txt"
tokenizer = Tokenizer(data_dir=data_path, vocab_size=10000)
dataset = NextWordPredDataset(data_path, tokenizer, max_seq_len=512)
model_config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_heads=12,
    num_layers=12,
    norm_epsilon=1e-5,
    dropout=0.1,
    max_seq_len=512
)
model = GPTModel(model_config)
training_config = TrainingConfig(
    learning_rate=3e-5,
    num_epochs=1,
    batch_size=32,
    warmup_steps=1000,
    total_steps=10000,
    max_seq_len=512
)
trainer = Trainer(training_config)
trainer.train(model, dataset)
trainer.evaluate()
```


   
