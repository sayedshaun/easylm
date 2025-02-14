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
from easylm.model import LlamaModel
from easylm.data import NextWordPredDataset
from easylm.tokenizer import Tokenizer
from easylm.config import TrainingConfig, LlamaConfig
from easylm.trainer import Trainer

data_path = "data/SherlockHolmes.txt"
tokenizer = Tokenizer(data_path, vocab_size=5000)
dataset = NextWordPredDataset(data_path, tokenizer, max_seq_len=50)
model = LlamaModel(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        norm_epsilon=1e-5,
        dropout=0.1,
        max_seq_len=50
    )
)
trainer = Trainer(
    TrainingConfig(
        model=model,
        train_data=dataset,
        learning_rate=5e-5,
        epochs=10,
        batch_size=8,
        device="cuda",
        logging_steps=10
    )
)
trainer.train()
trainer.evaluate()
```