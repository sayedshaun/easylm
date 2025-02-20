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

### Training

```python
from easylm.model import LlamaModel
from easylm.data import CauslLMDataset
from easylm.tokenizer import Tokenizer
from easylm.config import TrainingConfig, LlamaConfig
from easylm.trainer import Trainer

data_path = "data"
tokenizer = Tokenizer(data_path, vocab_size=5000)
dataset = CauslLMDataset(data_path, tokenizer, max_seq_len=50)
model = LlamaModel(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_seq_len=50
    )
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    pretrained_path="nano-llama",
    config=TrainingConfig(
        train_data=dataset,
        learning_rate=1e-4,
        epochs=10,
        batch_size=8,
        device="cpu",
        logging_steps=10
    )
)
trainer.train()
```

### Inference

```python
from easylm.model import LlamaModel
from easylm.tokenizer import Tokenizer

tokenizer = Tokenizer.from_pretrained("nano-llama")
model = LlamaModel.from_pretrained("nano-llama")
inputs = tokenizer.encode("Sherlock Holmes")
output = model.generate(inputs, tokenizer.eos_token_id, max_seq_len=50)
tokenizer.decode(output)
```