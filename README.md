# EasyLM, A python package for training Language Models from scratch

EasyLM is a python package for training Language Models from scratch. It provides a simple interface to train large Language Models from scratch with few lines of code.

## Installation

```bash
pip install git+https://github.com/sayedshaun/easylm.git
```

## Usage

### Training

```python
from easylm.model import LlamaModel
from easylm.data import CausalLMDataset
from easylm.tokenizer import Tokenizer
from easylm.config import TrainingConfig, LlamaConfig
from easylm.trainer import Trainer
from easylm.utils import trainable_parameters


data_path = "data"
tokenizer = Tokenizer(data_path, vocab_size=5000)
dataset = CausalLMDataset(data_path, tokenizer, max_seq_len=50)
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
    config=TrainingConfig(
        train_data=dataset,
        learning_rate=1e-4,
        epochs=10,
        batch_size=8,
        device="cuda",
        logging_steps=10
    )
)
print(trainable_parameters(model))
trainer.train()
```


### Pretrained Detailes:
Once the model is trained the pretrained dicretory will looks like this:
```
nano-llama/
    ├── train_config.yaml
    ├── model_config.yaml
    ├── pytorch_model.pt
    ├── VOCAB.model
    └── VOCAB.vocab
```

### Inference

```python
from easylm.model import LlamaModel
from easylm.tokenizer import Tokenizer

tokenizer = Tokenizer.from_pretrained("nano-llama")
model = LlamaModel.from_pretrained("nano-llama")
inputs = tokenizer.encode("Sherlock Holmes")
output = model.generate(inputs, eos_id=tokenizer.eos_token_id, max_new_tokens=50)
tokenizer.decode(output)
```