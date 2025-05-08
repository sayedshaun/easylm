![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic&logo=PyTorch&logoColor=white)


![alt text](docs/static/logo.png)


## A python package for training Language Models from scratch with few lines of code

LangTrain is a python package for training Language Models from scratch. It provides a simple interface to train large Language Models from scratch with few lines of code.

## Installation

#### Stable Version
```bash
pip install langtrain
```

#### Development Version
```bash
pip install git+https://github.com/sayedshaun/langtrain.git
```

## Usage

#### Quick Start

```python
import langtrain as lt

data_path = "data_directory"
tokenizer = lt.tokenizer.SentencePieceTokenizer(data_path, vocab_size=5000)
dataset = lt.dataset.SimpleCausalDataset(data_path, tokenizer, n_ctx=512)
model = lt.model.LlamaModel(
    lt.model.LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        hidden_layers=8,
        num_heads=8,
        dropout=0.2,
        norm_epsilon=1e-6,
        max_seq_len=dataset.n_ctx,
    )
)
train_config=lt.config.TrainingConfig(
    epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    device="cuda",
    precision="fp16",
)
trainer = lt.trainer.Trainer(
    model=model,
    train_config=train_config,
    dataset=dataset,
    tokenizer=tokenizer,
    collate_fn=lt.utils.collate_fn,
    model_name="nano-llama",
)
trainer.train()
```

#### Pretrained Detailes:
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

#### Inference

```python
import langtrain as lt

tokenizer = lt.tokenizer.Tokenizer.from_pretrained("nano-llama")
model = lt.model.LlamaModel.from_pretrained("nano-llama")
inputs = tokenizer.encode("Sherlock Holmes")
output = model.generate(inputs, eos_id=tokenizer.eos_token_id, max_new_tokens=50)
tokenizer.decode(output)
```
More tutorial can be found [here](docs/tutorial/tutorial.md)
## Available Model Architectures to train

| Architecture | Source |
|--------------------|--------------------------------------------|
| GPT                | [OpenAI GPT](https://openai.com/index/language-unsupervised/) |
| LLaMA              | [Meta LLaMA](https://arxiv.org/abs/2302.13971) |
| BERT               | [Google BERT](https://arxiv.org/abs/1810.04805) |
| VIT                | [Vision Transformer](https://arxiv.org/abs/2010.11929) |
