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
        ...
    )
)
train_config=lt.config.TrainingConfig(
    train_data=dataset,
    ...
)
trainer = lt.trainer.Trainer(
    model=model,
    tokenizer=tokenizer,
    model_name="nano-llama",
    collate_fn=lt.utils.collate_fn,
    config=train_config
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
