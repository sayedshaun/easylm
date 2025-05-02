import langtrain as lt
from langtrain.data import (
    SimpleDataloader, 
    SimpleCausalDataset, 
    LazyCausalDataset, 
    IterableCausalDataset, 
    ImageClassificationDataset, 
    MaskedDataset
)
from langtrain.trainer import Trainer
from langtrain.tokenizer import SentencePieceTokenizer
from langtrain.model import LlamaModel, GPTModel, BertModel, VITImageClassifier
from langtrain.config import LlamaConfig, GPTConfig, BertConfig, VITConfig, TrainingConfig


def test_import():
    assert lt
    assert lt.__version__
    assert LlamaModel
    assert GPTModel
    assert BertModel
    assert SentencePieceTokenizer
    assert Trainer
    assert SimpleDataloader
    assert SimpleCausalDataset
    assert LazyCausalDataset
    assert IterableCausalDataset
    assert ImageClassificationDataset
    assert MaskedDataset
    assert LlamaConfig
    assert GPTConfig
    assert BertConfig
    assert VITConfig
    assert TrainingConfig
    assert VITImageClassifier

if __name__ == "__main__":
    test_import()
    print("All imports are working correctly!")