import os
import yaml
import shutil
import warnings
import torch
import numpy as np
import sentencepiece as spm
from typing import List, Optional, Union
from platformdirs import user_cache_dir
from sentencepiece import SentencePieceProcessor
from langtrain.tokenizer.base import TextLoader, Tokenizer



class SentencePieceTokenizer(TextLoader, Tokenizer):
    """
    A tokenizer that uses SentencePiece in backend for tokenization. 

    Args:
        dir_or_path (str): Directory or path to the .txt training data.
        vocab_size (int): Size of the vocabulary.
        retrain (bool): Whether to retrain the model.
        model_type (str): Type of SentencePiece model. Default is "bpe".
        add_special_tokens (List[str]): List of special tokens to add.

    Example:
    ```python
    from langtrain.tokenizer import SentencePieceTokenizer
    tokenizer = SentencePieceTokenizer(
        dir_or_path="data_directory"
        vocab_size=32000,
        retrain=True,
        model_type="bpe",
    )

    # From pretrained
    tokenizer = SentencePieceTokenizer.from_pretrained("path_to_pretrained_model")
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    ```
    """

    def __init__(
        self,
        dir_or_path: str,
        vocab_size: int,
        retrain: bool = False,
        model_type: str = "bpe",
        add_special_tokens: Optional[List[str]] = None) -> None:
        super(SentencePieceTokenizer, self).__init__()
        self.model_type = model_type
        self.retrain = retrain
        self.add_special_tokens = add_special_tokens if add_special_tokens else []

        
        # Get the input file(s) for training.
        self._input_data = self.load(dir_or_path)
        self._tokenizer_dir = os.path.join(user_cache_dir("easylm"), "tokenizer")
        self._model_prefix = "VOCAB"
        self.vocab_size = vocab_size
        self._model_path = os.path.join(self._tokenizer_dir, f"{self._model_prefix}.model")
        self._vocab_path = os.path.join(self._tokenizer_dir, f"{self._model_prefix}.vocab")

        # Set default special tokens.
        self.special_tokens = [
            SentencePieceTokenizer.mask_token,
            SentencePieceTokenizer.cls_token,
            SentencePieceTokenizer.sep_token,
            SentencePieceTokenizer.pad_token,
            SentencePieceTokenizer.unk_token,
            SentencePieceTokenizer.sos_token,
            SentencePieceTokenizer.eos_token
        ]
        if add_special_tokens:
            warnings.warn("You must retrain the model after adding special tokens.")
            self.special_tokens.extend(add_special_tokens)
       
        # Ensure the model directory exists.
        os.makedirs(self._tokenizer_dir, exist_ok=True)
        if not os.path.exists(self._model_path):
            self._train_model()

        if retrain:
            self._train_model()

        # Load the SentencePiece model.
        self.processor = SentencePieceProcessor(model_file=self._model_path)
        self.mask_token_id = self.processor.piece_to_id(SentencePieceTokenizer.mask_token)
        self.cls_token_id = self.processor.piece_to_id(SentencePieceTokenizer.cls_token)
        self.sep_token_id = self.processor.piece_to_id(SentencePieceTokenizer.sep_token)
        self.pad_token_id = self.processor.piece_to_id(SentencePieceTokenizer.pad_token)
        self.unk_token_id = self.processor.piece_to_id(SentencePieceTokenizer.unk_token)
        self.sos_token_id = self.processor.piece_to_id(SentencePieceTokenizer.sos_token)
        self.eos_token_id = self.processor.piece_to_id(SentencePieceTokenizer.cls_token) 

    def _train_model(self) -> None:
        spm.SentencePieceTrainer.Train(
            input=self._input_data,
            model_prefix=os.path.join(self._tokenizer_dir, self._model_prefix),
            vocab_size=self.vocab_size,
            user_defined_symbols=self.special_tokens,
            add_dummy_prefix=False,
            model_type=self.model_type,
        )

    def encode(self, text: str) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        encoded = self.processor.encode(text, out_type=int)
        return encoded

    def decode(self, tokens: Union[Union[List[int], torch.Tensor, np.ndarray]]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        decoded = self.processor.decode(tokens)
        return "".join(decoded)

    def __len__(self) -> int:
        return self.processor.get_piece_size()
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_config.yaml"), "w") as f:
            yaml.dump(
                {
                    "name": str(self.__class__.__name__),
                    "model_type": self.model_type,
                    "vocab_size": self.vocab_size,
                    "sos_token": SentencePieceTokenizer.sos_token,
                    "eos_token": SentencePieceTokenizer.eos_token,
                    "pad_token": SentencePieceTokenizer.pad_token,
                    "unk_token": SentencePieceTokenizer.unk_token,
                    "sep_token": SentencePieceTokenizer.sep_token,
                    "cls_token": SentencePieceTokenizer.cls_token,
                    "mask_token": SentencePieceTokenizer.mask_token,
                }, f
            )
        shutil.copy(self._model_path, path)
        shutil.copy(self._vocab_path, path)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "SentencePieceTokenizer":
        model_path = os.path.join(pretrained_path, "VOCAB.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
        
        # Create an uninitialized instance of Tokenizer.
        tokenizer = SentencePieceTokenizer.__new__(SentencePieceTokenizer)
        
        # Set the necessary attributes.
        tokenizer._tokenizer_dir = pretrained_path
        tokenizer._model_path = model_path
        tokenizer.vocab_size = None  # or set an appropriate default if needed
        tokenizer.special_tokens = [
            SentencePieceTokenizer.mask_token,
            SentencePieceTokenizer.cls_token,
            SentencePieceTokenizer.sep_token,
            SentencePieceTokenizer.pad_token,
            SentencePieceTokenizer.unk_token,
            SentencePieceTokenizer.sos_token,
            SentencePieceTokenizer.eos_token
        ]
        
        # Load the pretrained SentencePiece model.
        tokenizer.processor = SentencePieceProcessor(model_file=model_path)
        
        # Setup token ids.
        tokenizer.mask_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.mask_token)
        tokenizer.cls_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.cls_token)
        tokenizer.sep_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.sep_token)
        tokenizer.pad_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.unk_token)
        tokenizer.sos_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.sos_token)
        tokenizer.eos_token_id = tokenizer.processor.piece_to_id(SentencePieceTokenizer.eos_token)
        
        return tokenizer
