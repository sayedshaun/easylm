import os
import shutil
from typing import List, Optional, Union
import warnings
import numpy as np
from platformdirs import user_cache_dir
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import torch
from abc import ABC


class TextLoader:
    def load(self, dir_or_path: str) -> str:
        if os.path.isdir(dir_or_path):
            return self.load_data_from_dir(dir_or_path)
        else:
            return self.load_data(dir_or_path)

    @staticmethod
    def load_data_from_dir(dir_path: str) -> str:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        txt_files = [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file.endswith(".txt")
        ]
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {dir_path}")
        return ",".join(txt_files)

    @staticmethod
    def load_data(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path


class Tokenizer(TextLoader):
    mask_token: str = "<|mask|>"
    cls_token: str = "<|cls|>" 
    sep_token: str = "<|sep|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    sos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"

    def __init__(
        self,
        dir_or_path: str,
        vocab_size: int,
        retrain: bool = False,
        model_type: str = "bpe",
        add_special_tokens: Optional[List[str]] = None) -> None:
        super().__init__()
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
            Tokenizer.mask_token,
            Tokenizer.cls_token,
            Tokenizer.sep_token,
            Tokenizer.pad_token,
            Tokenizer.unk_token,
            Tokenizer.sos_token,
            Tokenizer.eos_token
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
        self.mask_token_id = self.encode(Tokenizer.mask_token)
        self.cls_token_id = self.encode(Tokenizer.cls_token)
        self.sep_token_id = self.encode(Tokenizer.sep_token)
        self.pad_token_id = self.encode(Tokenizer.pad_token)
        self.unk_token_id = self.encode(Tokenizer.unk_token)
        self.sos_token_id = self.encode(Tokenizer.sos_token)
        self.eos_token_id = self.encode(Tokenizer.cls_token) 

    def _train_model(self) -> None:
        spm.SentencePieceTrainer.Train(
            input=self._input_data,
            model_prefix=os.path.join(self._tokenizer_dir, self._model_prefix),
            vocab_size=self.vocab_size,
            user_defined_symbols=self.special_tokens,
            add_dummy_prefix=False,
            model_type="bpe",

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
        shutil.copy(self._model_path, path)
        shutil.copy(self._vocab_path, path)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "Tokenizer":
        model_path = os.path.join(pretrained_path, "VOCAB.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
        
        # Create an uninitialized instance of Tokenizer.
        tokenizer = Tokenizer.__new__(Tokenizer)
        
        # Set the necessary attributes.
        tokenizer._tokenizer_dir = pretrained_path
        tokenizer._model_path = model_path
        tokenizer.vocab_size = None  # or set an appropriate default if needed
        tokenizer.special_tokens = [
            Tokenizer.mask_token,
            Tokenizer.cls_token,
            Tokenizer.sep_token,
            Tokenizer.pad_token,
            Tokenizer.unk_token,
            Tokenizer.sos_token,
            Tokenizer.eos_token
        ]
        
        # Load the pretrained SentencePiece model.
        tokenizer.processor = SentencePieceProcessor(model_file=model_path)
        
        # Setup token ids.
        tokenizer.mask_token_id = tokenizer.encode(Tokenizer.mask_token)
        tokenizer.cls_token_id = tokenizer.encode(Tokenizer.cls_token)
        tokenizer.sep_token_id = tokenizer.encode(Tokenizer.sep_token)
        tokenizer.pad_token_id = tokenizer.encode(Tokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.encode(Tokenizer.unk_token)
        tokenizer.sos_token_id = tokenizer.encode(Tokenizer.sos_token)
        tokenizer.eos_token_id = tokenizer.encode(Tokenizer.eos_token)
        
        return tokenizer



__all__ = ["Tokenizer"]


if __name__ == "__main__":
    tokenizer = Tokenizer(dir_or_path="/home/shaun/Desktop/data", vocab_size=1000, retrain=True)
    print(tokenizer.encode("Hello, world!"))
    tokenizer.save("pretrained_model")