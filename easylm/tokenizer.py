import os
from typing import List, Optional, Union
import warnings
import numpy as np
from platformdirs import user_cache_dir
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import torch
from abc import ABC


class TextLoader(ABC):
    def load(self, dir_or_path: str) -> str:
        if os.path.isdir(dir_or_path):
            return self.load_data_from_dir(dir_or_path)
        else:
            return self.load_data(dir_or_path)

    @staticmethod
    def load_data_from_dir(dir_path: str) -> str:
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
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path


class Tokenizer(TextLoader):
    def __init__(
        self,
        dir_or_path: str,
        vocab_size: int,
        retrain: bool = False,
        add_special_tokens: Optional[List[str]] = None
    ) -> None:
        # Get the input file(s) for training.
        self._input_data = self.load(dir_or_path)
        self._tokenizer_dir = os.path.join(user_cache_dir("easylm"), "tokenizer")
        self._model_prefix = "VOCAB"
        self.vocab_size = vocab_size
        self._model_path = os.path.join(self._tokenizer_dir, f"{self._model_prefix}.model")

        # Set default special tokens.
        self.special_tokens = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[SOS]", "[EOS]"]
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
        self.mask_token_id = self.encode("[MASK]")
        self.cls_token_id = self.encode("[CLS]")
        self.sep_token_id = self.encode("[SEP]")
        self.pad_token_id = self.encode("[PAD]")
        self.unk_token_id = self.encode("[UNK]")
        self.sos_token_id = self.encode("[SOS]")
        self.eos_token_id = self.encode("[EOS]") 

    def _train_model(self) -> None:
        spm.SentencePieceTrainer.Train(
            input=self._input_data,
            model_prefix=os.path.join(self._tokenizer_dir, self._model_prefix),
            vocab_size=self.vocab_size,
            user_defined_symbols=self.special_tokens,
            add_dummy_prefix=False
        )

    def encode(self, text: str, return_tensors: str = "pt") -> Union[List[int], torch.Tensor, np.ndarray]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        # If text is somehow a tensor, convert it to a list.
        if isinstance(text, torch.Tensor):
            text = text.tolist()
        encoded = self.processor.encode(text, out_type=int)
        if return_tensors == "np":
            return np.array(encoded)
        elif return_tensors == "list":
            return encoded
        elif return_tensors == "pt":
            return torch.tensor(encoded)
        else:
            raise ValueError("return_tensors must be one of ['np', 'list', 'pt'].")

    def decode(self, tokens: Optional[Union[List[int], torch.Tensor]]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.processor.decode(tokens)

    def __len__(self) -> int:
        return self.processor.get_piece_size()


__all__ = ["Tokenizer"]


if __name__ == "__main__":
    tokenizer = Tokenizer(dir_or_path="data", vocab_size=1000)
    print(tokenizer.encode("Hello, world!"))