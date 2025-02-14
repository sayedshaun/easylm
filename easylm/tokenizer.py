import os
from typing import List, Optional, Union
import warnings
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(
        self,
        data_dir: str,
        vocab_size: int,
        force_retrain: bool = False,
        add_special_tokens: Union[List[str], None] = None
    ) -> None:
        self._data_dir = data_dir
        self._tokenizer_dir = "easylm/.cache"
        self._model_prefix = "VOCAB"
        self.vocab_size = vocab_size
        self._model_path = os.path.join(self._tokenizer_dir, f"{self._model_prefix}.model")

        if add_special_tokens:
            warnings.warn("You must retrain the model after adding special tokens.")    
        
        self.special_tokens = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[SOS]", "[EOS]"]

        if add_special_tokens is not None:
            self.special_tokens.extend(add_special_tokens)
   
        # Ensure the model directory exists.
        os.makedirs(self._tokenizer_dir, exist_ok=True)

        if not os.path.exists(self._model_path):
            self.train_vocab_model()

        if force_retrain:
            self.train_vocab_model()

        # Load and cache the SentencePiece model.
        self.processor = SentencePieceProcessor(model_file=self._model_path)
        self.mask_token_id = self.encode("[MASK]")
        self.cls_token_id = self.encode("[CLS]")
        self.sep_token_id = self.encode("[SEP]")
        self.pad_token_id = self.encode("[PAD]")
        self.unk_token_id = self.encode("[UNK]")
        self.sos_token_id = self.encode("[SOS]")
        self.eos_token_id = self.encode("[EOS]") 

    def train_vocab_model(self) -> None:
        spm.SentencePieceTrainer.Train(
            input=self._data_dir,
            model_prefix=os.path.join(self._tokenizer_dir, self._model_prefix),
            vocab_size=self.vocab_size,
            user_defined_symbols=self.special_tokens,
            add_dummy_prefix=False
        )

    def encode(self, text: str) -> List[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, tokens: Optional[List[int]]) -> str:
        if tokens is None:
            raise ValueError("Tokens cannot be None.")
        return self.processor.decode(tokens)

    def __len__(self) -> int:
        return self.processor.get_piece_size()
    

__all__ = ["Tokenizer"]
