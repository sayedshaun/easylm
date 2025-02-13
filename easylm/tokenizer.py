import os
from typing import List, Optional, Union
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
        self.data_dir = data_dir
        self.tokenizer_dir = "easylm/.cache"
        self.model_prefix = "VOCAB"
        self.vocab_size = vocab_size
        self.add_special_tokens = add_special_tokens
        self.model_path = os.path.join(self.tokenizer_dir, f"{self.model_prefix}.model")

        # Ensure the model directory exists.
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            self.train_vocab_model()

        if force_retrain:
            self.train_vocab_model()

        # Load and cache the SentencePiece model.
        self.processor = SentencePieceProcessor(model_file=self.model_path)

    def train_vocab_model(self) -> None:
        spm.SentencePieceTrainer.Train(
            input=self.data_dir,
            model_prefix=os.path.join(self.tokenizer_dir, self.model_prefix),
            vocab_size=self.vocab_size,
            user_defined_symbols=self.add_special_tokens
        )

    def encode(self, text: str) -> List[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, tokens: Optional[List[int]]) -> str:
        return self.processor.decode(tokens) if tokens else ""

    def __len__(self) -> int:
        return self.processor.get_piece_size()
    

__all__ = ["Tokenizer"]
