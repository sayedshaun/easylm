import os
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
    


class Tokenizer(ABC):
    """
    Base class for tokenizers.
    """
    mask_token: str = "<|mask|>"
    cls_token: str = "<|cls|>"
    sep_token: str = "<|sep|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    sos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"
    
    
    def encode(self, text: str) -> list:
        """
        Encode the text into tokens.
        """
        raise NotImplementedError("Encode method not implemented.")
    
    def decode(self, tokens: list) -> str:
        """
        Decode the tokens back into text.
        """
        raise NotImplementedError("Decode method not implemented.")
    
    def __len__(self) -> int:
        """
        Return the size of the vocabulary.
        """
        raise NotImplementedError("Length method not implemented.")
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.
        """
        raise NotImplementedError("Save method not implemented.")
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "Tokenizer":
        """
        Load a pretrained tokenizer.
        """
        raise NotImplementedError("From pretrained method not implemented.")