import os
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from langtrain.tokenizer.base import Tokenizer
from typing import Iterable, List, Tuple, Generator



class DocumentLoader:
    def load(self, dir_or_path: str) -> str:
        if os.path.isdir(dir_or_path):
            return self.load_data_from_dir(dir_or_path)
        else:
            return self.load_data(dir_or_path)
            
    def collapse_newlines(self, text: str) -> str:
        # Replace two or more newline with a single newline.
        return re.sub(r'\n{2,}', '\n', text)

    def load_data_from_dir(self, dir_path: str) -> str:
        all_text = ""
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                full_path = os.path.join(dir_path, file)
                all_text += self.load_data(full_path) + "\n"
        return all_text

    def load_data(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Collapse multiple newlines to a single newline.
            text = self.collapse_newlines(text)
        return text.strip()
    

class IterableDocumentLoader:
    def stream_data_from_dir(self, dir_path: str) -> Generator[str, None, None]:
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                a_path = os.path.join(dir_path, file)
                yield from self.stream_data(a_path)

    def stream_data(self, file_path: str) -> Generator[str, None, None]:
        with open(file_path, "r", encoding="utf-8") as file:
            a_file = file.read().strip()
            yield self.collapse_newlines(a_file)

    def collapse_newlines(self, text: str) -> str:
        # Replace two or more newline with a single newline.
        return re.sub(r'\n{2,}', '\n', text)
    
    def load(self, dir_or_path: str) -> Tuple[List[str], Generator[str, None, None]]:
            if os.path.isdir(dir_or_path):
                files = sorted(os.listdir(dir_or_path))
                text_generator = self.stream_data_from_dir(dir_or_path)
            else:
                files = [dir_or_path]
                text_generator = self.stream_data(dir_or_path)
            return files, text_generator


class TokenDumpper:
    def __init__(self, file_path: str, tokenizer: Tokenizer, output_file: str, chunksize: int = 10000) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.output_file = output_file
        self.chunksize = chunksize

    def load_data_from_dir(self, dir_path: str) -> Iterable[str]:
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                a_file_path = os.path.join(dir_path, file)
                yield from self.load_file(a_file_path)
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                a_file_path = os.path.join(dir_path, file)
                yield from self.load_file(a_file_path)

    def load_file(self, file_path: str) -> Iterable[str]:
        with open(file_path, "r", encoding="utf-8") as f, Pool(processes=6) as pool:
            for tokens in pool.imap(self.tokenizer.encode, f, self.chunksize):
                yield tokens

    def tokenize_and_dump(self, data_generator: Iterable[Tuple[np.ndarray, np.ndarray]]) -> None:
        with open(self.output_file, "wb") as out_file:
            for idx, tokens in tqdm(
                enumerate(data_generator), 
                desc="Tokenizing and Dumping"
                ):
                if tokens:
                    array = np.array(tokens, dtype=np.uint16)
                    out_file.write(array.tobytes())
                    out_file.flush()

    def load(self, dir_or_path: str) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if os.path.isdir(dir_or_path):
            return self.load_data_from_dir(dir_or_path)
        elif os.path.isfile(dir_or_path):
            return self.load_file(dir_or_path)
        else:
            raise ValueError(f"Invalid file path: {dir_or_path}")

    def start(self) -> None:
        data_generator = self.load(self.file_path)
        self.tokenize_and_dump(data_generator)