import os
import random
import re
import torch
from abc import ABC
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from platformdirs import user_cache_dir
from src.tokenizer import Tokenizer
from typing import List, Optional, Tuple, Generator, Union
from torch.nn.utils.rnn import pad_sequence


random.seed(42)
torch.manual_seed(42)


class Document:
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
                all_text += self.load_data(full_path) + "\n"  # Using a single newline as a separator.
        return all_text

    def load_data(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Collapse multiple newlines to a single newline.
            text = self.collapse_newlines(text)
        return text.strip()


class CausalDataset(torch.utils.data.Dataset, Document):
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, n_ctx: int) -> None:
        self.n_ctx = n_ctx
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data)

    def __len__(self) -> int:
        return len(self.data) - self.n_ctx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load only the required part into memory
        block = self.data[idx: idx + self.n_ctx + 1]
        input_ids, target_ids = block[:-1], block[1:]
        return (
            torch.tensor(input_ids, dtype=torch.long), 
            torch.tensor(target_ids, dtype=torch.long)
            )
    

class IterableDocument:
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

class StreamingCausalDataset(torch.utils.data.IterableDataset, IterableDocument):
    def __init__(self, dir_or_path: str, tokenizer, n_ctx: int, stride: int = 1) -> None:
        self.dir_or_path = dir_or_path
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.stride = stride

    def __iter__(self):
        files, text_generator = self.load(self.dir_or_path)
        # If using multiple workers, split the files among them
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            all_files = files
            # Subsample files based on worker ID and total number of workers
            all_files = all_files[worker_info.id::worker_info.num_workers]
            text_generator = (open(file, "r", encoding="utf-8").read() for file in all_files)

        for text in text_generator:
            token_ids = self.tokenizer.encode(text)
            n_tokens = len(token_ids)
            # Use a sliding window to generate examples
            for i in range(0, n_tokens - self.n_ctx, self.stride):
                block = token_ids[i: i + self.n_ctx + 1]
                input_ids, target_ids = block[:-1], block[1:]
                yield torch.tensor(input_ids), torch.tensor(target_ids)



class IterableCausalDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset for causal language modeling.
    """
    def __init__(self, directory: str, tokenizer, n_ctx: int, batch: int) -> None:
        self.directory = directory
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.batch = batch

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        # Iterate over grouped windows of sentences
        for sentences in IterableCausalDataset.batch_stream(self.directory, self.batch):
            text = " ".join(sentences)
            ids = self.tokenizer.encode(text)

            # Slide over token IDs in non-overlapping blocks
            for i in range(0, len(ids) - 1, self.n_ctx):
                block = ids[i : i + self.n_ctx + 1]
                if len(block) < 2:
                    continue
                input_ids = block[:-1]
                target_ids = block[1:]
                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )

    @staticmethod
    def sentence_stream(directory: str) -> Generator[str, None, None]:
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith('.txt'):
                continue
            path = os.path.join(directory, fn)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s:
                        yield s

    @staticmethod
    def batch_stream(directory: str, num_sentences: int) -> Generator[List[str], None, None]:
        sentences: List[str] = []
        for sentence in IterableCausalDataset.sentence_stream(directory):
            sentences.append(sentence)
            if len(sentences) == num_sentences:
                yield sentences
                sentences = []
        if sentences:
            yield sentences


class MaskedDataset(torch.utils.data.Dataset, Document):
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, n_ctx: int, mask_prob: float = 0.15) -> None:
        self.mask_prob = mask_prob
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data).tolist()[0]

    def __len__(self):
        # We subtract 2 because we will add [CLS] and [SEP] tokens.
        return len(self.data) - (self.n_ctx - 2)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.n_ctx - 2]
        input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
        target_ids = [-100] * len(input_ids)

        for i in range(1, len(input_ids) - 1):
            if random.random() < self.mask_prob:
                target_ids[i] = input_ids[i]
                rand = random.random()
                if rand < 0.8:
                    # 80% of the time, replace with [MASK]
                    input_ids[i] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    # 10% of the time, replace with a random token from the vocabulary.
                    input_ids[i] = random.choice(range(self.tokenizer.vocab_size))
                else:
                    # 10% of the time, leave the token unchanged.
                    pass

        return torch.tensor(input_ids), torch.tensor(target_ids)
    

class ImageClassificationDataset(torch.utils.data.Dataset):
    """
    ### Args:
        image_dir (str): Path to the dataset directory.
        resize_image (int): Target image size (default: 224).
        normalize (bool): Apply ImageNet normalization (default: False).

    ### Structure:
    ```
        root_dir/
            class_0/
                image_0.jpg
                image_1.jpg
                ...
            class_1/
                image_0.jpg
                image_1.jpg
                ...
            class_2/
                image_0.jpg
                image_1.jpg
            ...
    ```
    ### Example:
    ```python
    from src.data import ImageClassificationDataset
    from torchvision import transforms

    custom_transform = [
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomInvert(p=0.5),
        transforms.RandomPosterize(bits=4, p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomSolarize(threshold=192, p=0.5)
    ]
    dataset = ImageClassificationDataset(
        root_dir="/path/to/dataset", 
        resize_image=224, 
        normalize=True,
        add_custom_transform=custom_transform
    )
    ```
    """
    def __init__(
            self, 
            root_dir: str, 
            resize_image: int, 
            normalize: bool = False, 
            add_custom_transform: Union[List[object], None] = None) -> None:
        
        self.image_dir = root_dir
        self.resize_image = resize_image
        self.normalize = normalize
        self.add_custom_transform = add_custom_transform
        
        classes = sorted(os.listdir(root_dir))
        self.class_to_label = {cls: idx for idx, cls in enumerate(classes)}

        self.images, self.labels = [], []
        for cls_name, cls_idx in self.class_to_label.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(cls_idx)

    def label_to_class(self, label: Optional[torch.Tensor]):
        """Convert label index to class name."""
        for cls_name, cls_idx in self.class_to_label.items():
            if cls_idx == label:
                return cls_name

    def transform_image(self, image: Image.Image) -> torch.FloatTensor:
        """Applies image transformations."""
        transform_list = [
            transforms.Resize((self.resize_image, self.resize_image)),
            transforms.ToTensor(),
        ]
        if self.add_custom_transform:
            transform_list += self.add_custom_transform
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )

        transform = transforms.Compose(transform_list)
        return transform(image)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.Tensor]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform_image(image)
        return image, torch.tensor(label, dtype=torch.long)
