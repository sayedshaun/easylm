import os
import random
import torch
from abc import ABC
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from easylm.tokenizer import Tokenizer
from typing import List, Optional, Tuple, Generator, Union

random.seed(42)
torch.manual_seed(42)


class Document(ABC):
    def load(self, dir_or_path: str) -> list:
        if os.path.isdir(dir_or_path):
            return Document.load_data_from_dir(dir_or_path)
        else:
            return Document.load_data(dir_or_path)
            
    @staticmethod
    def load_data_from_dir(dir_path: str) -> list:
        all_text = ""
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                full_path = os.path.join(dir_path, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n\n"  # Adding a newline as a separator
        return all_text

    @staticmethod
    def load_data(file_path: str) -> list:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    

class IterableDocument(ABC):
    @staticmethod
    def stream_data_from_dir(dir_path: str):
        """Generator that yields text from each .txt file in a directory."""
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                full_path = os.path.join(dir_path, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    yield f.read()

    @staticmethod
    def stream_data(file_path: str):
        """Generator that yields text from a single file."""
        with open(file_path, "r", encoding="utf-8") as file:
            yield file.read()


class CausalLMDataset(torch.utils.data.Dataset, Document):
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data).tolist()[0]

    def __len__(self) -> int:
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.data[idx: idx + self.max_seq_len + 1]
        input_ids, target_ids = block[:-1], block[1:]
        return torch.tensor(input_ids), torch.tensor(target_ids)
    


class StreamingCausalLMDataset(torch.utils.data.IterableDataset, IterableDocument):
    def __init__(self, dir_or_path: str, tokenizer, max_seq_len: int, stride: int = 1):
        self.dir_or_path = dir_or_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

    def __iter__(self):
        # Decide whether we're dealing with a directory or a single file
        if os.path.isdir(self.dir_or_path):
            text_generator = IterableDocument.stream_data_from_dir(self.dir_or_path)
            files = sorted(os.listdir(self.dir_or_path))
        else:
            text_generator = IterableDocument.stream_data(self.dir_or_path)
            files = [self.dir_or_path]

        # If using multiple workers, split the files among them
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Get the full list of files for proper splitting
            if os.path.isdir(self.dir_or_path):
                all_files = [os.path.join(self.dir_or_path, file) for file in files if file.endswith(".txt")]
            else:
                all_files = [self.dir_or_path]
            # Subsample files based on worker ID and total number of workers
            all_files = all_files[worker_info.id::worker_info.num_workers]
            text_generator = (open(file, "r", encoding="utf-8").read() for file in all_files)

        # For each text, tokenize and yield sliding windows as (input, target) pairs
        for text in text_generator:
            # Tokenize the text. Adjust as needed if your tokenizer supports streaming.
            token_ids = self.tokenizer.encode(text).tolist()[0]
            n_tokens = len(token_ids)
            # Use a sliding window to generate examples
            for i in range(0, n_tokens - self.max_seq_len, self.stride):
                block = token_ids[i: i + self.max_seq_len + 1]
                input_ids = block[:-1]
                target_ids = block[1:]
                yield torch.tensor(input_ids), torch.tensor(target_ids)



class MaskedLMDataset(torch.utils.data.Dataset, Document):
    def __init__(self, dir_or_path: str, tokenizer: Tokenizer, max_seq_len: int, mask_prob: float = 0.15) -> None:
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        data = self.load(dir_or_path)
        self.data = tokenizer.encode(data).tolist()[0]

    def __len__(self):
        # We subtract 2 because we will add [CLS] and [SEP] tokens.
        return len(self.data) - (self.max_seq_len - 2)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.max_seq_len - 2]
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