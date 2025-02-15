import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data
import torchvision
from easylm.tokenizer import Tokenizer
from typing import List, Optional, Tuple, Generator, Union

random.seed(42)
torch.manual_seed(42)


import torch
from torch.utils.data import Dataset
from typing import Tuple

class NextWordPredictDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.all_ids = self.load_data(file_path, tokenizer)

    @staticmethod
    def load_data(file_path: str, tokenizer) -> list:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return tokenizer.encode(text)

    def __len__(self) -> int:
        # Each sample requires block_size + 1 tokens
        return len(self.all_ids) - self.max_seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get a contiguous block of block_size+1 tokens
        block = self.all_ids[idx: idx + self.max_seq_len + 1]
        # Input: first block_size tokens
        input_ids = block[:-1]
        # Target: next tokens (i.e., the input shifted by one position)
        target_ids = block[1:]
        return (
            torch.tensor(input_ids, dtype=torch.long), 
            torch.tensor(target_ids, dtype=torch.long)
        )


class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, max_seq_len: int, mask_prob: float = 0.15) -> None:
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.all_ids = self.data_generator(file_path, tokenizer)

    @staticmethod
    def data_generator(file_path: str, tokenizer: Tokenizer):    
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        all_ids = tokenizer.encode(text)
        return all_ids

    def __len__(self):
        # We subtract 2 because we will add [CLS] and [SEP] tokens.
        return len(self.all_ids) - (self.max_seq_len - 2)

    def __getitem__(self, idx):
        chunk = self.all_ids[idx: idx + self.max_seq_len - 2]
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
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
    


__all__ = [
    "NextWordPredictDataset",
    "MaskedLMDataset",
    "ImageClassificationDataset",
]


