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


class NextWordPredDataset(torch.utils.data.Dataset):  
    """
    ### Args:
        file_path (str): Path to the dataset file.
        max_seq_len (int): Maximum sequence length for each training sample.

    ### Structure:
    ```
    file_path/
        text.txt
    ```
    ### Example:
    ```python
    from src.data import NextWordPredDataset

    dataset = NextWordPredDataset(file_path="/path/to/dataset/text.txt", max_seq_len=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    ``` 
    """
    def __init__(self, file_path: str, tokenizer: Tokenizer, max_seq_len: int) -> None:
        self.n_ctx = max_seq_len  # Context window size
        self.tokenizer = tokenizer
        
        # Read and tokenize the dataset
        with open(file_path, "r", encoding="utf-8") as file:
            self.text = file.read()

        # Convert text to token IDs
        self.all_ids = self.tokenizer.encode(self.text)

    def __len__(self):
        return len(self.all_ids) - self.n_ctx

    def __getitem__(self, idx):
        """Retrieve a training sample by index."""
        input_ids = torch.tensor(self.all_ids[idx: idx + self.n_ctx])
        target_ids = torch.tensor(self.all_ids[idx + 1: idx + self.n_ctx + 1])
        return input_ids.long(), target_ids.long()


class MaskedLMDataset(torch.utils.data.Dataset):  
    def __init__(self, file_path: str, tokenizer: Tokenizer, max_seq_len: int, mask_prob: float = 0.15, mask_token_id: int = 103) -> None:
        """
        mask_token_id: the token ID corresponding to the [MASK] token.
        """
        self.n_ctx = max_seq_len  # Context window size
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        
        # Read and tokenize the dataset
        with open(file_path, "r", encoding="utf-8") as file:
            self.text = file.read()

        # Convert text to token IDs
        self.all_ids = self.tokenizer.encode(self.text)

    def __len__(self):
        return len(self.all_ids) - self.n_ctx

    def __getitem__(self, idx):
        """
        Retrieve a training sample and apply random masking.
        Returns:
            input_ids: the input sequence with some tokens replaced by [MASK]
            labels: the original tokens for masked positions, and -100 (ignore index) elsewhere
        """
        original_ids = self.all_ids[idx: idx + self.n_ctx]
        input_ids = original_ids.copy()
        labels = [-100] * len(original_ids)  # -100 will be ignored in loss computation
        
        # Randomly mask tokens with probability mask_prob
        for i in range(len(original_ids)):
            if random.random() < self.mask_prob:
                # Save original token as label
                labels[i] = original_ids[i]
                # Replace token with [MASK] token id
                input_ids[i] = self.mask_token_id
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels


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
    "NextWordPredDataset",
    "MaskedLMDataset",
    "ImageClassificationDataset",
]