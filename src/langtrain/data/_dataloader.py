import random
import torch
from typing import List, Optional, Any, Union, Callable, Iterable, Generator

class DataLoader:
    def __init__(
        self, 
        dataset: Union[torch.utils.data.Dataset, Iterable], 
        batch_size: int = 1, 
        shuffle: bool = True, 
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if collate_fn is None:
            self.collate_fn = self.default_collate
        else:
            self.collate_fn = collate_fn

    def default_collate(self, batch: List[Any]) -> Any:
        transposed = list(zip(*batch))
        return [torch.stack(samples) for samples in transposed]

    def __iter__(self) -> Generator[Any, None, None]:
        # Create a list of indices for the dataset.
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        # Yield the last batch if it has any remaining samples.
        if batch:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        # Returns the total number of batches.
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size