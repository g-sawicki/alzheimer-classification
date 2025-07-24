import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class DatasetFromSubset(Dataset):
    """
    A custom Dataset class that applies a transformation to a subset of a dataset.

    Args:
        subset (torch.utils.data.Subset): A subset of a dataset.
        transform (transforms.Compose, optional): A transform applied to images.

    Methods:
        __getitem__(index): Retrieves the item at the specified index, applying the transform if provided.
        __len__(): Returns the length of the subset.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose | None = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)
