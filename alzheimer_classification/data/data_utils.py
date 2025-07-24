import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


def stratified_split(image_folder: ImageFolder) -> tuple[Subset, Subset, Subset]:
    targets = [label for _, label in image_folder.samples]

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(splitter1.split(np.zeros(len(targets)), targets))

    train_val_targets = [targets[i] for i in train_val_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(
        splitter2.split(np.zeros(len(train_val_targets)), train_val_targets)
    )

    train_subset = Subset(image_folder, [train_val_idx[i] for i in train_idx])
    val_subset = Subset(image_folder, [train_val_idx[i] for i in val_idx])
    test_subset = Subset(image_folder, test_idx)

    return train_subset, val_subset, test_subset
