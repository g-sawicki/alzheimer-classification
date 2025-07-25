import os

import kagglehub
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from alzheimer_classification.data.data_utils import stratified_split
from alzheimer_classification.data.dataset_from_subset import DatasetFromSubset


class AlzheimerDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling data loading for an Alzheimer's disease classification task.

    Args:
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 64.
        train_transform (transforms.Compose, optional): Transformations to apply to the training data.
        test_transform (transforms.Compose, optional): Transformations to apply to the validation/test data.
    """

    def __init__(
        self,
        batch_size: int,
        train_transform: transforms.Compose | None = None,
        test_transform: transforms.Compose | None = None,
    ):
        super().__init__()
        self.root_dir: str = ""
        self.batch_size = batch_size

        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        self.root_dir = kagglehub.dataset_download(
            "uraninjo/augmented-alzheimer-mri-dataset"
        )

    def setup(self, stage: str) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str): Stage of setup (fit or test).
        """
        augmented_dir = os.path.join(self.root_dir, "AugmentedAlzheimerDataset")
        image_folder = datasets.ImageFolder(augmented_dir)

        train_subset, val_subset, test_subset = stratified_split(image_folder)

        self.train_dataset = DatasetFromSubset(train_subset, self.train_transform)
        self.val_dataset = DatasetFromSubset(val_subset, self.test_transform)
        self.test_dataset = DatasetFromSubset(test_subset, self.test_transform)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
