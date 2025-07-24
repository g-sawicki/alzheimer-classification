import numpy as np
import pytest
from torchvision.datasets import ImageFolder

from alzheimer_classification.data.data_utils import stratified_split


class MockImageFolder(ImageFolder):
    """Mock ImageFolder for testing with controlled class distribution"""

    def __init__(self, class_distribution: list[int]) -> None:  # noqa
        self.samples = [
            (f"image_{i}.jpg", label)
            for i, label in enumerate(
                np.repeat(range(len(class_distribution)), class_distribution)
            )
        ]
        self.targets = [label for _, label in self.samples]


@pytest.fixture
def mock_dataset() -> MockImageFolder:
    return MockImageFolder([40, 30, 20, 10])


def test_stratification_preserved(mock_dataset):
    """Test that class ratios are preserved in all splits"""
    train, val, test = stratified_split(mock_dataset)

    def get_class_counts(subset):
        targets = [subset.dataset.targets[i] for i in subset.indices]
        return np.bincount(targets)

    train_counts = get_class_counts(train)
    val_counts = get_class_counts(val)
    test_counts = get_class_counts(test)

    assert np.allclose(
        train_counts / train_counts.sum(), [0.4, 0.3, 0.2, 0.1], atol=0.05
    )
    assert np.allclose(val_counts / val_counts.sum(), [0.4, 0.3, 0.2, 0.1], atol=0.05)
    assert np.allclose(test_counts / test_counts.sum(), [0.4, 0.3, 0.2, 0.1], atol=0.05)


def test_no_data_leakage(mock_dataset):
    """Test that there's no overlap between train/val/test indices"""
    train, val, test = stratified_split(mock_dataset)

    train_indices = set(train.indices)
    val_indices = set(val.indices)
    test_indices = set(test.indices)

    # Verify no overlap between any sets
    assert train_indices & val_indices == set()
    assert train_indices & test_indices == set()
    assert val_indices & test_indices == set()

    # Verify all original samples are accounted for
    all_indices = train_indices | val_indices | test_indices
    assert len(all_indices) == len(mock_dataset.targets)


def test_split_ratios_correct(mock_dataset):
    """Test that the splits have the correct sizes (64%/16%/20%)"""
    train, val, test = stratified_split(mock_dataset)
    total = len(mock_dataset.targets)

    # Expected ratios: 0.8*0.8=64% train, 0.8*0.2=16% val, 20% test
    assert len(train) == pytest.approx(0.64 * total, abs=1)
    assert len(val) == pytest.approx(0.16 * total, abs=1)
    assert len(test) == pytest.approx(0.2 * total, abs=1)


def test_deterministic_with_random_state(mock_dataset):
    """Test that splits are consistent with the same random state"""
    train1, val1, test1 = stratified_split(mock_dataset)
    train2, val2, test2 = stratified_split(mock_dataset)

    assert set(train1.indices) == set(train2.indices)
    assert set(val1.indices) == set(val2.indices)
    assert set(test1.indices) == set(test2.indices)
