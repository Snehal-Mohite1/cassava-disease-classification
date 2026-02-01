from sklearn.model_selection import KFold
from torch.utils.data import Subset
from typing import List, Tuple


def create_kfold_splits(dataset, num_folds: int = 5) -> List[Tuple[Subset, Subset]]:
    """
    Creates K-Fold train/validation splits for a dataset.

    Returns:
        List of (train_subset, val_subset)
    """
    kfold = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42
    )

    splits = []

    for train_indices, val_indices in kfold.split(range(len(dataset))):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        splits.append((train_subset, val_subset))

    return splits
