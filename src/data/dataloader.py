import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.

    Each subfolder inside root_dir represents a class.
    Example:
        root_dir/
            healthy/
            disease_1/
            disease_2/
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        """
        Reads directory structure and maps class names to labels.
        """
        class_names = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns one image and its label.
        """
        img_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(
    data_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Creates DataLoader for training or validation.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CassavaDataset(
        root_dir=data_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader
