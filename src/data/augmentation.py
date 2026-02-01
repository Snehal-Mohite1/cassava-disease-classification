from torchvision import transforms


def get_train_transforms():
    """
    Augmentations used ONLY during training.
    These help the model generalize better.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()
    ])


def get_val_transforms():
    """
    Transformations for validation/testing.
    No randomness here.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
