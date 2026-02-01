import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Training engine for CNN models.
    Handles training and validation loops.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.
        Returns average training loss.
        """
        self.model.train()
        running_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_one_epoch(self) -> float:
        """
        Evaluates the model on validation data.
        Returns average validation loss.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)
