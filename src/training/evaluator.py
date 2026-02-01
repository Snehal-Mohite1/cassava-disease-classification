import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


class Evaluator:
    """
    Evaluation engine for trained CNN models.
    Computes accuracy, confusion matrix, and classification report.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.model.to(self.device)

    def evaluate(self):
        """
        Runs evaluation on the dataset.
        Returns accuracy, confusion matrix, and classification report.
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        return accuracy, conf_matrix, report
