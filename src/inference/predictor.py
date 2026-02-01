import torch
from torchvision import transforms
from PIL import Image
from typing import List


class Predictor:
    """
    Runs inference on a single image using a trained model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: torch.device
    ):
        self.model = model
        self.class_names = class_names
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path: str) -> str:
        """
        Predicts class label for a single image.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)  # add batch dimension

        with torch.no_grad():
            outputs = self.model(image)
            _, pred = torch.max(outputs, 1)

        return self.class_names[pred.item()]
