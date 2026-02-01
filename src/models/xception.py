import torch
import torch.nn as nn
import torchvision.models as models


class XceptionNet(nn.Module):
    """
    Xception-based image classification model using transfer learning.

    Architecture:
    - Pretrained CNN backbone
    - Custom fully connected classifier head
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(XceptionNet, self).__init__()

        # Torchvision does not provide native Xception,
        # so we use Inception-style features as a strong alternative
        self.backbone = models.inception_v3(
            pretrained=pretrained,
            aux_logits=False
        )

        # Remove original classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
