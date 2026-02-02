import torch
import torch.optim as optim

from models.xception import XceptionNet
from data.dataloader import get_dataloader
from training.trainer import Trainer
from training.evaluator import Evaluator


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 5          # change based on dataset
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 3

    # Dataset path
    data_dir = "data/raw"

    # Model
    model = XceptionNet(num_classes=num_classes, pretrained=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloaders
    train_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )

    # Training loop
    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch()
        val_loss = trainer.validate_one_epoch()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    # Evaluation
    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        device=device
    )

    accuracy, conf_matrix, report = evaluator.evaluate()

    print("\nFinal Results")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    main()
