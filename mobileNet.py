from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import dataset


class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomMobileNet, self).__init__()

        # Load MobileNetV3 Small pretrained model
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Extract just the feature extraction layers
        self.features = backbone.features

        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False

        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3 Small has 576 output features (not 1280 like Large)
        in_features = 576

        # Custom classifier - simpler than Large version (no dropout in original)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # Pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)
        return x

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 10,
    device: torch.device = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = correct / total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        # Print epoch results
        print(f'\nEpoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | '
              f'Val Acc: {val_epoch_acc:.4f}')

    return history

if __name__ == '__main__':
    train_path = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/animals10/train"
    test_path = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/animals10/test"

    train_data_loader = dataset.loader(train_path)
    test_data_loader = dataset.test_loader(test_path)

    model = CustomMobileNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    history = train_model(
        model=model,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
    )

    current_time = datetime.now().strftime("%d_%B_%H:%M")
    torch.save(model.state_dict(), f"mobile_net_{current_time}.pth")

