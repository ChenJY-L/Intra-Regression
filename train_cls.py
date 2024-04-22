# Encoding utf-8
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from config import ClsConfig
from classification_models import *
from evaluate import *
from datasets import *


def train_loop(model, loss_fn, train_loader, test_loader, optimizer,
               scheduler=None, epochs=10, device=torch.device("cpu")):
    """
    Train loop for a PyTorch model.

    Args:
        model: The PyTorch model to train.
        loss_fn: The loss function.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test/validation data.
        optimizer: The optimizer for updating model parameters.
        scheduler (optional): Learning rate scheduler.
        epochs (int): Number of epochs to train the model.
        device: Device for train model.
    """
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Calculate training accuracy
        train_accuracy = 100. * correct / total

        # Validation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Calculate validation accuracy
        test_accuracy = 100. * correct / total

        if scheduler is not None:
            scheduler.step()

        # Print training progress
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%,"
            f" Test Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%")


def train():
    cfg = ClsConfig()

    """ 1. 创建数据集 """
    train_df, test_df = read_dataset(cfg.train_data_path, cfg.test_data_path)

    # 数据归一化
    x = cfg.x_index
    y = cfg.y_index
    x_scaler = MinMaxScaler()
    train_df.iloc[:, x:] = x_scaler.fit_transform(train_df.iloc[:, x:])
    test_df.iloc[:, x:] = x_scaler.transform(test_df.iloc[:, x:])

    y_scaler = MinMaxScaler()
    train_df.iloc[:, y] = y_scaler.fit_transform(train_df.iloc[:, cfg.y_index].values.reshape(-1, 1))
    test_df.iloc[:, y] = y_scaler.transform(test_df.iloc[:, cfg.y_index].values.reshape(-1, 1))

    train_data = ClsIntraDataset(train_df)
    test_data = ClsIntraDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    """ 2. 创建模型 """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLSRes(cfg.num_classes, cfg.feature_size, cfg.hidden_size)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    current_optim = "RMS"
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg.num_epochs)

    """ 3. 模型训练 """
    train_loop(model,
               loss_fn,
               train_loader,
               test_loader,
               optimizer,
               scheduler,
               cfg.num_epochs,
               device)


if __name__ == "__main__":
    train()
