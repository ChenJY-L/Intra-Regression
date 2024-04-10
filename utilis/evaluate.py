import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


def evaluate_model(m, test_loader):
    model = m.cpu()
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        for data in test_loader:
            x, y = data
            y_pred = model(x)
            all_predictions.append(y_pred.numpy())
            all_targets.append(y.numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        return all_predictions, all_targets


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.show()


def plot_predictions(predictions, targets):
    # 绘制散点图
    plt.scatter(targets, predictions, alpha=0.5)

    # 添加对角线
    plt.plot(targets, targets, color='red', linestyle='--')

    # 计算 R^2
    r2 = r2_score(targets, predictions)

    # 计算 RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # 添加标题和标签
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('True vs Predicted values\nRMSE: {:.2f}, R^2: {:.2f}'.format(rmse, r2))

    # 显示图例
    plt.legend(['Diagonal', 'Data Points'])

    # 显示图形
    plt.show()


def plot_residuals(predictions, targets):
    residuals = targets - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted values')
    plt.show()
