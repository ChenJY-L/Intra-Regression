import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


def evaluate_model(m, data_loader: DataLoader, y_scaler: MinMaxScaler):
    """
    模型评价函数
    :param m: 回归模型
    :param data_loader: 数据集
    :param y_scaler: y数据归一化方法
    :return: None
    """

    model = m.cpu()
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        for data in data_loader:
            x, y = data
            y_pred = model(x)
            all_predictions.append(y_pred.numpy())
            all_targets.append(y.numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        all_predictions = y_scaler.inverse_transform(all_predictions.reshape(-1, 1))
        all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1))
        return all_predictions, all_targets


def plot_loss_and_lr(train_losses, test_losses, learning_rate):
    plt.plot(learning_rate, label='Learning Rate')
    plot_loss(train_losses, test_losses)


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
    print("R2 {:.2f}, RMSE {:.2f}".format(r2, rmse))

    # 添加标题和标签
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('True vs Predicted values\nRMSE: {:.2f}, R^2: {:.2f}'.format(rmse, r2))

    # 显示图例
    plt.legend(['Diagonal', 'Data Points'])

    # 显示图形
    plt.show(block=False)

    return r2, rmse


def plot_tests(predictions, targets, test_predictions, test_targets):
    plt.scatter(targets, predictions)
    plt.scatter(test_targets, test_predictions, color='green')

    plt.plot(np.append(targets, test_targets), np.append(targets, test_targets), color='red', linestyle='--')

    plt.legend(['Validation points', 'Test points'])
    # 计算 R^2
    r2 = r2_score(test_targets, test_predictions)

    # 计算 RMSE
    rmse = np.sqrt(np.mean((test_predictions - test_targets) ** 2))
    print("R2 {:.2f}, RMSE {:.2f}".format(r2, rmse))

    # 添加标题和标签
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('True vs Predicted values\nRMSE: {:.2f}, R^2: {:.2f}'.format(rmse, r2))
    plt.show()


def plot_residuals(predictions, targets):
    residuals = targets - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted values')
    plt.show()
