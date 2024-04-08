"""
                  _ooOoo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                  O\ = /O
              ____/`---'\____
            .   ' \\| |// `.
             / \\||| : |||// \
           / _||||| -:- |||||- \
             | | \\\ - /// | |
           | \_| ''\---/'' | |
            \ .-\__ `-` ___/-. /
         ___`. .' /--.--\ `. . __
      ."" '< `.___\_<|>_/___.' >'"".
     | | : `- \`.;`\ _ /`;.`/ - ` : | |
         \ \ `-. \_ __\ /__ _/ .-` / /
 ======`-.____`-.___\_____/___.-`____.-'======
                    `=---='

  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             佛祖保佑       永无BUG
"""

# Encoding utf-8
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models import *
from evaluate import *


class Config:
    train_data_path = './data/Intra_train.xlsx'
    test_data_path = './data/Intra_test.xlsx'
    timestep = 1  # 时间步长
    batch_size = 16
    learning_rate = 3e-6
    feature_size = 6  # 输入特征
    hidden_size = 128  # 隐藏层维度
    output_size = 1
    num_layers = 1  # GRU层数
    dropout_prob = 0.3
    num_epochs = 1000
    best_loss = float('inf')
    model_name = 'cnn-lstm'
    save_path = './results/{}.pth'.format(model_name)


class IntraDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):
        # 将归一化后的数据转换为 PyTorch 张量
        self.transform = transform
        # self.data_x, self.data_y = split_data(df)
        self.data_x = torch.tensor(df.iloc[:, 1:].values).float()
        self.data_y = torch.tensor(df.iloc[:, 0].values).float()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        return x, y


def split_data(data: pd.DataFrame):
    data_x = []
    data_y = []

    for i in range(len(data)):
        data_x.append(data.iloc[i, 3:])
        data_y.append(data.iloc[i, 2])

    data_x = torch.tensor(data_x).float()
    data_y = torch.tensor(data_y).float()

    # # 计算数据的均值和标准差
    # mean_x = data_x.mean(dim=0)
    # std_x = data_x.std(dim=0)
    # mean_y = data_y.mean()
    # std_y = data_y.std()
    #
    # # 归一化操作
    # data_x = (data_x - mean_x) / std_x
    # data_y = (data_y - mean_y) / std_y

    return data_x, data_y


def spilt_data(data, time_steps, feature_size):
    data_x = []
    data_y = []

    for i in range(len(data) - time_steps):
        data_x.append(data[i:i + time_steps][:, 0:feature_size])
        data_y.append(data[i + time_steps][7])

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    train_size = int(len(data_x) * 0.7)

    x_train = data_x[:train_size, :].reshape(-1, time_steps, feature_size)
    y_train = data_y[:train_size].reshape(-1, 1)

    x_test = data_x[train_size:, :].reshape(-1, time_steps, feature_size)
    y_test = data_y[train_size:].reshape(-1, 1)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    return [x_train, y_train, x_test, y_test]


def train():
    # Load config
    config = Config()

    # 1. 创建数据集
    # 读取文件
    train_df = pd.read_excel(config.train_data_path)
    test_df = pd.read_excel(config.test_data_path)
    # train_df = pd.read_csv(config.train_data_path)
    # test_df = pd.read_csv(config.test_data_path)

    x_scaler = MinMaxScaler()
    train_df.iloc[:, 1:] = x_scaler.fit_transform(train_df.iloc[:, 1:])
    test_df.iloc[:, 1:] = x_scaler.transform(test_df.iloc[:, 1:])

    y_scaler = MinMaxScaler()
    train_df.iloc[:, 0] = y_scaler.fit_transform(train_df.iloc[:, 0].values.reshape(-1, 1))
    test_df.iloc[:, 0] = y_scaler.transform(test_df.iloc[:, 0].values.reshape(-1, 1))

    train_data = IntraDataset(train_df)
    test_data = IntraDataset(test_df)

    # 2. 创建迭代器
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # 3. 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MyGRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size, config.dropout_prob)
    # model = CNN1DRegression(config.timestep, config.output_size)
    # model = NNModel(config.feature_size, config.hidden_size, config.output_size)
    model = CNN_LSTM(config.feature_size, config.hidden_size, config.output_size, config.num_layers, config.dropout_prob)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=1e-2, weight_decay=1e-4)

    train_losses = []
    test_losses = []

    best_loss = float('inf')

    # 4. 训练模型
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for data in train_bar:
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_description('[{}/{}] Train Epoch: loss:{:.5f} '.format(epoch + 1, config.num_epochs, loss))
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                x, y = data[0].to(device), data[1].to(device)
                y_pred = model(x)
                test_loss += loss_fn(y_pred, y)
                test_bar.set_description('Test: loss:{:.5f}'.format(test_loss / len(test_loader)))

        test_loss /= len(test_loader)
        test_losses.append(test_loss.cpu())

        if test_loss < best_loss:
            best_loss = test_loss
            try:
                torch.save(model.state_dict(), config.save_path)
                print('\nBest loss: {}\n'.format(best_loss))
            finally:
                pass

    print('Finished')
    # 5. 评估模型
    plot_loss(train_losses, test_losses)

    # 获取所有预测值和目标值，用于绘制预测结果对比图
    all_predictions, all_targets = evaluate_model(model, train_loader)
    plot_predictions(all_predictions, all_targets)
    # print(all_predictions)

    # print(all_targets)


if __name__ == '__main__':
    train()
