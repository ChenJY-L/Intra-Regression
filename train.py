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
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models import *



class Config:
    data_path = './data/data.xlsx'
    timestep = 1  # 时间步长
    batch_size = 64
    learning_rate = 0.0003
    feature_size = 6  # 输入特征
    hidden_size = 128  # 隐藏层维度
    output_size = 1
    num_layers = 1  # GRU层数
    dropout_prob = 0.7
    num_epochs = 1000
    best_loss = 1
    model_name = 'lstm'
    save_path = './results/{}.pth'.format(model_name)


class IntraDataset(Dataset):

    def __init__(self, file_path, time_steps, feature_size):
        # 读取文件
        df = pd.read_excel(file_path)

        # 数据归一化
        scaler = MinMaxScaler()
        df_normalized = scaler.fit_transform(df)

        # 将归一化后的数据转换为 PyTorch 张量
        self.data_x, self.data_y = self.split_data(df_normalized, time_steps, feature_size)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

    def split_data(self, data, time_steps, feature_size):
        data_x = []
        data_y = []

        for i in range(len(data) - time_steps):
            data_x.append(data[i:i + time_steps][:, 3:])
            data_y.append(data[i + time_steps][1])

        data_x = torch.tensor(data_x).float()
        data_y = torch.tensor(data_y).float()

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

    # 1. 读入数据
    df = pd.read_excel(config.data_path)
    scaler = MinMaxScaler()
    scaler_model = MinMaxScaler()
    df = scaler.fit_transform(np.array(df))
    # df = np.array(df)
    x_train, y_train, x_test, y_test = spilt_data(df, config.timestep, config.feature_size)

    # 2. 创建数据集
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    # 3. 创建迭代器
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # 4. 创建模型
    # model = MyLSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size, config.dropout_prob)
    model = MyGRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size, config.dropout_prob)
    # model = NNModel(config.feature_size, config.hidden_size, config.output_size)
    # model = Attention(config.feature_size, config.output_size, 1)
    # model = CNN1DRegression(config.timestep, config.output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs // 2)

    # 5. 训练模型
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for data in train_bar:
            x, y = data
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = '[{}/{}] Train Epoch: loss:{:.3f} '.format(epoch + 1, config.num_epochs, loss)
        scheduler.step()
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                x, y = data
                y_pred = model(x)
                test_loss += loss_fn(y_pred, y)
                test_bar.desc = 'Test: loss:{:.3f}'.format(test_loss)

            if test_loss < config.best_loss:
                config.best_loss = test_loss
                torch.save(model.state_dict(), config.save_path)
                print('\nBest loss: {}\n'.format(config.best_loss))

    print('Finished')
    # model.load_state_dict(torch.load(config.save_path))
    # 绘制训练结果
    plot_size = 200
    plt.figure(figsize=(12, 8))
    y_pred = model(x_train).detach().numpy()
    plt.plot(y_pred, label='train')
    plt.plot(y_train.detach().numpy(), label='ref')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
