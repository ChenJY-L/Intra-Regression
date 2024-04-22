import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


class IntraDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):
        # 将归一化后的数据转换为 PyTorch 张量
        self.transform = transform
        # self.data_x, self.data_y = split_data(df)
        self.data_x = torch.tensor(df.iloc[:, 3:].values).float()
        self.data_y = torch.tensor(df.iloc[:, 2].values).float()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        return x, y


class ClsIntraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_label, transform=None):
        self.transform = transform
        self.class_to_label = class_to_label

        self.data_x = torch.tensor(df.iloc[:, 3:].values).float()
        self.data_y = df.iloc[:, 0].values

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        y = self.class_to_label[y]
        return x, y


def read_data(file_path):
    """
    Read data from a file into a pandas DataFrame.

    Args:
        file_path (str): The path to the file.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the file.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

    return df


def read_dataset(train, test):
    return read_data(train), read_data(test)


def divide_dataframe(df, ratio):
    """
    Divide a DataFrame into training and testing DataFrames based on the input ratio.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be divided.
        ratio (float): The ratio of training data to total data. Value should be between 0 and 1.

    Returns:
        tuple: A tuple containing the training DataFrame and testing DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not 0 < ratio < 1:
        raise ValueError("Input 'ratio' must be a float between 0 and 1.")

    train_size = int(len(df) * ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]

    return train_df, test_df


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
