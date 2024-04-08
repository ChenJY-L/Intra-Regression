import torch
import torch.nn as nn


class NNModel(nn.Module):
    # RMSE: 0.15
    # R2 0.74
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 2, hidden_size * 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size * 4, 16))

        self.backbone_net = nn.Sequential(nn.Conv1d(1, 16, 3),
                                          nn.ReLU(),
                                          nn.Conv1d(16, 32, 3),
                                          nn.ReLU(),
                                          nn.Conv1d(32, 64, 3))
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Sequential(nn.ReLU(),
                                 nn.Linear(640, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, output_size))

    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.backbone_net(x1)
        x3 = self.dropout(x2)

        x3 = x3.view(1, -1)
        y = self.fc2(x3)

        return y


class MyGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(MyGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x, state=None):
        batch_size = x.size(0)

        if state is None:
            h0 = x.data.new(self.num_layers, batch_size, self.hidden_size).zero_()
        else:
            h0 = state

        x = torch.unsqueeze(x, 1)
        y, state = self.rnn(x, h0)
        batch_size, time_steps, hidden_size = y.size()
        y = y.reshape(-1, hidden_size)
        y = self.dropout(y)
        y = self.fc(y)
        y = y.reshape(time_steps, batch_size, -1)

        return y[-1]


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)

        batch_size, time_steps, _ = x.size()
        x = x.reshape(time_steps, batch_size, -1)
        return x[-1]


class CNN1DRegression(nn.Module):
    def __init__(self, input_channel, output_size, num_channels=None, kernel_sizes=None, dropout=0.3):
        super(CNN1DRegression, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        if num_channels is None:
            num_channels = [8, 16]
        assert len(num_channels) == len(kernel_sizes), "Number of channels and kernel sizes should match"

        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=num_channels[0], kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=num_channels[0], out_channels=num_channels[1], kernel_size=kernel_sizes[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.max(x, dim=2)[0]  # Max pooling over time dimension
        x = self.fc(x)
        return x


class CNN_LSTM(nn.Module):
    # RMSE: 0.12
    # R2: 0.83
    def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.3):
        super(CNN_LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 2, hidden_size * 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size * 4, hidden_size))

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.backbone_net = nn.Sequential(nn.Conv1d(1, 16, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(16),
                                          nn.Conv1d(16, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(32),
                                          nn.Conv1d(32, 64, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(64))

        self.res = nn.Conv1d(1, 64, 1)

        self.dropout = nn.Dropout(drop_prob)
        self.fc2 = nn.Sequential(nn.LeakyReLU(),
                                 nn.Linear(hidden_size * 64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, output_size))

    def forward(self, x, state=None):
        batch_size = x.size(0)

        x = self.fc(x)  # (batch_size, 16)

        y, state = self.lstm(x)  # (batch_size, 128)

        y = torch.unsqueeze(y, dim=1)  # (batch_size, 1, 128)

        x_res = self.res(y)  # (batch_size, 64, 128)
        y = self.backbone_net(y)  # (batch_size, 64, 128)

        y = y + x_res  # (batch_size, 64, 128)

        y = self.dropout(y)  # (batch_size, 64, 128)

        y = y.view(batch_size, -1)  # (batch_size, 1, 64*128)
        y = self.fc2(y)

        return y


class Attention(nn.Module):
    def __init__(self, input_size, output_size, num_heads=2, hidden_size=16, dropout=0.1):
        super(Attention, self).__init__()

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)

        # Feedforward Layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Self-Attention
        x, _ = self.self_attention(x, x, x)

        # Feedforward
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    pass
