import torch
import torch.nn as nn
import torch.nn.functional as F


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
        batch_size = x.size(0)

        x1 = self.fc(x)
        x1 = x1.unsqueeze(1)
        x2 = self.backbone_net(x1)
        x3 = self.dropout(x2)

        x3 = x3.view(batch_size, -1)
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


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)

        # Shortcut connection if input and output dimensions are different
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add shortcut connection
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class ResNet1D(nn.Module):
    # RMSE: 0
    # R2: 0
    def __init__(self, input_size, hidden_size, output_size, num_blocks=3, drop_prob=0.3):
        super(ResNet1D, self).__init__()

        # self.num_layers = num_layers
        # self.hidden_size = hidden_size

        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size // 4),
                                nn.ELU(),
                                nn.Linear(hidden_size // 4, hidden_size * 4),
                                nn.ELU(),
                                nn.Linear(hidden_size * 4, hidden_size))

        # self.conv = nn.Conv1d(1, 64, 1)
        self.backbone_net1 = ResidualBlock1D(1, 32)
        self.backbone_net2 = ResidualBlock1D(32, 64)
        self.avg_pool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(drop_prob)

        self.fc2 = nn.Sequential(nn.ELU(),
                                 nn.Linear(hidden_size * 64, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, output_size))

    def _make_blocks(self, base_channels, num_blocks):
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualBlock1D(base_channels, base_channels))
        return nn.Sequential(*blocks)

    def forward(self, x, state=None):
        batch_size = x.size(0)
        y = self.fc(x)  # (batch_size, hidden_size)
        y = y.unsqueeze(1)
        y = self.backbone_net1(y)  # (batch_size, 32, hidden_size)
        y = self.backbone_net2(y)  # (batch_size, 64, hidden_size)
        y = self.avg_pool(y)
        y = self.dropout(y)  # (batch_size, 64, hidden_size)
        y = y.view(batch_size, -1)  # (batch_size, 1, 64*hidden_size)
        y = self.fc2(y)
        y = y.squeeze(1)
        return y


class CNN_LSTM(nn.Module):
    # RMSE: 0.12
    # R2: 0.83
    def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.3):
        super(CNN_LSTM, self).__init__()

        # self.num_layers = num_layers
        # self.hidden_size = hidden_size

        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size // 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 4, hidden_size * 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size * 4, hidden_size))

        self.backbone_net = nn.Sequential(nn.Conv1d(1, 16, 1, 1, 0),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(16),
                                          nn.Conv1d(16, 32, 3, 1, 1),
                                          # nn.ReLU(),
                                          # nn.BatchNorm1d(32),
                                          # nn.Conv1d(32, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(32),
                                          nn.Conv1d(32, 64, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(64),
                                          nn.Dropout(0.1))

        self.res = nn.Conv1d(1, 64, 1)

        self.backbone_net2 = nn.Sequential(nn.ReLU(),
                                           nn.BatchNorm1d(64),
                                           nn.Conv1d(64, 128, 3, 1, 1),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(128),
                                           nn.Conv1d(128, 128, 3, 1, 1),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(128),
                                           nn.Conv1d(128, 64, 3, 1, 1),
                                           nn.Dropout(0.1))

        # self.lstm = nn.LSTM(input_size=hidden_size*64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc2 = nn.Sequential(nn.ReLU(),
                                 nn.Linear(hidden_size * 64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, output_size))

    def forward(self, x, state=None):
        batch_size = x.size(0)
        y = self.fc(x)  # (batch_size, hidden_size)
        y = torch.unsqueeze(y, dim=1)  # (batch_size, 1, hidden_size)
        res = self.res(y)  # (batch_size, 64, hidden_size)
        y = self.backbone_net(y)  # (batch_size, 64, hidden_size)
        res2 = y  # (batch_size, 64, hidden_size)
        y = y + res  # (batch_size, 64, hidden_size)
        y = self.backbone_net2(y)  # (batch_size, 64, hidden_size)
        y = y + res2  # (batch_size, 64, hidden_size)
        y = self.dropout(y)  # (batch_size, 64, hidden_size)
        y = y.view(batch_size, -1)  # (batch_size, 1, 64*hidden_size)
        # y, state = self.lstm(y)  # (batch_size, hidden_size)
        y = self.fc2(y)
        y = y.squeeze(1)
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
