import torch
import torch.nn as nn


class NNModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, output_size))

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


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


class CNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, kernel_size=3):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(feature_size, hidden_size, kernel_size),
                                  nn.Conv1d(hidden_size, feature_size, kernel_size))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        res = self.conv(x)
        x = x + res
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, feature_size, output_size, num_heads=1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(feature_size, num_heads=num_heads)

        self.fc = nn.Sequential(nn.Linear(feature_size, output_size))

    def forward(self, x):
        x = self.attention(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    pass
