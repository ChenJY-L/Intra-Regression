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


class CNN1DRegression(nn.Module):
    def __init__(self, input_size, output_size, num_channels=[64, 128], kernel_sizes=[3, 3], dropout=0.5):
        super(CNN1DRegression, self).__init__()
        assert len(num_channels) == len(kernel_sizes), "Number of channels and kernel sizes should match"

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels[0], kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=num_channels[0], out_channels=num_channels[1], kernel_size=kernel_sizes[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.max(x, dim=2)[0]  # Max pooling over time dimension
        x = self.fc(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()

        # CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        # CNN forward pass
        x = self.cnn(x)

        # Reshape for LSTM input: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        # LSTM forward pass
        _, (h_n, _) = self.lstm(x)

        # Take the last hidden state
        out = self.fc(h_n[-1])
        return out

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
