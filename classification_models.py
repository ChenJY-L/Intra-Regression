import torch
import torch.nn as nn
import torch.nn.functional as F

from regression_models import ResidualBlock1D


class CLSRes(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_blocks=3, drop_prob=0.3):
        super(CLSRes, self).__init__()

        # self.num_layers = num_layers
        # self.hidden_size = hidden_size

        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_size // 4),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size // 4, hidden_size * 4),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size * 4, hidden_size))

        self.backbone_net1 = ResidualBlock1D(1, 32)
        self.backbone_net2 = ResidualBlock1D(32, 64)
        self.avg_pool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(drop_prob)

        self.fc2 = nn.Sequential(nn.ReLU(),
                                 nn.Linear(hidden_size*64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, num_classes),
                                 nn.Softmax())

    def _make_blocks(self, base_channels, num_blocks):
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualBlock1D(base_channels, base_channels))
        return nn.Sequential(*blocks)

    def forward(self, x, state=None):
        batch_size = x.size(0)
        y = self.MLP(x)  # (batch_size, hidden_size)
        y = y.unsqueeze(1)
        y = self.backbone_net1(y)  # (batch_size, 32, hidden_size)
        y = self.backbone_net2(y)  # (batch_size, 64, hidden_size)
        y = self.avg_pool(y)
        y = self.dropout(y)  # (batch_size, 64, hidden_size)
        y = y.view(batch_size, -1)  # (batch_size, 1, 64*hidden_size)
        y = self.fc2(y)
        y = y.squeeze(1)
        return y
