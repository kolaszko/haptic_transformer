import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class SignalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 16, (3, 1))
        # self.maxpool = nn.MaxPool2d((3, 1), stride=2)
        self.conv_2 = nn.Conv2d(16, 32, (3, 1))
        self.conv_3 = nn.Conv2d(32, 32, (3, 1))
        self.conv_4 = nn.Conv2d(32, 64, (3, 1))
        self.conv_5 = nn.Conv2d(64, 64, (3, 1))
        self.max = nn.AdaptiveMaxPool2d(8)

    def forward(self, inputs):
        x = self.conv_1(inputs.float())
        # x = self.maxpool(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        # x = self.max(x)

        return x
