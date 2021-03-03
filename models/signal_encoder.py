import torch.nn as nn

from .positional_encoding import PositionalEncoding


class SignalEncoderConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv1d(1, 16, (3, 1))
        self.conv_2 = nn.Conv2d(16, 32, (3, 1))
        self.conv_3 = nn.Conv2d(32, 32, (3, 1))
        self.conv_4 = nn.Conv2d(32, 64, (3, 1))
        self.conv_5 = nn.Conv2d(64, 64, (3, 1))
        self.max = nn.AdaptiveMaxPool2d(8)

    def forward(self, inputs):
        x = self.conv_1(inputs.float())
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = self.max(x)

        return x


class SignalEncoderLinear(nn.Module):
    def __init__(self, num_patches, projection_dim, position=True):
        super().__init__()

        self.num_patches = num_patches
        self.projection = nn.Linear(3, projection_dim)
        if position:
            self.position_embedding = PositionalEncoding(projection_dim)

    def forward(self, inputs):
        x = self.projection(inputs.float())
        if self.position_embedding:
            x = self.position_embedding(x)

        return x
