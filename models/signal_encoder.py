import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding, LearnablePositionalEncoding


class SignalEncoderConv(nn.Module):
    def __init__(self, num_patches, projection_dim, position=True, modalities=6, learnable=False):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 8, (3, 1))
        self.bn_1 = nn.BatchNorm2d(8)
        self.conv_2 = nn.Conv2d(8, 16, (3, 1))
        self.bn_2 = nn.BatchNorm2d(16)
        self.conv_3 = nn.Conv2d(16, 64, (3, 1))
        self.bn_3 = nn.BatchNorm2d(64)

        self.max = nn.AdaptiveMaxPool2d(4)

        if position:
            if learnable:
                self.position_embedding = LearnablePositionalEncoding(dict_size=num_patches,
                                                                      num_pos_feats=projection_dim)
            else:
                self.position_embedding = PositionalEncoding(projection_dim)

    def forward(self, inputs):
        x = F.relu(self.bn_1(self.conv_1(inputs.float())))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.max(x)

        x = x.flatten(-2, -1)
        x = x.unsqueeze(1)

        if self.position_embedding:
            x = self.position_embedding(x)

        return x


class SignalEncoderLinear(nn.Module):
    def __init__(self, num_patches, projection_dim, position=True, num_channels=6, learnable=False):
        super().__init__()

        self.num_patches = num_patches
        self.projection = nn.Sequential(
            nn.Linear(num_channels, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        if position:
            if learnable:
                self.position_embedding = LearnablePositionalEncoding(dict_size=num_patches,
                                                                      num_pos_feats=projection_dim)
            else:
                self.position_embedding = PositionalEncoding(projection_dim)

    def forward(self, inputs):
        x = self.projection(inputs)
        if self.position_embedding:
            x = self.position_embedding(x)

        return x
