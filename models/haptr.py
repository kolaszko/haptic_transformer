import torch.nn as nn
import torch

from .signal_encoder import SignalEncoderLinear, SignalEncoderConv


class HAPTR(nn.Module):
    def __init__(self, num_classes, projection_dim, hidden_dim, nheads, num_encoder_layers,
                 dropout=0.5, prediction=False):
        super().__init__()
        self.signal_encoder = SignalEncoderLinear(hidden_dim, projection_dim, learnable=True)
        # self.signal_encoder = SignalEncoderConv(hidden_dim, projection_dim, learnable=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout, dim_feedforward=1024, activation='gelu')
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        if prediction:
            self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
            )

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(1)
        x = self.mlp_head(x)

        return x


class HAPTRLV(nn.Module):
    def __init__(self, haptr):
        super(HAPTRLV, self).__init__()

        self.signal_encoder = haptr.signal_encoder
        self.transformer = haptr.transformer

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.transformer(x)
        return x
