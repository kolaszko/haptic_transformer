import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .signal_encoder import SignalEncoder


class HAPTR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers):
        super().__init__()

        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.signal_encoder = SignalEncoder()

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim*64),
            nn.Linear(hidden_dim*64, hidden_dim*32),
            nn.LayerNorm(hidden_dim * 32),
            nn.Linear(hidden_dim * 32, num_classes)
        )


    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = self.pos_encoding(x.flatten(2))
        x = self.transformer(x)
        x = x.flatten(1)
        x = self.mlp_head(x)

        return x
