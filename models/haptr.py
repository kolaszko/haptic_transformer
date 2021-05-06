import torch
import torch.nn as nn

from .signal_encoder import SignalEncoderLinear


class HAPTR(nn.Module):
    def __init__(self, num_classes, projection_dim, hidden_dim, nheads, num_encoder_layers, feed_forward,
                 dropout=0.5):
        super().__init__()
        self.signal_encoder = SignalEncoderLinear(160, projection_dim, learnable=True)
        # self.signal_encoder = SignalEncoderConv(hidden_dim, projection_dim, learnable=False)

        # encoder_layer = MyTransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout, dim_feedforward=1024, activation='gelu')
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout, dim_feedforward=feed_forward, activation='gelu')
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(160, num_classes)
        )


    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, -1)
        x = self.mlp_head(x)

        return x


class HAPTRLV(nn.Module):
    def __init__(self, haptr):
        super(HAPTRLV, self).__init__()

        self.signal_encoder = haptr.signal_encoder
        self.transformer = haptr.transformer

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        return x
