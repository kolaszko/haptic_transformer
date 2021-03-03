import torch.nn as nn

from .signal_encoder import SignalEncoderLinear


class HAPTR(nn.Module):
    def __init__(self, num_classes, projection_dim, hidden_dim, nheads, num_encoder_layers, mlp_head_dims=(2048, 1024),
                 dropout=0.5):
        super().__init__()

        self.signal_encoder = SignalEncoderLinear(hidden_dim, projection_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim * projection_dim, mlp_head_dims[0]),
            nn.BatchNorm1d(mlp_head_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_head_dims[0], mlp_head_dims[1]),
            nn.BatchNorm1d(mlp_head_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(mlp_head_dims[1], num_classes)

    def forward(self, inputs):
        x = self.signal_encoder(inputs)

        x = x.squeeze(1).permute(0, 2, 1)

        x = self.transformer(x)

        x = x.flatten(1)
        x = self.dropout(x)

        x = self.mlp_head(x)
        x = self.classifier(x)

        return x
