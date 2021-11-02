import torch
import torch.nn as nn

from .signal_encoder import SignalEncoderLinear


class HAPTR(nn.Module):

    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout=0.5):
        super().__init__()
        self.signal_encoder = SignalEncoderLinear(sequence_length, projection_dim, learnable=False)
        # self.signal_encoder = SignalEncoderConv(hidden_dim, projection_dim, learnable=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=nheads, dropout=dropout,
                                                   dim_feedforward=feed_forward, activation='gelu')
        encoder_norm = nn.LayerNorm(projection_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.mlp_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(sequence_length, num_classes)
        )

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)

        attention_out = torch.stack([l.self_attn(x, x, x)[0] for l in self.transformer.layers], -1)
        anim_attn_weights = attention_out.mean((-1, -2)).permute(1, 0)

        x = x.permute(1, 0, 2)
        x = torch.mean(x, -1)
        x = self.mlp_head(x)

        return x, anim_attn_weights


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
