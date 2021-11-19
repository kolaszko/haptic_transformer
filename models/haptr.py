import torch
import torch.nn as nn

from .signal_encoder import SignalEncoderLinear


class HAPTR(nn.Module):

    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
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

        self.dim_modalities = dim_modalities
        self.num_modalities = num_modalities

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, -1)
        x = self.mlp_head(x)
        return x


class HAPTR_ModAtt(HAPTR):

    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
        super().__init__(num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                         dropout, dim_modalities, num_modalities)

        self.mod_attn = ModalityAttention(num_modalities, dim_modalities, sequence_length, dropout)

    def forward(self, inputs):
        _, w = self.mod_attn(inputs)
        x = inputs * w.unsqueeze(2)
        x = x.view(*x.shape[:-2], -1)
        x = self.signal_encoder(x)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, -1)
        x = self.mlp_head(x)

        return x


class ModalityAttention(nn.Module):
    def __init__(self, num_modalities, dim_modalities, sequence_length, dropout):
        super().__init__()
        self.num_modalities = num_modalities
        self.dim_modalities = dim_modalities
        assert len(self.dim_modalities) == self.num_modalities

        self.seq_length = sequence_length
        self.dropout = dropout
        self.self_attn = nn.MultiheadAttention(embed_dim=self.num_modalities, num_heads=1, dropout=self.dropout,
                                               kdim=self.seq_length, vdim=self.seq_length)
        self.flat_nn = nn.ModuleList([FlattenModality(dim, dropout) for dim in self.dim_modalities])

    def forward(self, inputs):
        mods = torch.stack([self.flat_nn[i](inputs[..., i]) for i in range(self.num_modalities)])
        q = mods.permute(2, 1, 0)
        x, w = self.self_attn(q, mods, mods, need_weights=True)
        return x, w


class FlattenModality(nn.Module):
    def __init__(self, dim_modality_in, dropout):
        super().__init__()

        self.flatten = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_modality_in, 1)
        )

    def forward(self, x, squeeze_output=True):
        x = self.flatten(x)
        if squeeze_output:
            x = x.squeeze(-1)
        return x
