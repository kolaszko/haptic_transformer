import torch
import torch.nn as nn

from .signal_encoder import SignalEncoderLinear


class HAPTR(nn.Module):

    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
        super().__init__()
        self.sequence_length = sequence_length
        self.dim_modalities = dim_modalities
        self.num_modalities = num_modalities

        self.signal_encoder = SignalEncoderLinear(sequence_length, projection_dim, num_channels=sum(dim_modalities),
                                                  learnable=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim,
                                                   nhead=nheads,
                                                   dropout=dropout,
                                                   dim_feedforward=feed_forward,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm=nn.LayerNorm(projection_dim))

        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sequence_length, num_classes)
        )

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, -1)
        x = self.mlp_head(x)
        return x, {}  # no weights

    def warmup(self, device, num_reps=1, num_batch=1):
        for _ in range(num_reps):
            dummy_input = torch.randn((num_batch, self.sequence_length, sum(self.dim_modalities)),
                                      dtype=torch.float).to(device)
            self.forward(dummy_input)


class HAPTR_ModAtt(HAPTR):

    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
        super().__init__(num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                         dropout, dim_modalities, num_modalities)

        self.mod_attn = ModalityAttention(num_modalities, dim_modalities, sequence_length, dropout)

        self.mlp_head = nn.Sequential(
            nn.Linear(num_modalities + projection_dim, 1, 1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(sequence_length, num_classes)
        )

    def forward(self, inputs):
        xw, w = self.mod_attn(inputs)
        x = torch.cat(inputs, -1)
        x = self.signal_encoder(x)
        x = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = torch.concat([x, xw], -1)
        x = x.permute(1, 0, 2)
        x = self.mlp_head(x)
        return x, {"mod_weights": w}

    def warmup(self, device, num_reps=1, num_batch=1):
        for _ in range(num_reps):

            dummy_input = list()
            for mod_dim in self.dim_modalities:
                shape = [num_batch, self.sequence_length, mod_dim]
                mod_input = torch.randn(shape, dtype=torch.float).to(device)
                dummy_input.append(mod_input)

            self.forward(dummy_input)


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
        mods = torch.stack([self.flat_nn[i](inputs[i]) for i in range(self.num_modalities)])
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
