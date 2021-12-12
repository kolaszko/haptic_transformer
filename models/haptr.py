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

        # encodes a temporal position of values in timeseries
        self.signal_encoder = SignalEncoderLinear(sequence_length, projection_dim, num_channels=sum(dim_modalities),
                                                  learnable=False)

        # create a transformer layer that converts input timeseries into feature timeseries
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=projection_dim,
                                                                                          nhead=nheads,
                                                                                          dropout=dropout,
                                                                                          dim_feedforward=feed_forward,
                                                                                          activation='gelu'),
                                                 num_layers=num_encoder_layers,
                                                 norm=nn.LayerNorm(projection_dim))

        # flatten each timestep with N channels into signle channel (each timestep separately)
        self.conv1d = Conv1D(projection_dim, 1, dropout)

        # classify resulting timeseries
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(sequence_length),
            nn.Dropout(dropout),
            nn.Linear(sequence_length, num_classes)
        )

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        transformer_input = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(transformer_input)
        conv_input = x.permute(1, 2, 0)
        x = self.conv1d(conv_input)
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
        self.conv1d = Conv1D(projection_dim + num_modalities, 1, dropout)

    def forward(self, inputs):
        x_weighted, weights = self.mod_attn(inputs)
        x = self.signal_encoder(torch.cat(inputs, -1))
        transformer_input = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(transformer_input)
        conv_input = torch.cat([x, x_weighted], -1).permute(1, 2, 0)
        x = self.conv1d(conv_input)
        x = self.mlp_head(x)
        return x, {"mod_weights": weights}

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
        assert len(dim_modalities) == num_modalities

        self.num_modalities = num_modalities
        self.dim_modalities = dim_modalities
        self.seq_length = sequence_length
        self.dropout = dropout

        # attention layer with one head
        self.self_attn = nn.MultiheadAttention(embed_dim=self.num_modalities,
                                               num_heads=1,
                                               dropout=self.dropout)

        # flatten modalities to obtain 1 weight per each
        self.flat_nn = nn.ModuleList([Conv1D(dim, 1, dropout) for dim in self.dim_modalities])

    def forward(self, inputs):
        mods = torch.stack([self.flat_nn[i](inputs[i].permute(0, 2, 1)) for i in range(self.num_modalities)])
        sa_input = mods.permute(2, 1, 0)
        x, w = self.self_attn(query=sa_input, key=sa_input, value=sa_input, need_weights=True)
        return x, w


class Conv1D(nn.Module):
    def __init__(self, dim_modality_in, dim_modality_out, dropout=0.1):
        super().__init__()
        self.flatten = nn.Sequential(
            nn.Conv1d(dim_modality_in, dim_modality_out, 1),
            nn.BatchNorm1d(dim_modality_out),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, squeeze_output=True):
        '''
        Convolves each timestep with kernel separately.
        INPUT: BATCH x CHANNELS x LENGTH
        RETURNS: BATCH x LENGTH x CHANNELS
        '''
        x = self.flatten(x).permute(0, 2, 1)

        if squeeze_output:
            x = x.squeeze(-1)
        return x
