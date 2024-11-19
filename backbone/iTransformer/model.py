import torch
from torch import nn

from backbone.iTransformer.atten import AttentionLayer, FullAttention
from backbone.iTransformer.embed import DataEmbedding_inverted
from backbone.iTransformer.layer import Encoder, EncoderLayer
from plugin.Plugin.model import Plugin


class iTransformer(nn.Module):
    def __init__(self, args, channel):
        super(iTransformer, self).__init__()
        self.args = args
        self.channel = channel

        self.pred_len = args.pred_len
        self.hist_len = args.hist_len

        d_model = 512
        d_ff = 2048
        dropout = 0.1
        factor = 3
        n_heads = 8
        e_layers = 4
        activation = 'gelu'
        output_attention = False

        self.enc_embedding = DataEmbedding_inverted(self.hist_len, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

        self.flag = args.flag
        if self.flag == 'Plugin':
            self.plugin = Plugin(args, channel)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        pred = dec_out[:, -self.pred_len:, :]

        if self.flag == 'Plugin':
            pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred
