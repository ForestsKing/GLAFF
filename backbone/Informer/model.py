import torch
from torch import nn

from backbone.Informer.atten import AttentionLayer, ProbAttention
from backbone.Informer.embed import DataEmbedding
from backbone.Informer.layer import Encoder, EncoderLayer, ConvLayer, Decoder, DecoderLayer
from plugin.Plugin.model import Plugin


class Informer(nn.Module):
    def __init__(self, args, channel):
        super(Informer, self).__init__()
        self.args = args
        self.channel = channel

        self.pred_len = args.pred_len
        self.hist_len = args.hist_len

        enc_in = channel
        c_out = channel
        dec_in = channel
        d_model = 512
        d_ff = 2048
        n_heads = 8
        e_layers = 2
        d_layers = 1
        dropout = 0.1
        factor = 3
        distil = True
        activation = 'gelu'
        output_attention = False

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

        self.flag = args.flag
        if self.flag == 'Plugin':
            self.plugin = Plugin(args, channel)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        pred = dec_out[:, -self.pred_len:, :]

        if self.flag == 'Plugin':
            pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred
