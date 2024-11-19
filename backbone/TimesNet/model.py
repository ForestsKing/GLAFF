import torch
from torch import nn

from backbone.TimesNet.embed import DataEmbedding
from backbone.TimesNet.layer import TimesBlock
from plugin.Plugin.model import Plugin


class TimesNet(nn.Module):
    def __init__(self, args, channel):
        super(TimesNet, self).__init__()
        self.args = args
        self.channel = channel

        self.pred_len = args.pred_len
        self.hist_len = args.hist_len

        enc_in = channel
        c_out = channel
        top_k = 5
        d_model = 128
        d_ff = 256
        num_kernels = 6
        e_layers = 2
        dropout = 0.1

        self.model = nn.ModuleList([
            TimesBlock(self.hist_len, self.pred_len, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)
        ])
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.predict_linear = nn.Linear(self.hist_len, self.pred_len + self.hist_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.flag = args.flag
        if self.flag == 'Plugin':
            self.plugin = Plugin(args, channel)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.hist_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.hist_len, 1))
        pred = dec_out[:, -self.pred_len:, :]

        if self.flag == 'Plugin':
            pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred
