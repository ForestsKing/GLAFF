import torch
from torch import nn

from backbone.DLinear.layer import series_decomp
from plugin.Plugin.model import Plugin


class DLinear(nn.Module):
    def __init__(self, args, channel):
        super(DLinear, self).__init__()
        self.args = args
        self.channel = channel

        self.pred_len = args.pred_len
        self.hist_len = args.hist_len

        enc_in = channel
        moving_avg = 25
        individual = False

        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.hist_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.hist_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.hist_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.hist_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter((1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))

        self.flag = args.flag
        if self.flag == 'Plugin':
            self.plugin = Plugin(args, channel)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()
        x = x_enc

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        dec_out = seasonal_output + trend_output
        dec_out = dec_out.permute(0, 2, 1)
        pred = dec_out[:, -self.pred_len:, :]

        if self.flag == 'Plugin':
            pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred
