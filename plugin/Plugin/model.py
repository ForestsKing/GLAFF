import torch
from torch import nn


class Plugin(nn.Module):
    def __init__(self, args, channel):
        super(Plugin, self).__init__()
        self.args = args
        self.channel = channel

        self.q = args.q
        self.hist_len = args.hist_len
        self.pred_len = args.pred_len

        self.Encoder = nn.Sequential(
            nn.Linear(6, args.dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=args.dim,
                    nhead=args.head_num,
                    dim_feedforward=args.dff,
                    dropout=args.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=args.layer_num,
                norm=nn.LayerNorm(args.dim)
            ),
            nn.Linear(args.dim, self.channel)
        )

        self.MLP = nn.Sequential(
            nn.Linear(self.hist_len, args.dff),
            nn.GELU(),
            nn.Linear(args.dff, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_enc_true, x_mark_enc, x_dec_pred, x_mark_dec):
        means = torch.mean(x_enc_true, dim=1, keepdim=True)
        stdev = torch.std(x_enc_true, dim=1, keepdim=True) + 1e-6
        x_enc_true = (x_enc_true - means) / stdev
        x_dec_pred = (x_dec_pred - means) / stdev

        # map
        x_enc_map = self.Encoder(x_mark_enc)
        x_dec_map = self.Encoder(x_mark_dec)

        # denormalize
        robust_means_true = torch.median(x_enc_true, dim=1, keepdim=True)[0]
        robust_means_map = torch.median(x_enc_map, dim=1, keepdim=True)[0]
        robust_stdev_true = torch.quantile(x_enc_true, self.q, 1, True) - torch.quantile(
            x_enc_true, 1 - self.q, 1, True) + 1e-6
        robust_stdev_map = torch.quantile(x_enc_map, self.q, 1, True) - torch.quantile(
            x_enc_map, 1 - self.q, 1, True) + 1e-6
        x_enc_map = (x_enc_map - robust_means_map) / robust_stdev_map * robust_stdev_true + robust_means_true
        x_dec_map = (x_dec_map - robust_means_map) / robust_stdev_map * robust_stdev_true + robust_means_true

        # combine
        error = x_enc_true - x_enc_map
        weight = self.MLP(error.permute(0, 2, 1)).unsqueeze(1)
        x_dec = torch.stack([x_dec_map, x_dec_pred], dim=-1)
        pred = torch.sum(x_dec * weight, dim=-1)

        pred = pred * stdev + means
        
        return pred
