import os
from time import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.DLinear.model import DLinear
from backbone.Informer.model import Informer
from backbone.TimesNet.model import TimesNet
from backbone.iTransformer.model import iTransformer
from data.dataset import MyDataset
from util.eval import getModelSize, evaluate
from util.stoper import Stopper

model_dict = {
    'Informer': Informer,
    'DLinear': DLinear,
    'TimesNet': TimesNet,
    'iTransformer': iTransformer,
}


class Solver:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

        self.device = self._acquire_device()
        self.model_path = self._make_dirs()
        self.train_loader, self.valid_loader, self.test_loader = self._get_loader()

        self.channel = self.train_loader.dataset.data.shape[-1]
        self.model = model_dict[self.args.model](self.args, self.channel).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.stopper = Stopper(patience=self.args.patience, path=self.model_path + '/' + 'checkpoint.pth')
        self.criterion = nn.MSELoss()

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.device))
            print('Use GPU: cuda:{}'.format(self.args.device))
        else:
            device = torch.device('cpu')
            print('Use CPU')

        return device

    def _make_dirs(self):
        model_path = os.path.join(self.args.save_path, self.setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        return model_path

    def _get_loader(self):
        train_set = MyDataset(self.args, flag='train')
        valid_set = MyDataset(self.args, flag='valid')
        test_set = MyDataset(self.args, flag='test')

        print('Train Data Shape: ', train_set.data.shape)
        print('Valid Data Shape: ', valid_set.data.shape)
        print('Test Data Shape: ', test_set.data.shape)

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader

    def _process_one_batch(self, x_time, x_data, y_time, y_data):
        x_time = x_time.float().to(self.device)
        x_data = x_data.float().to(self.device)
        y_time = y_time.float().to(self.device)
        y_data = y_data.float().to(self.device)

        label_len = self.args.hist_len // 2
        x_enc = x_data.to(self.device)
        x_mark_enc = x_time.to(self.device)
        x_dec = torch.concat([x_data[:, -label_len:, :], torch.zeros_like(y_data)], dim=1).to(self.device)
        x_mark_dec = torch.concat([x_time[:, -label_len:, :], y_time], dim=1).to(self.device)

        pred, reco, map1, map2 = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.args.flag == 'Plugin':
            loss = self.criterion(torch.concat([pred, reco, map1, map2], dim=1),
                                  torch.concat([y_data, x_data, y_data, y_data], dim=1))
        else:
            loss = self.criterion(pred, y_data)

        return loss, pred

    def train(self):
        i = 0
        for e in range(self.args.epoch):
            start = time()

            self.model.train()
            train_loss = []
            for (x_time, x_data, y_time, y_data) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss, _ = self._process_one_batch(x_time, x_data, y_time, y_data)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (x_time, x_data, y_time, y_data) in tqdm(self.valid_loader):
                    loss, _ = self._process_one_batch(x_time, x_data, y_time, y_data)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f} || Cost: {3:.6f}s".format(
                e, train_loss, valid_loss, end - start)
            )

            self.stopper(valid_loss, self.model)
            i = i + 1
            if self.stopper.early_stop:
                break

        return i

    def test(self):
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth', map_location=self.device))
        model_size = getModelSize(self.model)

        with torch.no_grad():
            self.model.eval()
            hist, true, pred = [], [], []
            for (x_time, x_data, y_time, y_data) in tqdm(self.test_loader):
                _, output = self._process_one_batch(x_time, x_data, y_time, y_data)
                hist.append(x_data.detach().cpu().numpy())
                true.append(y_data.detach().cpu().numpy())
                pred.append(output.detach().cpu().numpy())

        hist = np.concatenate(hist, axis=0)
        pred = np.concatenate(pred, axis=0)
        true = np.concatenate(true, axis=0)

        print('Hist Shape:', hist.shape)
        print('Pred Shape:', pred.shape)
        print('True Shape:', true.shape)

        mse, mae = evaluate(pred, true)
        print('MSE:{0:.4f}, MAE:{1:.4f}'.format(mse, mae))

        return mse, mae, model_size
