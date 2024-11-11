import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, args, flag='train'):
        df = pd.read_csv(args.data_path + args.dataset + '.csv')

        time = self._get_time_feature(df[['date']])
        data = df.drop(['date'], axis=1).values

        scaler = StandardScaler()
        train_data = data[:int(0.6 * len(data)), :]
        scaler.fit(train_data)
        data = scaler.transform(data)

        if flag == 'train':
            self.data = data[:int(0.6 * len(data)), :]
            self.time = time[:int(0.6 * len(data)), :]
        elif flag == 'valid':
            self.data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
            self.time = time[int(0.6 * len(data)):int(0.8 * len(data)), :]
        elif flag == 'test':
            self.data = data[int(0.8 * len(data)):, :]
            self.time = time[int(0.8 * len(data)):, :]
        else:
            raise ValueError('Invalid flag: %s' % flag)

        self.hist_len = args.hist_len
        self.pred_len = args.pred_len

    @staticmethod
    def _get_time_feature(df):
        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].apply(lambda row: row.month / 12 - 0.5)
        df['day'] = df['date'].apply(lambda row: row.day / 31 - 0.5)
        df['weekday'] = df['date'].apply(lambda row: row.weekday() / 6 - 0.5)
        df['hour'] = df['date'].apply(lambda row: row.hour / 23 - 0.5)
        df['minute'] = df['date'].apply(lambda row: row.minute / 59 - 0.5)
        df['second'] = df['date'].apply(lambda row: row.second / 59 - 0.5)

        return df[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values

    def __getitem__(self, index):
        x_time = self.time[index:index + self.hist_len, :]
        x_data = self.data[index:index + self.hist_len, :]

        y_time = self.time[index + self.hist_len:index + self.hist_len + self.pred_len, :]
        y_data = self.data[index + self.hist_len:index + self.hist_len + self.pred_len, :]

        return x_time, x_data, y_time, y_data

    def __len__(self):

        return len(self.data) - self.hist_len - self.pred_len + 1
