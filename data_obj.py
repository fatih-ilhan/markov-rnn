import os
import pickle as pkl
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from simulator import simulate_ar_data, simulate_ar_markov_data, simulate_binary_data, simulate_ts_data, simulate_sin_markov_data

"""
Includes utilization functions
"""


class Data:
    def __init__(self, dataset_name, data_dir="data", scale_mode="none", tensor_flag=True, diff_flag=False):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.scale_mode = scale_mode
        self.tensor_flag = tensor_flag
        self.diff_flag = diff_flag

        scaler_dispatcher = {"standard": StandardScaler,
                             "minmax": MinMaxScaler}

        if scale_mode != "none":
            self.data_scaler = scaler_dispatcher[scale_mode]()
            self.pred_scaler = scaler_dispatcher[scale_mode]()

        # data: tuple of lists for input target pairs
        # input_series: x_0, x_1, x_2 ...
        # output_series: y_0, y_1, y_2 ...
        self.data, self.input_series, self.output_series = self.prepare_data()
        self.input_dim = self.data[0][0].shape[-1]
        self.output_dim = self.data[0][1].shape[-1]
        self.len = len(self.data[0])

        if scale_mode != "none":
            self.pred_scaler.fit(self.output_series[:, None])

    def prepare_data(self):

        helper_dispatcher = {"elev": self.prepare_data_elev,
                             "puma32f": self.prepare_data_puma32f,
                             "pumadyn": self.prepare_data_pumadyn,
                             "alcoa": self.prepare_data_alcoa,
                             "euro": self.prepare_data_euro,
                             "sim_arima": self.prepare_data_sim_arima,
                             "sim_arima_markov": self.prepare_data_sim_arima_markov,
                             "sim_binary": self.prepare_data_sim_binary,
                             "sim_ts": self.prepare_data_sim_ts,
                             "sim_sin_markov": self.prepare_data_sim_sin_markov,
                             "USDEUR": self.prepare_data_usdeur,
                             "USDGBP": self.prepare_data_usdgbp,
                             "USDCHF": self.prepare_data_usdchf,
                             "USDJPY": self.prepare_data_usdjpy,
                             "USDTRY": self.prepare_data_usdtry,
                             "USDXAU": self.prepare_data_usdxau}

        return helper_dispatcher[self.dataset_name]()

    def prepare_data_elev(self):
        matfile = loadmat(os.path.join(self.data_dir, "elev_data.mat"))
        seq = matfile["nngc_data"]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(seq)):
            inp = norm_seq[i, :-1].reshape(1, -1)
            out = norm_seq[i, -1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:, :-1], seq[:, -1]

    def prepare_data_puma32f(self):
        matfile = loadmat(os.path.join(self.data_dir, "puma32f.mat"))
        seq = matfile["nngc_data"]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq)):
            inp = norm_seq[i, :-1].reshape(1, -1)
            out = norm_seq[i, -1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:, :-1], seq[:, -1]

    def prepare_data_pumadyn(self):
        matfile = loadmat(os.path.join(self.data_dir, "puma3_data.mat"))
        seq = matfile["our_data"]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq)):
            inp = norm_seq[i, :-1].reshape(1, -1)
            out = norm_seq[i, -1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:, :-1], seq[:, -1]

    def prepare_data_alcoa(self):
        matfile = loadmat(os.path.join(self.data_dir, "fin_multi_data2.mat"))
        seq = matfile["nngc_data"]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq)-1):
            inp = norm_seq[i, :].reshape(1, -1)
            out = norm_seq[i+1, 0].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:, 0]  # todo check

    def prepare_data_euro(self):
        matfile = loadmat(os.path.join(self.data_dir, "euro_data2.mat"))

        seq = matfile["nngc_data"]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq)-1):
            inp = norm_seq[i, :].reshape(1, -1)
            out = norm_seq[i+1, 0].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:, 0]  # todo check

    def prepare_data_usdeur(self):
        return self.prepare_data_currency("USDEUR")

    def prepare_data_usdgbp(self):
        return self.prepare_data_currency("USDGBP")

    def prepare_data_usdchf(self):
        return self.prepare_data_currency("USDCHF")

    def prepare_data_usdjpy(self):
        return self.prepare_data_currency("USDJPY")

    def prepare_data_usdtry(self):
        return self.prepare_data_currency("USDTRY")

    def prepare_data_usdxau(self):
        return self.prepare_data_currency("USDXAU")

    def prepare_data_currency(self, cur_pair):
        data_df = pd.read_csv("data/USD__TRY_CHF_JPY_EUR_GBP_XAU.csv")
        data_df = data_df.dropna()
        data_df = data_df.drop(columns=[data_df.columns[0]])
        data_df["Date"] = pd.to_datetime(data_df["Date"])
        selected_columns = ["Date"] + [col for col in data_df.columns.values if "Close" in col]
        data_df = data_df[selected_columns]

        def construct_features(df):
            mean = df.resample("D", on="Date").mean()
            mean.columns = "mean_" + mean.columns.values

            # high = df.resample("D", on="Date").max()
            # high.columns = "high_" + high.columns.values
            #
            # low = df.resample("D", on="Date").min()
            # low.columns = "low_" + low.columns.values

            std = df.resample("D", on="Date").std()
            std.columns = "std_" + std.columns.values

            out_df = pd.concat([mean, std], axis=1)
            return out_df

        data_features = construct_features(data_df)
        drop_columns = [col for col in data_features.columns.values if "Date" in col]
        data_features = data_features.drop(columns=drop_columns)

        cols = [col for col in data_features.columns if cur_pair in col]
        seq = np.array(data_features[cols]).reshape((len(data_features), -1))

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq)-1):
            inp = norm_seq[i, :].reshape(1, -1)
            out = norm_seq[i+1, 0].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1, :], seq[1:, 0]  # todo check

    def prepare_data_sim_arima(self):
        data_path = "data/sim_arima.npy"
        if os.path.isfile(data_path):
            seq = np.load(data_path)
        else:
            ar_list = [[1], [-0.9]]
            std_list = [.1, .1]
            period_list = [1000] * 5
            ratio_list = [.5, .5]
            seq = simulate_ar_data(ar_list, std_list, period_list, ratio_list)
            np.save(data_path, seq)

        if self.diff_flag:
            seq[:-1] = seq[1:] - seq[:-1]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(seq) - 1):
            inp = norm_seq[i].reshape(1, -1)
            out = norm_seq[i + 1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:]

    def prepare_data_sim_arima_markov(self):
        data_path = "data/sim_arima_markov.pkl"
        if os.path.isfile(data_path):
            seq = pkl.load(open(data_path, 'rb'))
        else:
            ar_list = [[0.95, 0.5, -0.5], [0.9, -0.5, 0.5]]
            std_list = [.1, .1]
            psi = np.array([[0.998, 0.002], [0.004, 0.996]])
            seq_len = 5000
            seq = simulate_ar_markov_data(ar_list, std_list, seq_len, psi)
            pkl.dump(seq, open(data_path, 'wb'))
        seq = seq[0]

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(seq) - 1):
            inp = norm_seq[i].reshape(1, -1)
            out = norm_seq[i + 1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:]

    def prepare_data_sim_sin_markov(self):
        data_path = "data/sim_sin_markov.pkl"
        if os.path.isfile(data_path):
            seq = pkl.load(open(data_path, 'rb'))
        else:
            std_list = [0.1, 0.1]
            psi = np.array([[0.99, 0.01], [0.01, 0.99]])
            period_list = [50, 200]
            mag_list = [0.5, 0.5]
            seq_len = 5000
            seq = simulate_sin_markov_data(period_list, mag_list, std_list, seq_len, psi)
            pkl.dump(seq, open(data_path, 'wb'))
        seq = seq[0]

        if self.diff_flag:
            seq[:-1] = seq[1:] - seq[:-1]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(seq) - 1):
            inp = norm_seq[i].reshape(1, -1)
            out = norm_seq[i + 1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:]

    def prepare_data_sim_binary(self):
        operation_list = ["+", "-"]
        period_list = [50] * 1
        ratio_list = [.5, .5]
        seq = simulate_binary_data(operation_list, period_list, ratio_list)

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(norm_seq) - 1):
            inp = norm_seq[i, :-1].reshape(1, -1)
            out = norm_seq[i, -1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:, :-1], seq[:, -1]

    def prepare_data_sim_ts(self):
        trend_list = [-0., 0., 0.]
        seasonality_list = [[100], [500], [200]]
        std_list = [0.1, 0.1, 0.1]
        period_list = [2000] * 4
        ratio_list = [0.25, 0.5, 0.25]

        seq = simulate_ts_data(trend_list, seasonality_list, std_list, period_list, ratio_list)

        if self.diff_flag:
            seq[:-1, 0] = seq[1:, 0] - seq[:-1, 0]
            seq = seq[:-1]

        norm_seq = self.normalize_data(seq)
        if self.tensor_flag:
            norm_seq = torch.FloatTensor(norm_seq)

        data = []
        for i in range(len(seq) - 1):
            inp = norm_seq[i].reshape(1, -1)
            out = norm_seq[i + 1].reshape(1, -1)
            data.append((inp, out))

        return data, seq[:-1], seq[1:]

    def normalize_data(self, data):
        if self.scale_mode != "none":
            if len(data.shape) == 1:
                data = self.data_scaler.fit_transform(data.reshape(-1, 1))
            else:
                data = self.data_scaler.fit_transform(data)
        data[np.isnan(data)] = 0
        return data


if __name__ == '__main__':
    dataset_name = "elev"
    data = Data(dataset_name)
    print(len(data.data))