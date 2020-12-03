import numpy as np
import matplotlib.pyplot as plt


class ARIMASimulator:
    def __init__(self):
        self.data = []

    def generate_samples(self, ar, std, num_samples, ma=None, diff=None):
        if ma is not None or diff is not None:
            raise NotImplementedError

        out = []
        for i in range(num_samples):
            new_sample = np.random.randn() * std
            for j in range(len(ar)):
                if len(self.data) < j + 1:
                    this_sample = 0
                else:
                    this_sample = self.data[-(j+1)]
                new_sample += ar[j] * this_sample
            out.append(new_sample)
            self.data.append(new_sample)

        return out

    def reset(self):
        self.data = []


class SinusoidalSimulator:
    def __init__(self):
        self.raw_data = []
        self.data = []
        self.dir = 1

    def generate_samples(self, period, mag, std, num_samples):
        out = []

        val = 0
        if self.raw_data:
            val = self.raw_data[-1]

        theta = np.arcsin(val)
        if not self.dir:
            theta = np.pi - theta

        tau = np.round(theta * period / (2 * np.pi))
        for i in range(num_samples):
            noise = np.random.randn() * std
            new_sample = mag * np.sin(2 * np.pi * (i + tau) / period)
            out.append(new_sample + noise)
            self.data.append(new_sample + noise)
            next_sample = np.sin(2 * np.pi * (i + tau + 1) / period)
            self.raw_data.append(next_sample)

        if next_sample < 0:
            if next_sample >= new_sample / mag:
                self.dir = 1
            else:
                self.dir = 0
        else:
            if next_sample > new_sample / mag:
                self.dir = 1
            else:
                self.dir = 0

        return out

    def reset(self):
        self.data = []


class BinarySimulator:
    def __init__(self):
        self.data = []

    def generate_samples(self, num_samples, operation):
        sequence = np.random.randint(low=0, high=2, size=(num_samples, 2))
        int_1 = int("".join([str(x_) for x_ in np.flip(sequence[:, 0]).tolist()]), 2)
        int_2 = int("".join([str(x_) for x_ in np.flip(sequence[:, 1]).tolist()]), 2)

        if operation == "+":
            target = [int(x_) for x_ in str(int(bin(abs(int_1 + int_2))[2:]))]
        elif operation == "-":
            target = [int(x_) for x_ in str(int(bin(abs(int_1 - int_2))[2:]))]
        else:
            raise NotImplementedError

        target = np.flip(np.array(target))
        if len(target) > sequence.shape[0]:
            target = target[:-1]
        elif len(target) < sequence.shape[0]:
            target = np.array(target.tolist() + [0] * (sequence.shape[0] - len(target)))

        sequence = np.concatenate([sequence, target[:, None]], axis=1)

        return sequence


class TSSimulator:
    def __init__(self):
        self.data = []

    def generate_samples(self, num_samples, trend, periods, std):
        x = np.arange(num_samples)

        trend_component = x * trend
        if self.data:
            trend_component += self.data[-1]

        seasonality_component = np.zeros(num_samples)
        for period in periods:
            seasonality_component += np.cos(2 * np.pi * x / period)

        out = trend_component + seasonality_component + np.random.randn(num_samples) * std
        self.data.extend(out)

        return out


def simulate_ar_data(ar_list, std_list, period_list, ratio_list):
    assert len(ar_list) == len(ratio_list)
    num_modes = len(ar_list)
    simulator = ARIMASimulator()

    x = []
    for period in period_list:
        split_lens = [int(ratio * period) for ratio in ratio_list]
        for i in range(num_modes):
            x.extend(simulator.generate_samples(ar=ar_list[i], std=std_list[i], num_samples=split_lens[i]))

    return np.array(x)


def simulate_ar_markov_data(ar_list, std_list, seq_len, psi):
    simulator = ARIMASimulator()

    x = []
    state = 0
    state_list = []

    for i in range(seq_len):
        state = np.where(np.random.multinomial(1, psi[state, :]))[0][0]
        state_list.append(state)
        x.extend(simulator.generate_samples(ar=ar_list[state], std=std_list[state], num_samples=1))

    return [np.array(x), state_list]


def simulate_sin_markov_data(period_list, mag_list, std_list, seq_len, psi):
    simulator = SinusoidalSimulator()

    x = []
    state = 0
    state_list = []

    for i in range(seq_len):
        state = np.where(np.random.multinomial(1, psi[state, :]))[0][0]
        state_list.append(state)
        x.extend(simulator.generate_samples(period=period_list[state], mag=mag_list[state], std=std_list[state], num_samples=1))

    return [np.array(x), state_list]


def simulate_ts_data(trend_list, seasonality_list, std_list, period_list, ratio_list):
    assert len(trend_list) == len(seasonality_list) == len(ratio_list)
    num_modes = len(trend_list)
    simulator = TSSimulator()

    x = []
    for period in period_list:
        split_lens = [int(ratio * period) for ratio in ratio_list]
        for i in range(num_modes):
            x.extend(simulator.generate_samples(num_samples=split_lens[i], trend=trend_list[i],
                                                periods=seasonality_list[i], std=std_list[i]))

    return np.array(x)


def simulate_binary_data(operation_list, period_list, ratio_list):
    assert len(operation_list) == len(ratio_list)
    num_modes = len(operation_list)
    simulator = BinarySimulator()

    x = []
    for period in period_list:
        split_lens = [int(ratio * period) for ratio in ratio_list]
        for i in range(num_modes):
            x.extend(simulator.generate_samples(operation=operation_list[i], num_samples=split_lens[i]))

    return np.array(x)


if __name__ == '__main__':
    mode = "sin"
    if mode == "arima":
        ar_list = [[1], [.9], [.5]]
        std_list = [.1, .1, .1]
        period_list = [1000] * 3
        ratio_list = [.34, .33, .33]
        x = simulate_ar_data(ar_list, std_list, period_list, ratio_list)
        plt.plot(np.concatenate([x]))
        plt.show()
    elif mode == "binary":
        operation_list = ["+", "-"]
        period_list = [10] * 1
        ratio_list = [.5, .5]
        x = simulate_binary_data(operation_list, period_list, ratio_list)
    elif mode == "sin":
        std_list = [0.01, 0.01]
        psi = np.array([[0.99, 0.01], [0.01, 0.99]])
        period_list = [50, 200]
        mag_list = [0.5, 0.5]
        seq_len = 5000
        x = simulate_sin_markov_data(period_list, mag_list, std_list, seq_len, psi)
        plt.figure(figsize=(20, 8))
        plt.plot(x[0])
        plt.show()
    else:
        raise NotImplementedError

