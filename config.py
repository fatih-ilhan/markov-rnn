import itertools
from random import shuffle

model_params_pool = {"rnn": {"hidden_dim": [16],
                             "bias": [[0, 0]],
                             "trunc_len": [8],
                             "window_len": [1],
                             "lr": [0.01],
                             "weight_decay": [0],
                             "optimizer": ["sgd"],
                             "shuffle_flag": [False]},
                     "markov_rnn": {"hidden_dim": [8],
                                    "bias": [[0, 0]],
                                    "alpha": [0.7],  # psi diagonal
                                    "beta": [0.99],
                                    "trunc_len": [8],
                                    "num_state": [2],
                                    "lr": [3e-3],
                                    "weight_decay": [0],
                                    "optimizer": ["adam"]},
                     "lstm": {"hidden_dim": [8, 32],
                              "bias": [[0, 0]],
                              "trunc_len": [4, 16],
                              "window_len": [1],
                              "lr": [0.003, 0.01, 0.03],
                              "weight_decay": [0],
                              "optimizer": ["sgd"],
                              "shuffle_flag": [False]},
                     "markov_lstm": {"hidden_dim": [32],
                                     "bias": [[0, 0]],
                                     "alpha": [0.7],  # psi diagonal
                                     "beta": [0.99],
                                     "trunc_len": [8],
                                     "num_state": [2],
                                     "lr": [3e-3],
                                     "weight_decay": [0],
                                     "optimizer": ["adam"]},
                     "gru": {"hidden_dim": [8, 32],
                             "bias": [[0, 0]],
                             "trunc_len": [4, 16],
                             "window_len": [1],
                             "lr": [0.003, 0.01, 0.03],
                             "weight_decay": [0],
                             "optimizer": ["sgd"],
                             "shuffle_flag": [False]},
                     "markov_gru": {"hidden_dim": [32],
                                    "bias": [[0, 0]],
                                    "alpha": [0.7],  # psi diagonal
                                    "beta": [0.99],
                                    "trunc_len": [8],
                                    "num_state": [2],
                                    "lr": [3e-3],
                                    "weight_decay": [0],
                                    "optimizer": ["adam"]}
                     }


# TODO : bugs in online mode - state continuity
class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self, model_name):
        self.experiment_params = {"evaluation_metric": ["mean_squared_error", "mae", "mape"],
                                  "diff_flag": False,
                                  "scale_mode": "minmax",
                                  "online_flag": False,
                                  "num_epochs": 30,
                                  "early_stop_tolerance": 5,
                                  "train_val_test_ratio": [0.6, 0.2, 0.2]}

        assert len(self.experiment_params["train_val_test_ratio"]) == 3 and \
                   sum(self.experiment_params["train_val_test_ratio"]) == 1

        self.model_name = model_name
        self.model_params = model_params_pool[self.model_name]

        self.conf_list = self.create_params_list(dict(**self.model_params))
        self.num_confs = len(self.conf_list)

    def next(self):
        for conf in self.conf_list:
            yield conf

    @staticmethod
    def create_params_list(pool):
        params_list = []
        keys = pool.keys()
        lists = [l for l in pool.values()]
        all_lists = list(itertools.product(*lists))
        for i in range(len(all_lists)):
            params_list.append(dict(zip(keys, all_lists[i])))
        shuffle(params_list)
        return params_list