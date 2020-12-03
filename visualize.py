import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_trainer(model_name, dataset_name, scale_mode):
    path = os.path.join("output", model_name + "_" + "_" + dataset_name + "_" + scale_mode + ".pkl")
    trainer = pickle.load(open(path, "rb"))
    return trainer


def plot_loss(key_list, loss_array_list, window_len=None, poly_deg=None, save_path=None):
    plt.figure(figsize=(10, 10))

    for loss_array in loss_array_list:
        loss_array = refine_array(loss_array, window_len, poly_deg)
        plt.plot(loss_array)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(key_list)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def refine_array(array, window_len=0, poly_deg=0):
    if window_len > 0:
        array = np.convolve(array, np.ones(window_len) / window_len, mode="valid")
    if poly_deg > 0:
        array = np.poly1d(np.polyfit(np.arange(len(array)), array, poly_deg))(np.arange(len(array)))
    return array


def gather_loss_array_list(model_name_list, dataset_name_list, scale_mode):
    loss_mean_array_list = []
    loss_std_array_list = []
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            trainer = load_trainer(model_name, dataset_name, scale_mode)
            loss_list = np.array([trainer.results_dict[idx]["results_mean"]
                                  [trainer.config_obj.experiment_params["evaluation_metric"][0]]
                                  for idx in range(len(trainer.results_dict))])
            loss_list[np.isnan(loss_list)] = 1e9
            best_conf_idx = np.argmin(loss_list)
            best_loss_mean = trainer.results_dict[best_conf_idx]["loss_array_mean"]
            best_loss_std = trainer.results_dict[best_conf_idx]["loss_array_std"]
            loss_mean_array_list.append(best_loss_mean)
            loss_std_array_list.append(best_loss_std)

    return loss_mean_array_list, loss_std_array_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=["rnn"], nargs="+")  # "rnn"
    parser.add_argument('--optimizer', type=str, default=["sgd"], nargs="+")  # "sgd", "rmsprop", "adam"
    parser.add_argument('--dataset', type=str, default=["elev"], nargs="+")  # string
    parser.add_argument('--normalization', type=str, default="minmax")
    parser.add_argument('--save', type=bool, default=1)  # save results flag
    parser.add_argument('--window_len', type=int, default=151)  # smoothing window size
    parser.add_argument('--poly_deg', type=int, default=0)  # poly_fit degree

    args = parser.parse_args()

    model_name_list = args.model
    dataset_name_list = args.dataset
    scale_mode = args.normalization
    window_len = args.window_len
    poly_deg = args.poly_deg

    loss_mean_array_list, loss_std_array_list = \
        gather_loss_array_list(model_name_list, dataset_name_list, scale_mode)

    plot_loss(loss_mean_array_list, window_len=window_len, poly_deg=poly_deg)