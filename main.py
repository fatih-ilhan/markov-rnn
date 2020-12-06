import argparse
import pickle
import os
import time
import multiprocessing as mp
import numpy as np
import torch

from config import Config
from data_obj import Data
from trainer import Trainer


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=["rnn"], nargs="+")  # "rnn", "markov_rnn"
    parser.add_argument('--dataset', type=str, default=["elev"], nargs="+")  # string
    parser.add_argument('--num_repeat', type=int, default=1)  # number of repeats
    parser.add_argument('--num_process', type=int, default=mp.cpu_count())  # number of processes
    parser.add_argument('--save', type=int, default=1)  # save results flag
    parser.add_argument('--verbose', type=int, default=1)  # print info flag
    parser.add_argument('--device', type=str, default="cpu")  # device cpu/cuda

    args = parser.parse_args()

    model_name_list = args.model
    dataset_name_list = args.dataset
    num_repeat = args.num_repeat
    num_process = args.num_process
    save = args.save
    verbose = args.verbose
    device = args.device

    num_process = min(num_process, mp.cpu_count())

    for model_name in model_name_list:
        config_obj = Config(model_name)
        scale_mode = config_obj.experiment_params["scale_mode"]

        for dataset_name in dataset_name_list:

            print(model_name, dataset_name)
            data = Data(dataset_name, scale_mode=scale_mode, tensor_flag=True, diff_flag=config_obj.experiment_params["diff_flag"])
            save_path = model_name + "_" + dataset_name + "_" + scale_mode

            trainer = Trainer(config_obj, model_name, dataset_name, data, device=device)
            train_results = trainer.grid_search(data, model_name, num_repeat=num_repeat, num_process=num_process, verbose=verbose, **config_obj.experiment_params)
            trainer.labels = data.output_series
            trainer.data_obj = None

            if save:
                key = np.random.randint(low=0, high=999999999)
                key = str(key).zfill(0)

                pickle.dump(trainer, open(os.path.join("results", str(key) + "_" + save_path + ".pkl"), "wb"))

    print(f"Took {time.time() - start_time} seconds")
