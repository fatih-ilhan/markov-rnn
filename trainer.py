import json
import numpy as np
import torch
from multiprocessing import Process, Manager
from tqdm import tqdm
from copy import deepcopy
import matplotlib as mpl

from utils.evaluation_utils import evaluate, merge_results
from models.rnn import RNN
from models.markov_rnn import MarkovRNN


class Trainer:
    def __init__(self, config_obj, model_name, dataset_name, data_obj, device="cpu", num_jobs=1):

        self.config_obj = config_obj
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_obj = data_obj
        self.device = device

        self.results_dict = {}
        self.models_dict = {}
        self.best_conf_idx = None
        self._is_fit = False

        self.model_dispatcher = {"rnn": RNN, "markov_rnn": MarkovRNN, "lstm": RNN, "markov_lstm": MarkovRNN,
                                 "gru": RNN, "markov_gru": MarkovRNN}

    def grid_search(self, data, model_name, num_repeat=1, num_process=4, verbose=True, **experiment_params):

        num_confs = self.config_obj.num_confs  # 50
        num_sub_confs = int(np.ceil(num_confs / num_process))  # 50 / 4 -> 13

        conf_idx = 0
        for i in range(num_sub_confs):
            sub_conf_list = self.config_obj.conf_list[i * num_process: (i+1) * num_process]  # 0:4 .... 48:52

            manager = Manager()
            results_dict = manager.dict()
            models_dict = manager.dict()
            process_list = []

            for sub_conf_idx, conf in enumerate(sub_conf_list):
                conf["cell_type"] = model_name.split("_")[-1]
                args = (data, conf_idx, conf, num_repeat, num_confs, verbose, experiment_params, results_dict, models_dict)
                conf_idx += 1
                self.grid_search_worker(*args)
            #     p = Process(target=self.grid_search_worker, args=args)
            #     process_list.append(p)
            #     p.start()
            #
            # for p in process_list:
            #     p.join()

            self.results_dict.update(dict(results_dict))
            self.models_dict.update(dict(models_dict))

        self._is_fit = True

        loss_list = np.array([self.results_dict[idx]["val_results"]["results_mean"]
                              [self.config_obj.experiment_params["evaluation_metric"][0]]
                              for idx in range(len(self.config_obj.conf_list))])
        self.best_conf_idx = np.argmin(loss_list)
        best_loss = loss_list[self.best_conf_idx]
        print(f"Validation: Best average loss: {best_loss} achieved with {self.config_obj.conf_list[self.best_conf_idx]}")

        return self.results_dict

    def grid_search_worker(self, data, conf_idx, conf, num_repeat, num_confs, verbose, experiment_params, results_dict, models_dict):
        results_dict_ = {}
        models_dict_ = {}

        input_dim = data.input_dim
        output_dim = data.output_dim
        data_pair = data.data
        offset = data.offset

        train_results_dict = {}
        train_conf_results_list = []
        train_conf_loss_list = []
        all_train_pred_list = []

        val_results_dict = {}
        val_conf_results_list = []
        val_conf_loss_list = []
        all_val_pred_list = []
        
        test_results_dict = {}
        test_conf_results_list = []
        test_conf_loss_list = []
        all_test_pred_list = []

        online_flag = experiment_params["online_flag"]
        num_epochs = experiment_params["num_epochs"]
        train_val_test_ratio = experiment_params["train_val_test_ratio"]
        evaluation_metric = self.config_obj.experiment_params["evaluation_metric"]

        data_len = len(data_pair)
        if online_flag:
            split_indices = [int(data_len * train_val_test_ratio[0]), data_len]
        else:
            split_indices = [int(data_len * train_val_test_ratio[0]), int(data_len * sum(train_val_test_ratio[:2]))]

        for rep in range(num_repeat):
            best_model = None

            best_train_results = None
            best_train_loss_array = None
            best_train_pred = None

            best_val_results = None
            best_val_loss_array = None
            best_val_pred = None
            best_val_loss = None

            tolerance = 0

            model = self.model_dispatcher[self.model_name](input_dim, output_dim, device=self.device, **conf).to(
                self.device)
            train_data = data_pair[:split_indices[0]]
            val_data = data_pair[split_indices[0]: split_indices[1]]
            test_data = data_pair[split_indices[1]:]

            for epoch in range(num_epochs):
                try:
                    if verbose:
                        print("********************")
                        print(f"Configuration: {conf_idx + 1}/{num_confs} | Repeat index: {rep + 1}/{num_repeat} "
                              f"| Epoch: {epoch + 1}/{num_epochs}")
                        print(conf)

                    train_pred, train_loss_list = model.fit(train_data, verbose=verbose)
                    if hasattr(self.data_obj, "pred_scaler"):
                        train_pred = self.data_obj.pred_scaler.inverse_transform(np.stack(train_pred).reshape(-1, 1))
                    else:
                        train_pred = np.stack(train_pred).reshape(-1, 1)
                    train_label = data.output_series[split_indices[0] - len(train_pred): split_indices[0]].reshape(-1,1)
                    train_loss_array = np.array(train_loss_list)

                    if data.diff_flag:
                        train_pred = offset + np.cumsum(train_label) - train_label[:, 0] + train_pred[:, 0]
                        train_label = offset + np.cumsum(train_label)

                    train_results = evaluate(train_pred, train_label, evaluation_metric)

                    if online_flag:
                        state_dict = deepcopy(model.state_dict())
                        try:
                            op_dict = deepcopy(model.optimizer.state_dict())
                        except:
                            pass
                        val_pred, val_loss_list = model.fit(val_data, verbose=verbose)
                        model.load_state_dict(state_dict)
                        try:
                            model.optimizer.load_state_dict(op_dict)
                        except:
                            pass
                    else:
                        val_pred, val_loss_list = model.predict(val_data)

                    if hasattr(self.data_obj, "pred_scaler"):
                        val_pred = self.data_obj.pred_scaler.inverse_transform(np.stack(val_pred).reshape(-1, 1))
                    else:
                        val_pred = np.stack(val_pred).reshape(-1, 1)
                    val_label = data.output_series[split_indices[1]-len(val_pred): split_indices[1]].reshape(-1, 1)
                    val_loss_array = np.array(val_loss_list)

                    if data.diff_flag:
                        val_pred = offset + np.cumsum(val_label) + train_label[-1] - val_label[:, 0] + val_pred[:, 0]
                        val_label = offset + np.cumsum(val_label) + train_label[-1]

                    val_results = evaluate(val_pred, val_label, evaluation_metric)

                    print("Train results: ", train_results)
                    print("Validaton results: ", val_results)

                    if best_val_loss is None or val_results[evaluation_metric[0]] < best_val_loss:
                        tolerance = 0
                        best_val_loss = val_results[evaluation_metric[0]]
                        best_model = model
                        best_train_results = train_results
                        best_train_loss_array = train_loss_array
                        best_train_pred = train_pred
                        best_val_results = val_results
                        best_val_loss_array = val_loss_array
                        best_val_pred = val_pred

                    if best_val_loss is None or val_results[evaluation_metric[0]] > 0.999 * best_val_loss:
                        tolerance += 1

                    if tolerance > self.config_obj.experiment_params["early_stop_tolerance"] or epoch == num_epochs - 1:
                        print(f"Exiting at epoch {epoch+1}")
                        models_dict_[rep] = best_model
                        train_conf_results_list.append(best_train_results)
                        train_conf_loss_list.append(best_train_loss_array)
                        all_train_pred_list.append(best_train_pred)
                        val_conf_results_list.append(best_val_results)
                        val_conf_loss_list.append(best_val_loss_array)
                        all_val_pred_list.append(best_val_pred)
                        break

                except Exception as error:
                    print(conf)
                    raise error

            # Test
            if online_flag:
                test_pred, test_loss_list = model.fit(test_data, verbose=verbose)
            else:
                test_pred, test_loss_list = model.predict(test_data)

            if hasattr(self.data_obj, "pred_scaler"):
                test_pred = self.data_obj.pred_scaler.inverse_transform(np.stack(test_pred).reshape(-1, 1))
            else:
                test_pred = np.stack(test_pred).reshape(-1, 1)

            test_label = data.output_series[- len(test_pred):].reshape(-1, 1)
            test_loss_array = np.array(test_loss_list)

            if data.diff_flag:
                test_pred = offset + np.cumsum(test_label) + val_label[-1] - test_label[:, 0] + test_pred[:, 0]
                test_label = offset + np.cumsum(test_label) + val_label[-1]

            test_results = evaluate(test_pred, test_label, evaluation_metric)

            test_conf_results_list.append(test_results)
            test_conf_loss_list.append(test_loss_array)
            all_test_pred_list.append(test_pred)

        models_dict[conf_idx] = {}  # TODO : can't move models between subprocesses

        train_results_mean = merge_results(train_conf_results_list, "mean")
        train_results_std = merge_results(train_conf_results_list, "std")
        train_loss_array_mean = np.stack(train_conf_loss_list, axis=0).mean(axis=0)
        train_loss_array_std = np.stack(train_conf_loss_list, axis=0).std(axis=0)

        val_results_mean = merge_results(val_conf_results_list, "mean")
        val_results_std = merge_results(val_conf_results_list, "std")
        val_loss_array_mean = np.stack(val_conf_loss_list, axis=0).mean(axis=0)
        val_loss_array_std = np.stack(val_conf_loss_list, axis=0).std(axis=0)
        
        test_results_mean = merge_results(test_conf_results_list, "mean")
        test_results_std = merge_results(test_conf_results_list, "std")
        test_loss_array_mean = np.stack(test_conf_loss_list, axis=0).mean(axis=0)
        test_loss_array_std = np.stack(test_conf_loss_list, axis=0).std(axis=0)

        if verbose:
            print("\nTrain results (mean):", json.dumps(train_results_mean, indent=4))
            print("Train results (std):", json.dumps(train_results_std, indent=4))
            print("\nValidation results (mean):", json.dumps(val_results_mean, indent=4))
            print("Validation results (std):", json.dumps(val_results_std, indent=4))
            print("\nTest results (mean):", json.dumps(test_results_mean, indent=4))
            print("Test results (std):", json.dumps(test_results_std, indent=4))

        train_results_dict["results_mean"] = train_results_mean
        train_results_dict["results_std"] = train_results_std
        train_results_dict["loss_array_mean"] = train_loss_array_mean
        train_results_dict["loss_array_std"] = train_loss_array_std

        val_results_dict["results_mean"] = val_results_mean
        val_results_dict["results_std"] = val_results_std
        val_results_dict["loss_array_mean"] = val_loss_array_mean
        val_results_dict["loss_array_std"] = val_loss_array_std
        
        test_results_dict["results_mean"] = test_results_mean
        test_results_dict["results_std"] = test_results_std
        test_results_dict["loss_array_mean"] = test_loss_array_mean
        test_results_dict["loss_array_std"] = test_loss_array_std
        
        results_dict_.update({"train_results": train_results_dict})
        results_dict_.update({"val_results": val_results_dict})
        results_dict_.update({"test_results": test_results_dict})
        
        results_dict_.update({"train_pred_list": all_train_pred_list})
        results_dict_.update({"val_pred_list": all_val_pred_list})
        results_dict_.update({"test_pred_list": all_test_pred_list})

        results_dict[conf_idx] = results_dict_
