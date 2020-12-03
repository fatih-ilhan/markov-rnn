from config import Config
from data_obj import Data
import statsmodels.api as sm
import numpy as np


dataset_name = "USDEUR"
scale_mode = "none"
switch_mode_list = ["kns"]
data = Data(dataset_name, scale_mode=scale_mode)

labels = data.output_series
data_len = len(labels)

train_val_test_ratio = [0.6, 0.2, 0.2]
val_start_idx = int(data_len * train_val_test_ratio[0])
test_start_idx = int(data_len * sum(train_val_test_ratio[:2]))


if "hamilton" in switch_mode_list:
    mod_hamilton = sm.tsa.MarkovAutoregression(labels, k_regimes=2, order=4, switching_ar=False)
    res_hamilton = mod_hamilton.fit()


if "kns" in switch_mode_list:
    mod_kns = sm.tsa.MarkovRegression(labels, k_regimes=2, trend='nc', switching_variance=True)
    res_kns = mod_kns.fit()


if "filardo" in switch_mode_list:
    exog_tvtp = np.stack([np.ones_like(labels), np.arange(labels.shape[0])], axis=1)
    mod_filardo = sm.tsa.MarkovAutoregression(labels, k_regimes=2, order=4, switching_ar=False, exog_tvtp=exog_tvtp)
    res_filardo = mod_filardo.fit(search_reps=20)