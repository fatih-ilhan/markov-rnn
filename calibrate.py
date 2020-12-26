import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from utils.evaluation_utils import mape


trainer = pkl.load(open("results/14947622_markov_rnn_USDGBP_none.pkl", "rb"))
val_preds = trainer.results_dict[trainer.best_conf_idx]["val_pred_list"][0].squeeze()
val_labels = trainer.labels[-2*val_preds.shape[0]:-val_preds.shape[0]]
test_preds = trainer.results_dict[trainer.best_conf_idx]["test_pred_list"][0].squeeze()
test_labels = trainer.labels[-test_preds.shape[0]:]

linear_reg = LinearRegression().fit(val_preds.reshape(-1, 1), val_labels.reshape(-1, 1))
isotonic_reg = IsotonicRegression().fit(val_preds, val_labels)

test_preds_lin = linear_reg.predict(test_preds.reshape(-1, 1))
test_preds_iso = isotonic_reg.predict(test_preds)

print("test MSE:", mean_squared_error(test_labels, test_preds))
print("test MAE:", mean_absolute_error(test_labels, test_preds))
print("test MAPE:", mean_absolute_percentage_error(test_labels, test_preds))

print("test MSE lin:", mean_squared_error(test_labels, test_preds_lin))
print("test MAE lin:", mean_absolute_error(test_labels, test_preds_lin))
print("test MAPE lin:", mean_absolute_percentage_error(test_labels, test_preds_lin))

print("test MSE iso:", mean_squared_error(test_labels, test_preds_iso))
print("test MAE iso:", mean_absolute_error(test_labels, test_preds_iso))
print("test MAPE iso:", mean_absolute_percentage_error(test_labels, test_preds_iso))