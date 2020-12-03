import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
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

print("test MSE:", ((test_labels-test_preds)**2).mean())
print("test MAE:", (abs(test_labels-test_preds).mean()))
print("test MAPE:", (mape(test_labels, test_preds)))

print("test MSE lin:", ((test_labels-test_preds_lin.squeeze())**2).mean())
print("test MAE lin:", (abs(test_labels-test_preds_lin.squeeze()).mean()))
print("test MAPE lin:", (mape(test_labels, test_preds_lin.squeeze())))

print("test MSE iso:", ((test_labels-test_preds_iso.squeeze())**2).mean())
print("test MAE iso:", (abs(test_labels-test_preds_iso.squeeze()).mean()))
print("test MAPE iso:", (mape(test_labels, test_preds_iso.squeeze())))