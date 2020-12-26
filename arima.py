import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data_obj import Data


pred_dict = {}

dataset = "sales"
data = Data(dataset_name=dataset, tensor_flag=False, diff_flag=False)

for model in ["arima"]:
    print(model)

    p = 4
    k = 2

    X = data.data
    train_end_idx = int(len(X) * 0.6)
    train, test = X[0:train_end_idx], X[train_end_idx:]
    predictions = list()
    history = [x[0][0][0] for x in train]

    if model == "arima":
        train_model = sm.tsa.SARIMAX(history, order=(p, 0, 0))
    elif model == "hamilton":
        train_model = sm.tsa.MarkovAutoregression(history, k_regimes=k, order=p, switching_ar=True)
    elif model == "kns":
        train_model = sm.tsa.MarkovRegression(history, k_regimes=k, trend='nc', switching_variance=True)
    elif model == "tvtp":
        exog_tvtp = np.stack([np.ones_like(history), np.arange(len(history))], axis=1)
        train_model = sm.tsa.MarkovAutoregression(history, k_regimes=k, order=p, switching_ar=True, exog_tvtp=exog_tvtp)
    else:
        raise NotImplementedError

    train_model_fit = train_model.fit(disp=0)

    for t in tqdm(range(len(test))):
        if t >= len(test) / 2:

            if model == "arima":
                test_model = sm.tsa.SARIMAX(history, order=(p, 0, 0))
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.forecast()[0]
            elif model == "hamilton":
                test_model = sm.tsa.MarkovAutoregression(history, k_regimes=k, order=p, switching_ar=True)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]
            elif model == "kns":
                test_model = sm.tsa.MarkovRegression(history, k_regimes=k, trend='nc', switching_variance=True)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]
            elif model == "tvtp":
                exog_tvtp = np.stack([np.ones_like(history), np.arange(len(history))], axis=1)
                test_model = sm.tsa.MarkovAutoregression(history, k_regimes=k, order=p, switching_ar=True,
                                                         exog_tvtp=exog_tvtp)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]

            predictions.append(yhat)
        obs = test[t][0][0][0]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    predictions = np.array(predictions)
    idxs = np.logical_and(~np.isnan(predictions), ~np.isinf(predictions))
    targets = np.array(history[-len(predictions):])

    predictions = predictions[idxs]
    targets = targets[idxs]

    print(idxs.mean())

    mse_error = mean_squared_error(targets, predictions)
    print('Test MSE: %.5f' % mse_error)
    print('Test RMSE: %.5f' % np.sqrt(mse_error))

    mae_error = mean_absolute_error(targets, predictions)
    print('Test MAE: %.5f' % mae_error)

    mape_error = mean_absolute_percentage_error(targets, predictions)
    print('Test MAPE: %.5f' % mape_error)

    pred_dict[model] = predictions

# plot
# plt.plot(test[-test_len:])
# plt.plot(predictions, color='red')
# plt.show()