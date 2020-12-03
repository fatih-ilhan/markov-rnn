import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import numpy as np

from data_obj import Data
from utils.evaluation_utils import mape


for model in ["tvtp"]:
    print(model)
    dataset = "sim_arima_markov"
    data = Data(dataset_name=dataset, tensor_flag=False, diff_flag=False)

    X = data.data
    train_end_idx = int(len(X) * 0.6)
    train, test = X[0:train_end_idx], X[train_end_idx:]
    predictions = list()

    if model == "arima":
        history = [x[0][0][0] for x in train]
        train_model = sm.tsa.SARIMAX(history, order=(3, 0, 0))
    elif model == "hamilton":
        history = [x[0][0][0]for x in train]
        train_model = sm.tsa.MarkovAutoregression(history, k_regimes=2, order=3, switching_ar=True)
    elif model == "kns":
        history = [x[0][0][0] for x in train]
        train_model = sm.tsa.MarkovRegression(history, k_regimes=2, trend='nc', switching_variance=True)
    elif model == "tvtp":
        history = [x[0][0][0] for x in train]
        exog_tvtp = np.stack([np.ones_like(history), np.arange(len(history))], axis=1)
        train_model = sm.tsa.MarkovAutoregression(history, k_regimes=2, order=3, switching_ar=True, exog_tvtp=exog_tvtp)

    train_model_fit = train_model.fit(disp=0)

    for t in tqdm(range(len(test))):
        if t >= len(test) / 2:

            if model == "arima":
                test_model = sm.tsa.SARIMAX(history, order=(1, 0, 0))
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.forecast()[0]
            elif model == "hamilton":
                test_model = sm.tsa.MarkovAutoregression(history, k_regimes=2, order=3, switching_ar=True)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]
            elif model == "kns":
                test_model = sm.tsa.MarkovRegression(history, k_regimes=2, trend='nc', switching_variance=True)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]
            elif model == "tvtp":
                exog_tvtp = np.stack([np.ones_like(history), np.arange(len(history))], axis=1)
                test_model = sm.tsa.MarkovAutoregression(history, k_regimes=2, order=3, switching_ar=True,
                                                         exog_tvtp=exog_tvtp)
                test_model_fit = test_model.filter(train_model_fit.params)
                yhat = test_model_fit.predict()[-1]

            predictions.append(yhat)
        obs = test[t][0][0][0]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    predictions = np.array(predictions)
    test_len = int(len(X) * 0.2)
    idxs = np.logical_and(~np.isnan(predictions), ~np.isinf(predictions))
    mse_error = mean_squared_error([x[0][0][0] for i, x in enumerate(test[-test_len:]) if i in np.nonzero(idxs)[0]], predictions[idxs][:-1])
    print('Test MSE: %.9f' % mse_error)
    mae_error = mean_absolute_error([x[0][0][0] for i, x in enumerate(test[-test_len:]) if i in np.nonzero(idxs)[0]], predictions[idxs][:-1])
    print('Test MAE: %.9f' % mae_error)
# mape_error = mape([x[0][0][0] for x in test[-test_len:]], predictions[-test_len:])
# print('Test MAPE: %.9f' % mape_error)

# plot
# plt.plot(test[-test_len:])
# plt.plot(predictions, color='red')
# plt.show()