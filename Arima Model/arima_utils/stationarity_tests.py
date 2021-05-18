# Import Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm


# Check Stationary by many methods like diff , log
def rolling_stats(ts_data):
    roll_mean = ts_data.rolling(30).mean()
    roll_std = ts_data.rolling(5).std()

    # Plot rolling statistics
    # fig = plt.figure(figsize=(20, 10))
    # plt.subplot(211)
    # plt.plot(ts_data, color='black', label='Original Data')
    # plt.plot(roll_mean, color='red', label='Rolling Mean(30 days)')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(roll_std, color='green', label='Rolling Std Dev(5 days)')
    # plt.legend()


def dickey_fuller(ts_data):
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '# of lags', '# of obs'])
    print(test_result)
    for k, v in df_test[4].items():
        print('Critical value at %s: %1.5f' % (k, v))


def log_diff(ts_data):
    df_final_log = np.log(ts_data)
    df_final_log_diff = df_final_log - df_final_log.shift()
    df_final_log_diff.dropna(inplace=True)
    # plt.figure(figsize=(10, 6))
    # plt.plot(df_final_log_diff)
    # plt.title("data after log differencing")
    # plt.show()


def auto_correlation(ts_data):
    df_final_diff = ts_data - ts_data.shift()
    df_final_diff.dropna(inplace=True)
    df_acf = acf(df_final_diff)

    # Partial autocorrelation function
    df_pacf = pacf(df_final_diff)

    # visualizing acf and pacf
    # fig1 = plt.figure(figsize=(20, 10))
    # ax1 = fig1.add_subplot(211)
    # fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)
    # ax2 = fig1.add_subplot(212)
    # fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)

