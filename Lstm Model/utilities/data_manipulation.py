import math
import pandas_datareader as pdr
from .visuals import plotting
import numpy as np


def get_data(data_set='AAPL', start='2014-01-01', end='2019-09-30', look_back=60):
    """
    Getting the desired data from yahoo, then doing some data manipulation
    such as plotting, train-test-split and data
    reshaping

    Args:
        (str) data_set - the ticker of desired dataset (company)
        (str) start - the start date of the desired dataset
        (str) end - the end date of the desired dataset
        (int) look_back - number of days the forecast is based on

    Returns:
        (np array) x_train, y_train : reshaped arrays to train the model with
        (np array) x_test - reshaped array to test the model with
        (np array) y_test - to validate the model on
        (int) training_data_len - the number to split the data with into train and test
        close_df - a data frame of the close price after resetting the index
    """
    # getting the data
    df = pdr.DataReader(data_set, data_source='yahoo', start=start, end=end)
    print("Data set shape:", df.shape)

    # plotting the data
    plotting(df['Close'], title="Close Price History", x_label='Date', y_label='Close Price US ($)')

    # creating a new df with Xt - Xt-1 values of the close prices
    close_df = df['2012-01-01':].reset_index()['Close']
    plotting(close_df, title="Close price trend ",
             x_label="Date", y_label="Price")

    close_diff = close_df.diff().dropna()
    # plotting the differenced data
    plotting(close_diff, "close price after data differencing", "Date")

    # splitting the data 80% for training and 20% for testing
    training_data_len = math.ceil(len(close_diff) * 0.8)
    data = np.array(close_diff).reshape(-1, 1)

    # for doing a forecast based on a specific period (in days),,,
    # creating x_train having the first specified period data in column 1 and so on
    # creating y_train having the value of the day next to the specified period
    train_data = data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i - look_back:i, 0])
        y_train.append(train_data[i, 0])
    print("Training set shape:", train_data.shape)

    # converting x_train, y_train to np arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshaping the data to 3D to be accepted by our LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # creating the test data set
    test_data = data[training_data_len - look_back:, :]
    x_test = []
    y_test = data[training_data_len:, :]
    print("Testing set shape:", test_data.shape)
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - look_back:i, 0])

    # converting x_test to be a 3D np array to be accepted by the LSTM model
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, training_data_len, close_df
