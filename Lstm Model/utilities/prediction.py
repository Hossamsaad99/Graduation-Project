import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def predict(predictions, y_test, training_data_len, close_df):
    """
    Testing the model and validating its predictions

    Args:
        (np array) predictions -  variable to store the result of (model.predict(test_data))
        (np array) x_test - reshaped array to test the model with
        (np array) y_test - to validate the model on
        (int) training_data_len - the number to split the data with into train and test
        close_df - a data frame of the close price after resetting the index

    Returns:
        validation_df - a df contains the predicted prices and the real data
    """

    # getting the real prediction values instead of the price change in each prediction....
    # reshaping the close_df to be the same shape as the model output
    close_df = np.array(close_df).reshape(-1, 1)
    # real test data without last value
    test_df = np.delete(close_df[training_data_len:, :], -1, 0)
    # real test data shifted
    test_df_shifted = close_df[training_data_len+1:, :]
    # the logic of reversing the data from difference to real
    real_data_prediction = predictions + test_df

    # Calculate/Get the value of MSE
    mse = mean_squared_error(predictions, y_test)
    print("MSE value:", mse)
    # Calculate/Get the value of MAE
    mae = mean_absolute_error(predictions, y_test)
    print("MAE value:", mae)

    # creating a new df to assign the predictions to its equivalent days and comparing them to the real data
    validation_df = pd.DataFrame(real_data_prediction, columns=["predictions"])
    validation_df['real data'] = test_df_shifted
    print(validation_df)

    return validation_df
