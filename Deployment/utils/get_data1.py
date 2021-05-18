import pandas_datareader as pdr
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_data(seq_len = 32):
    df = pdr.DataReader('AAPL',data_source='yahoo' ,start='2014-01-01', end='2019-09-30')
    
    #df.drop('Volume',axis=1,inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    #df[df.columns] = scaler.fit_transform(df) 
    df=df[['High','Low','Open','Close','Volume']]

    '''Create training, validation and test split'''

    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    print('Training data shape: {}'.format(train_data.shape))
    print('Validation data shape: {}'.format(val_data.shape))
    print('Test data shape: {}'.format(test_data.shape))
    train_data_len=len(train_data)
    val_data_len=len(train_data)+len(val_data)

    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
        y_train.append(train_data[:, 3][i]) # Value of 4th column (Close Price) of df-row 128+1
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i-seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i-seq_len:i])
        y_test.append(test_data[:, 3][i])    
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print('Training set shape', X_train.shape, y_train.shape)
    print('Validation set shape', X_val.shape, y_val.shape)
    print('Testing set shape' ,X_test.shape, y_test.shape)
    
    return df, X_train, y_train, X_val, y_val, X_test, y_test,train_data_len,val_data_len,scaler