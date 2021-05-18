from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras
from tensorflow.keras.models import load_model


def get_model(x_train, units=None, dropout=0.2, loss='mean_squared_error', metrics='MAE'):
    """
    Building the LSTM architecture to have three LSTM layer and dense layer with 1 neuron as output layer
    Args:
        (np array) x_train - a 3D shaped array input to the model
        (list) units - number of units in each layer
        (int) dropout - Dropout percentage to control over-fitting during training
        (str) loss - the loss function to be used
        (str) metrics - the metric to be used
    Returns:
        model - the model after being compiled
    """
    if units is None:
        units = [40, 40, 50]
    model = Sequential()
    model.add(LSTM(units=units[0], return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units[1], return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units[2]))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    lr_dc = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.6)
    opt = keras.optimizers.Adam(learning_rate=lr_dc)
    model.compile(optimizer=opt, loss=loss, metrics=[metrics])
    model.save('LSTM.hdf5')
    model = load_model('LSTM.hdf5')
    return model
