import tensorflow as tf
from tensorflow.keras.layers import Layer

class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        # Slope Matrix for time series
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
        # Inercept with y matrix (φ)
        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
        # Slope matrix for periodic time (W)
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
        # Inercept with y matrix  for non_periodic time(φ)
        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        # x will be input_values(batch=32 , seq_len=128 , Features = 5)
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1)  # find the mean of dim

        # calculate the non-periodic (linear) time feature
        time_linear = self.weights_linear * x + self.bias_linear # Linear time feature

        time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
        # calculate the periodic time feature and applied Sin Function (not Relu)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        # expand the dimension by 1
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
        # Concat time_linear(non_periodic) with time _periodic
        return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config