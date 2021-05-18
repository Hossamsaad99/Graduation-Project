import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling1D
import keras
from .Encoder import TransformerEncoder
from .Time2Vector import Time2Vector

def get_model(seq_len, d_k, d_v, n_heads, ff_dim, encoder_stack_size = 3, loss='mse', output_activation='linear'):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    Encoder_layer = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, 5))
    time_embedding_layer = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, time_embedding_layer])
    
    for _ in range(encoder_stack_size):
        x = Encoder_layer((x, x, x))
    
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation=output_activation)(x) # sigmoid
    lr_dc=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=10000,decay_rate=0.3)
    #optimizer=tf.keras.optimizers.Adam(
    #learning_rate=0.001,decay=0.3)
    optimizer=tf.keras.optimizers.Adam(lr_dc)
    
    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    return model