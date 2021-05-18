import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.Time2Vector import Time2Vector
from utils.Attention import MultiAttention, SingleAttention
from utils.Encoder import TransformerEncoder
from utils.get_data import get_data

def train_model(model, epoches=50, batch=32):
    seq_len = 128
    df, X_train, y_train, X_val, y_val, X_test, y_test,train_data_len,val_data_len,scaler = get_data()

    CP = ModelCheckpoint(filepath='Transformer+TimeEmbedding.hdf5',
                           monitor='val_loss',
                           save_best_only=True,
                           verbose=1)

    ES = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=10)

    history = model.fit(X_train, y_train,
                        batch_size=batch,
                        epochs=epoches,
                        callbacks=[CP, ES],
                        verbose=1,
                        validation_data=(X_val, y_val))

    model = load_model('Transformer+TimeEmbedding.hdf5',
                       custom_objects={'Time2Vector': Time2Vector,
                                       'SingleAttention': SingleAttention,
                                       'MultiAttention': MultiAttention,
                                       'TransformerEncoder': TransformerEncoder})
                                                       

    #Print evaluation metrics for all datasets
    train_eval = model.evaluate(X_train, y_train, verbose=0)
    val_eval = model.evaluate(X_val, y_val, verbose=0)
    test_eval = model.evaluate(X_test, y_test, verbose=0)

    print('\nEvaluation metrics')
    print('Training Data - Loss: {:.4f}, MAE: {:.4f}'.format(train_eval[0], train_eval[1]))
    print('Validation Data - Loss: {:.4f}, MAE: {:.4f}'.format(val_eval[0], val_eval[1]))
    print('Test Data - Loss: {:.4f}, MAE: {:.4f}'.format(test_eval[0], test_eval[1]))
    # Visualize Loss Vs. epochs

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
