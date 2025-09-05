import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_softmax(tw, din, k, hidden=64, dropout=0.2, lr=1e-3):
    model = Sequential([
        LSTM(hidden, input_shape=(tw, din)),
        Dropout(dropout),
        Dense(hidden, activation="relu"),
        Dropout(dropout),
        Dense(k, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="categorical_crossentropy")
    return model
