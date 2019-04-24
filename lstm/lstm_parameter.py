from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow.keras as keras

model = tf.keras.Sequential([
    keras.Input(shape=(5,2), name="input"),
    keras.layers.LSTM(units=2),
    keras.layers.Dense(1)

])

print(model.summary())
