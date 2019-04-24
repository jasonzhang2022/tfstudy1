from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

print(tf.version.VERSION)

'''
Purpose: we received message: title, body, and tags. We want
1. assign priority: high and low
2. assign to one department out of four.
'''


num_tags = 12
num_words = 10000 # voicabulary size for embedding
num_department = 4 # route the message to four department.

'''
We are going to have 3 inputs
1 title text
2. body text
2. tags

For title text, and body text, we share embedding layer.
'''

'''
if shape=(2,): It is one dimensional input with two elements.
if shape=(None): It is one dimensional inptu with undefined number of inputs.
'''

def model():
    embedding = layers.Embedding(num_words, 120, name="embedding")

    title_input = keras.Input(shape=(None,), dtype="int32", name="title")
    title_embedding = embedding(title_input)
    title_lstm = layers.LSTM(128, name="title_lstm")(title_embedding)

    body_input = keras.Input(shape=(None,), dtype="int32", name="body")
    body_embedding = embedding(body_input)
    body_lstm = layers.LSTM(32, name="body_lstm")(body_embedding)


    # 12: [0, 1, 0, 1, ..]
    tag_features = keras.Input(shape=(num_tags,), name="tags", dtype="float32")

    allfeatures = layers.concatenate([title_lstm, body_lstm, tag_features], name="all")
    x = layers.Dense(10, activation="relu", name="dense10")(allfeatures)

    priority_prediction = layers.Dense(1, activation = "sigmoid", name='priority')(x)
    department_prediction = layers.Dense(num_department, activation= "softmax", name="department")(x)

    model = keras.Model(inputs = [title_input, body_input, tag_features],
                        outputs=[priority_prediction, department_prediction])
    model.compile(optimizer = keras.optimizers.RMSprop(1e-3),
                  loss=["binary_crossentropy", "categorical_crossentropy"],
                  loss_weights=[1.0, 0.2])
    return model


samples = 10000
titles = tf.random.uniform(shape=(samples, 10), minval=0, maxval=num_words, dtype=tf.dtypes.int32)
body = tf.random.uniform(shape=(samples, 123), minval=0, maxval=num_words, dtype=tf.dtypes.int32)
tags = tf.constant(np.random.randint(2, size=(samples, num_tags)).astype('float32'))
priorities  = tf.constant(np.random.randint(2, size=(samples, 1)).astype('float32'))
departments = np.zeros(shape=[samples, num_department], dtype=np.int8)
departments[range(samples), np.random.randint(num_department, size=(samples))]=1
departments = tf.constant(departments)

model = model()
model.fit({'title': titles, "body": body, "tags": tags
           }, {'priority': priorities, "department": departments},

          epochs=2, batch_size=32)

#keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

#print(model.summary())




