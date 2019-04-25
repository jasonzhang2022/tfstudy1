#https://www.tensorflow.org/alpha/tutorials/text/text_generation
from __future__ import absolute_import, unicode_literals, print_function, division

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np
import os
import time
import sys
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("train", True, "Train if true. Generate if false")
# timestep size
fragment_len = 32
batch_size = 64
SHUFFLE_SIZE = 10000

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
log_dir="./logs"
checkpoint_dir = "chkpts"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "chkpt_{epoch}")

'''
flow: 
1. collect all characters.
2. language model: 32 character as input. 
3. generate: always pick up the character of greatest possibility.  
'''

#-------------------------------------- practice below
'''
step 1: prepare input: string to integer
'''
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
vocabulary = set()
for c in text:
    vocabulary.add(c)

char_to_index = {c[0]: c[1] for c in zip(vocabulary, range(len(vocabulary)))}
index_to_char = {value: key for key, value in char_to_index.items()}

#65 character
voc_len = len(char_to_index)

encoded_text = [char_to_index[c] for c in text]

'''
step 2: create dataset
'''
def _map_one_frament(int_elements):
    temp_x = tf.one_hot(int_elements[:-1], voc_len)
    temp_y = int_elements[1:]
    return temp_x, temp_y

def prepare_data_dataset():
    ds = tf.data.Dataset.from_tensor_slices(encoded_text).batch(fragment_len+1, drop_remainder=True)
    ds = ds.map(_map_one_frament).shuffle(SHUFFLE_SIZE).batch(batch_size, drop_remainder=True)
    return ds

dataset = prepare_data_dataset()


'''
step 3: create model
'''
tensorboard_callback = keras.callbacks.TensorBoard("./logs")
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
def build_model(batch_size):
    internal_model = keras.Sequential(
       [keras.layers.LSTM(64, input_shape=(fragment_len, voc_len), return_sequences=True, stateful=True, batch_size=batch_size),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(voc_len)]
    )
    internal_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam" )
    return internal_model


# generate
temperature = 1.0
num_words = 1000
def generate(model, start_text):
    generated_text = ''
    for iteration in range(num_words):
        start_ints= [ char_to_index[c] for c in start_text]
        start_ints = tf.one_hot(start_ints, voc_len)
        start_ints = tf.expand_dims(start_ints, 0)
        predictions = model(start_ints)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        predicted_id = predicted_id[-1, 0].numpy()

         #predictions = tf.argmax(predictions, -1)
        generated_text = generated_text + index_to_char[predicted_id]
        start_text = index_to_char[predicted_id]

    return start_text + generated_text


def test_generate(model, start_text):
    start_ints= [ char_to_index[c] for c in start_text]
    start_ints = tf.one_hot(start_ints, voc_len)
    start_ints = tf.expand_dims(start_ints, 0)
    predictions = model(start_ints)
    predictions = tf.squeeze(predictions, 0)
    predictions = tf.argmax(predictions, -1).numpy()
    print(predictions)
    text = ''.join(index_to_char[i] for i in predictions)
    print(text)

def main(args):
    del args
    if FLAGS.train:
        model = build_model(batch_size)
        model.fit(dataset, epochs=100, callbacks=[tensorboard_callback, checkpoint_callback])
    else:
        # batch size can change from train to predict.
        model = build_model(1)
        print (tf.train.latest_checkpoint(checkpoint_dir))
        status = model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        # change the batch shape for prediction
        #model.build(tf.TensorShape([1, None]))
        #model.reset_states()
        test_generate(model, u"ROMEO: ")
        status.assert_existing_objects_matched()
       # final_text = generate(m, u"ROMEO: ")
       # print(final_text)

if __name__ == '__main__':
    app.run(main)

