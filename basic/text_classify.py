from __future__ import print_function, absolute_import, division, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow.keras.datasets.imdb as imdb
import os
from os.path import exists, join


#use 0 for padding
'''
train_data: (25000,)
train_labels: (25000,)
test_data: (25000,)
test_labels: (25000,)
'''
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words,
                                                                      start_char=1,
                                                                      oov_char=2,
                                                                      index_from=3
                                                                      )
'''
How index works here:
Totally there are 88585 words in the corpra: len(imdb.get_word_index())
imdb.get_word_index() is the map for these 88585 words. It ignores the start_char, oov_char, index_from for load_data.

However, load_data preprocess the text according to parameters. 
 It will number the top word from index: index_from+1.
 It will use 1 as start_char, 2 as oov.
'''
word_to_index = imdb.get_word_index()
word_to_index = {k: v+3 for k, v in word_to_index.items()}
index_to_word = {v: k for k, v in word_to_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<OOV>"
index_to_word[3] = "<UNUSED>"

def decode(encoded_text):
    return (' '.join([index_to_word[i] for i in encoded_text]))

checkpoint_path = "chkpt/model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size=218
if not exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
log_dir = "./logs"
if not exists(log_dir):
    os.mkdir(log_dir)


#------------------------ code below can be deleted
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, dtype='int32', padding='post', truncating='post', value=0)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, dtype='int32', padding='post', truncating='post', value=0)


def create_model():
    model = keras.Sequential([
        keras.layers.Embedding(num_words, 16, input_length=256, name="embedding"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(128, activation="relu", name="dense", kernel_regularizer="l2"),
        keras.layers.Dense(1, activation="sigmoid", name="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
if exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path)
else:
    model = create_model()

#cp= keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="loss", verbose=1, save_best_only = True, mode="min")
tensorboard = keras.callbacks.TensorBoard(join(log_dir, "dropout"), histogram_freq=5, batch_size=batch_size,
                                          embeddings_freq=1,
                                          write_graph=True, write_grads=True)

history = model.fit(train_data, train_labels, epochs=20, batch_size=batch_size, validation_split=0.1,
          callbacks=[tensorboard])

evaluation = model.evaluate(test_data, test_labels)


