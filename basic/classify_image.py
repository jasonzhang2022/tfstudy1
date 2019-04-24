from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from os.path import exists, join, dirname

fashion_mnist = keras.datasets.fashion_mnist

'''
train_image: (60000, 28, 28)
train_labels: (60000,)
test_images: (10000, 28, 28)
'''
batch_size = 128
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.astype("float32")
test_images.astype("float32")
train_images = train_images /255.0
test_images = test_images /255.0
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[1])
    test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])
else:
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[1], 1)
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

print(test_images.shape)
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" ]
checkpoint_path = "training_embed/cp-{epoch:04d}.ckpt"
logdir = "./logs"
checkpoint_dir = dirname(checkpoint_path)
if not exists(checkpoint_dir):
    mkdir(checkpoint_dir)

if not exists(logdir):
    mkdir(logdir)
metadata_file = join(logdir, "metadata.tsv")
if not exists(metadata_file):
    with open(metadata_file, "w") as f:
        np.savetxt(f, test_labels)

#-------------------

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1), name="conv1", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3,3),  name="conv2", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3,3),  name="conv3", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu", name="dense1"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

cp = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)
tensorboard = keras.callbacks.TensorBoard("./tb/embed", histogram_freq=5, batch_size=batch_size,
                                          write_graph=True, write_grads=True, write_images=True,
                                          embeddings_layer_names="dense1",
                                          embeddings_metadata = metadata_file,
                                          embeddings_freq=1
                                          )
early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.005, patience=4)

model =create_model()
history= model.fit(train_images, train_labels, epochs=20, batch_size=batch_size, validation_split=0.1,
                      callbacks=[cp, tensorboard, early_stopping])

evaluation = model.evaluate(test_images, test_labels)
print(evaluation)


