from __future__ import print_function, absolute_import, unicode_literals,  division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow_datasets as tfds

split_weights=(8,1,1)
#found this from mobile_net
IMAGE_SIZE = (224, 224)
splits = list(tfds.Split.TRAIN.subsplit(weighted=split_weights))
#buffer size can't be big loadintg so much image in memory uses a lot of resorces
BUFFER_SIZE = 1000
BATCH = 32
(raw_train, raw_validation, raw_test), info = tfds.load("cats_vs_dogs", split=splits, with_info=True, as_supervised=True)
'''
image characteristic: each image has its own size.
the shape is (None, None, 3).
the dtype is uint8. 
'''
num_train_examples = info.splits['train'].num_examples

def process_image(image, label):
    # This is not necessary. In python3, call math division is float division
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, size=IMAGE_SIZE)
    #value is from [-1, 1]
    #feature scaling and normalization
    image = image/127.5 -1

    #label is integer, 0, 1
    # 0 is cat, 1 is dog
    return image, label

# Map has to be done before batch since image has different size and can't be batched.
train_ds = raw_train.shuffle(BUFFER_SIZE).map(process_image).batch(BATCH, drop_remainder=True)
validation_ds = raw_validation.map(process_image).batch(BATCH, drop_remainder=True)
test_ds = raw_test.map(process_image).batch(BATCH, drop_remainder=True)

# input_shape is default (224, 224, 3)
# weights is default : imagenet
feature_extractor = keras.applications.mobilenet_v2.MobileNetV2(include_top=False)
feature_extractor.trainable = False

model = keras.Sequential([
    feature_extractor,
    keras.layers.GlobalAveragePooling2D(),
    #You don't need an activation function here because this prediction will be treated as a logit, or a raw prediciton value. Positive numbers predict class 1, negative numbers predict class 0.
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
step_per_epoch = num_train_examples*8//10
initial_epochs = 10
total_epochs = 20
model.fit(train_ds, validation_data=validation_ds, steps_per_epoch=step_per_epoch, epochs=initial_epochs)

feature_extractor.trainable = True
for layer in feature_extractor.layers[:100]:
    layer.trainable = False
model.fit(train_ds, validation_data=validation_ds, steps_per_epoch=step_per_epoch, epochs=total_epochs, initial_epoch = initial_epochs)
print(model.evaluate(validation_ds))







