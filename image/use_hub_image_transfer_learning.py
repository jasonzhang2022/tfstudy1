from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tensorflow_hub as hub
import os.path as path

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow.keras.backend as K
import PIL
#constant
FLOWERS_URL="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
HUB_MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
FEATURE_EXTRACTOR_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
IMAGE_LABELS = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
TEST_IMAGE = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'

#download and save files
flower_photos_diretory = keras.utils.get_file("flower_photos", FLOWERS_URL, untar=True)
all_existing_label_file = keras.utils.get_file(path.basename(IMAGE_LABELS), IMAGE_LABELS)
test_image_file = keras.utils.get_file(path.basename(TEST_IMAGE), TEST_IMAGE)

#prepare data
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(HUB_MODEL_URL))
all_existing_labels = np.array(open(all_existing_label_file).read().splitlines())
flower_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flower_data = flower_data_generator.flow_from_directory(flower_photos_diretory, target_size=IMAGE_SIZE, batch_size=32)
sorted_flowers=sorted(flower_data.class_indices.items(), key= lambda pair: pair[1])
flowers = [name for name, index in sorted_flowers]

feature_extractor_layer= keras.layers.Lambda(lambda x: hub.Module(FEATURE_EXTRACTOR_URL)(x), input_shape=IMAGE_SIZE+[3])
feature_extractor_layer.trainable = False
model = keras.Sequential([
   feature_extractor_layer,
    keras.layers.Dense(len(flowers), activation="softmax")
])
print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer = tf.train.AdamOptimizer(), metrics=["accuracy"])

#init hub module
K.get_session().run(tf.global_variables_initializer())

class CollectBatchStatus(keras.callbacks.Callback):

    def __init__(self):
        self.losses=[]
        self.accuracy=[]

    def on_batch_end(self, batch, log):
        self.losses.append(log['loss'])
        self.accuracy.append(log['acc'])

'''
what does flower_data has
1. it is an iterator each element is a batch. Each element in batch is (image, label)
2. num_classes
3. samples.
4. class_indice
'''


steps = flower_data.samples // 32
batch_stats = CollectBatchStatus()
stats=model.fit( (image_batch for image_batch in flower_data), epochs = 1,
           steps_per_epoch = steps,
           validation_split=0.2, callbacks=[batch_stats])

print(stats)



def plotBatchStatus():
    plt.figure(figsize=[2,4])
    plt.subplot(2,1, 1)
    batches = range(1, steps+1)
    plt.plot(batches, batch_stats.losses)
    plt.subplot(2,1, 2)
    plt.plot(batches, batch_stats.accuracy)
    plt.show()

plotBatchStatus()

iterator = iter(flower_data)
(flower_batch, flower_label) = next(iterator)
predictions = model.predict(flower_batch)
predicted_index = np.argmax(predictions, axis=1)
flower_index = np.argmax(flower_label, axis=1)
print(flower_index)
print(predicted_index)

equals = np.equal(flower_index, predicted_index)
print(equals)
print(np.sum(equals))




