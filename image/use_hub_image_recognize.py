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
flower_data = flower_data_generator.flow_from_directory(flower_photos_diretory, target_size=IMAGE_SIZE)
sorted_flowers=sorted(flower_data.class_indices.items(), key= lambda pair: pair[1])
flowers = [name for name, index in sorted_flowers]
test_image = PIL.Image.open(test_image_file).resize(IMAGE_SIZE)
test_image_as_batch = np.array(test_image)[np.newaxis, ...]/255
model = keras.Sequential([
    keras.layers.Lambda(lambda x: hub.Module(HUB_MODEL_URL)(x), input_shape=IMAGE_SIZE+[3])
])

session = K.get_session()
session.run(tf.global_variables_initializer())
predicitons = model.predict(test_image_as_batch)
print(predicitons.shape)
predicted_index = np.argmax(predicitons, axis=1)
print(predicted_index)

print("predicted index: {}. predicted class: {}".format(predicted_index, all_existing_labels[predicted_index]))

#32 images.
#8 rows. Each rows has 4 images.
#Each image is one inch
def show_one_batch_image(images, batch_labels):
    plt.figure(figsize=[4, 8])
    plot_in_col = 4
    row=len(batch_labels)/plot_in_col
    for index in range(len(batch_labels)):
        print("{}, {}, {}".format(row, plot_in_col, index+1))
        plt.subplot(row, plot_in_col, index+1)
        plt.imshow(images[index])
        flower_index  = np.argmax(batch_labels[index])
        plt.xlabel(flowers[flower_index])
        plt.legend()
    plt.show()




