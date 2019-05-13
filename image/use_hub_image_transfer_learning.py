from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tensorflow_hub as hub
import os

import tensorflow.keras as keras

print(tf.version)

#constant
FLOWERS_URL="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
HUB_MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
FEATURE_EXTRACTOR_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
IMAGE_LABELS = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
TEST_IMAGE = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'

#download and save files
flower_photos_diretory = keras.utils.get_file("flower_photos", FLOWERS_URL, untar=True)
all_existing_label_file = keras.utils.get_file(os.path.basename(IMAGE_LABELS), IMAGE_LABELS)
test_image_file = keras.utils.get_file(os.path.basename(TEST_IMAGE), TEST_IMAGE)

IMAGE_SIZE = (224, 224)


def prepare_labels():
    lines = [line.strip()  for line in open(all_existing_label_file, "r").readlines() ]

    return lines

imagenet_labels = prepare_labels()


'''
1. Use MobileNet to classify test_imag
'''
test_image = tf.io.decode_jpeg(open(test_image_file, "rb").read(), channels=3)
test_image = tf.image.resize(test_image, size=IMAGE_SIZE)
test_image = test_image / 255
batch_image = tf.expand_dims(test_image, 0)
model = keras.Sequential([hub.KerasLayer(HUB_MODEL_URL, trainable=False, input_shape=IMAGE_SIZE + (3,))])


prediction = model.predict(batch_image)
predicted_index = tf.math.argmax(prediction, axis=-1)[0]
print(predicted_index)
print("predicted class: ", imagenet_labels[predicted_index.numpy()])
'''
2. transfer learning
'''
checkpoint_dir = "hub_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

BATCH_SIZE = 32
images = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
images_data = images.flow_from_directory(flower_photos_diretory, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE)

#properties not in document.
num_classes  = images_data.num_classes
num_examples = images_data.samples
class_indices = images_data.class_indices

index_classes = { key: c for (c, key) in class_indices.items() }

new_model = keras.Sequential(
    [hub.KerasLayer(FEATURE_EXTRACTOR_URL, trainable=False, input_shape=IMAGE_SIZE + (3,)),
     keras.layers.Dense(num_classes, activation="softmax")
     ]
)

step = num_examples//BATCH_SIZE

new_model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['acc'])
if not tf.train.latest_checkpoint(checkpoint_dir):
    chkpt_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"/cp-{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
    new_model.fit(images_data, epochs=2, steps_per_epoch=step, callbacks=[chkpt_callback])
else:
    lastest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    new_model.load_weights(lastest_checkpoint)

for image_batch, labels in images_data:
    predictions = new_model.predict(image_batch)
    predicted_indices = tf.argmax(predictions, axis=-1)
    labels = tf.argmax(labels, axis=-1)
    for i in range(tf.size(predicted_indices)):
        predicted = predicted_indices[i].numpy()
        real = labels[i].numpy()
        print ("real {}: predicted {} ".format(index_classes[real], index_classes[predicted]))

    no_equals_sum = tf.reduce_sum(tf.cast(tf.math.not_equal(predicted_indices, labels), tf.int32))
    print( "{:d} of out 32 are not equal.".format(no_equals_sum.numpy()) )

    break




