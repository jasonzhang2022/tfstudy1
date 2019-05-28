from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as display
import time
import tensorflow.keras as keras

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

content_path = tf.keras.utils.get_file('turtle.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

labels = np.array([line.strip() for line in open(labels_path).readlines()])

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

style_weight = 1e-2
# content has a much great weight e^6 more than style_weight
# This because we start with content image. The style loss is huge at very beginning
content_weight = 1e4

'''
idea here is simple. 
1. for a particular layer, uses cross channel correlation(dot product) to represent style
2. use channel value to represent image features. 

The calculate the loss

'''

# ---------------------------------- practice here
'''
load an image
value [ 0, 1)
size (244, 244)
'''


def load_image(file_path):
    max_dim = 512
    bytes = tf.io.read_file(file_path)
    # image right how has a shape (1, width, height, 3)
    image = tf.image.decode_image(bytes, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    return image


def show_img(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


content_image = load_image(content_path)
style_image = load_image(style_path)


def construct_model(layer_names):
    vgg = keras.applications.VGG19(include_top=True, weights="imagenet")
    vgg.trainable = False
    output = [vgg.get_layer(name).output for name in layer_names]

    model = keras.Model([vgg.input], output)
    return model


def gram_matrix(input_tensor):
    # shape (b, c=d, c=d)
    correlation = tf.linalg.einsum("bwhc,bwhd->bcd", input_tensor, input_tensor)
    shape = tf.shape(input_tensor)
    num_cells = tf.cast(shape[1] * shape[2], correlation.dtype)
    correlation_normlaized_by_Cell = correlation / num_cells
    return correlation_normlaized_by_Cell


class StyleContentModel(keras.models.Model):
    def __init__(self, style_layer_names, content_layer_names):
        super(StyleContentModel, self).__init__()
        self.vgg = construct_model(style_layer_names + content_layer_names)
        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style = len(style_layer_names)
        self.vgg.trainable = False

    def call(self, image):
        image = image * 255
        image = keras.applications.vgg19.preprocess_input(image)
        outputs = self.vgg(image)

        style_outputs = outputs[:self.num_style]
        content_outputs = outputs[self.num_style:]

        content = {name: value for name, value in zip(self.content_layers, content_outputs)}

        style = {name: gram_matrix(value) for name, value in zip(self.style_layers, style_outputs)}

        return {'content': content, 'style': style}


extractor = StyleContentModel(style_layers, content_layers)
style_target_dict = extractor(style_image)['style']
content_target_dict = extractor(content_image)['content']
trained_image = tf.Variable(content_image)
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-2)


def clip_to_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def step_loss(outputs):
    content_output = outputs['content']
    style_output = outputs['style']

    content_loss = tf.add_n(
        [tf.reduce_mean(tf.square(content_output[name] - content_target_dict[name])) for name in content_output.keys()])
    # loss contributed by a data point
    content_loss = content_loss / len(content_output)

    style_loss = tf.add_n(
        [tf.reduce_mean(tf.square(style_output[name] - style_target_dict[name])) for name in style_output.keys()])
    style_loss = style_loss / len(style_output)

    return content_weight * content_loss + style_weight * style_loss


@tf.function
def train_one_step():
    with tf.GradientTape() as tape:
        outputs = extractor(trained_image)
        loss = step_loss(outputs)

    grads = tape.gradient(loss, trained_image)
    optimizer.apply_gradients([(grads, trained_image)])
    return loss


def train():
    for i in range(500):
        loss = train_one_step()
        print("{:d}: {:015.4f}".format(i, loss))
        trained_image.assign(clip_to_1(trained_image))
    show_img(trained_image, "trained image")
    plt.show()


train()
