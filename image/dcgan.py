from __future__ import print_function, absolute_import, division, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import numpy as np
import matplotlib.pyplot as plt

import PIL
import os
import time
import glob
import imageio
import re
from IPython import display
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_bool("train", True, "Train if true. Generate if false")

(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
#train_images shape is (60000, 28, 28)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images-127.5)/127.5 # value is (-1, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 32
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#At each epoch, use our trained generator to go through this
#seed to see the quality of generator.
# TODO: should the value be limited to (-1, 1)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

#------------------- exercise below
# No batchNormalization.
def make_discriminator_model():
    model  = keras.Sequential()
    #input will be (batch, 28, 28, 1)
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding="same", input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()

def make_generator_model():
    model  = keras.Sequential()
    #flat layer first: 100 random number.
    # (7,7, 256) is what we want in next step
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7, 256)))

    #unsampling
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1,1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #upsampling : (7,7)->(14, 14)
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2,2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #(14,14)->(28, 28)
    # use tanh so we have value (-1, 1)
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2,2), padding="same", use_bias=False, activation="tanh"))


    return model

generator = make_generator_model()

#from logits, the predicted value is logits, not probabilities
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

'''
real_output: logits from real image (batch, 1)
fake_output: logits from fake image (batch, 1)

'''
def disc_loss(real_output, fake_output):
    loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss

'''
discriminator output for fake image (batch 1)
'''
def gen_loss(fake_output):
    # we want disciminator to classify it as real. So
    # this is why we have tf.ones_like
    return cross_entropy(tf.ones_like(fake_output), fake_output)

gen_optimizer = keras.optimizers.Adam(1e-4)
disc_optimizer = keras.optimizers.Adam(1e-4)

checkpoint_dir = "./dcgan_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
chkpt_prefix = os.path.join(checkpoint_dir, "chkpt")
checkpoint = tf.train.Checkpoint(gen_model= generator, disc_model=discriminator,
                                 gen_optimizer = gen_optimizer, disc_optimizer = disc_optimizer)


@tf.function
def one_step_train(batch_images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_batch = generator(noise, training=True)
        real_output = discriminator(batch_images,  training=True)
        fake_output = discriminator(generated_batch,  training=True)

        disc_loss_value = disc_loss(real_output, fake_output)
        gen_loss_value = gen_loss(fake_output)

    gen_grads = gen_tape.gradient(gen_loss_value, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss_value, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return disc_loss_value, gen_loss_value



def train():
    start = 1
    if (tf.train.latest_checkpoint(checkpoint_dir)):
        basename= os.path.basename(tf.train.latest_checkpoint(checkpoint_dir))
        start = int(basename.split(".")[0].split('-')[-1])
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(start, EPOCHS+1):
        step =0
        for batch_images in train_dataset:
            step += 1
            disc_loss_value, gen_loss_value = one_step_train(batch_images)
            if step%10 == 0:
                print("epoch {:d} step {:d}".format(epoch, step))
        checkpoint.save(file_prefix=chkpt_prefix)
        print(" training for epoch {:d}".format(epoch))

        display.clear_output(wait=True)
        generate_and_save_image(generator, epoch, seed)


def animated_image():
    anim_file = "dcgan.gif"
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = glob.glob("image*.png")
        filenames = sorted(filenames)
        last =-1
        for i, filename in enumerate(filenames):
            frame=2*(i**0.5)
            if (round(frame) > round(last)):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    display.Image(filename=anim_file)

def generate_and_save_image(model, epoch, test_input):

    #values is between (-1, 1) since tanh is used
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4,4))
    #num of examples, seed = 16
    for i in range(16):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i, :, :, 0]*127.5+127.5, cmap="gray")
        plt.axis('off')
    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    #plt.show()

def main(args):
    del args
    if FLAGS.train:
        train()
    else:
        animated_image()

if __name__ == '__main__':
    app.run(main)