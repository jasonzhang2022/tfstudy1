from __future__ import absolute_import, print_function, division
import tensorflow as tf
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
batch_size = 8

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
'''
1. defined model
2. define train function
  2.1 go many epoches
  2.1.1: go many batches
    each batch: 
      calculate logit->softmax prediciton
      calculate loss
      track gradient.
      apply gradient(optimizer)
      accumulate metrics: loss, accuracy.
 2.1.2: 
   calculate epoch metrics
'''
#------------------------ delete below and practice

train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp,
                                                      batch_size=batch_size,
                                                      num_epochs=1,
                                                      shuffle=False,
                                                      header=True,
                                                      column_names=column_names,
                                                      label_name=column_names[-1]
                                                      )
def convert_ds(features_dict):
    return tf.stack(list(features_dict.values()), axis=1)

model = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=(4,)),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(3)
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)

def one_batch(x, y):
    with tf.GradientTape() as t:
      logits = model(x)
      probabilities = K.softmax(logits)
      loss = keras.losses.sparse_categorical_crossentropy(y, probabilities)
    gradients = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return (loss, logits)

losses= []
accuracies = []
def train(epochs):
    for epoch in range(epochs):
        epoch_losses = keras.metrics.Mean()
        epoch_accuracies = keras.metrics.SparseCategoricalAccuracy()
        for features_dict, labels in train_dataset:
            x = convert_ds(features_dict)
            loss, logits = one_batch(x, labels)
            epoch_losses(loss)
            new_logits = model(x)
            epoch_accuracies(labels, new_logits)
        losses.append(epoch_losses.result())
        accuracies.append(epoch_accuracies.result())
        print(accuracies[-1].numpy())

def plot():
    epochs = range(len(losses))
    plt.figure(1, figsize=(6,12))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, losses, 'g')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(False)

    plt.subplot(2,1,2)
    plt.plot(epochs, accuracies, 'c')
    plt.title("acuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(False)

    plt.show()

train(40)
plot()
