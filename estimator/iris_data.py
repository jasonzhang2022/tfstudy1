from __future__ import print_function
import tensorflow as tf
import os.path as path
import numpy as np

#tf.enable_eager_execution()
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_file = tf.keras.utils.get_file(path.basename(TRAIN_URL), TRAIN_URL)
    test_file = tf.keras.utils.get_file(path.basename(TEST_URL), TEST_URL)
    return train_file, test_file

def arrayToObject(*args):
    map = dict({CSV_COLUMN_NAMES[index] :value for index, value in enumerate(args)})

    label = map.pop('Species')
    return map, label

def load_file_as_dataset(file):
    return tf.data.experimental.CsvDataset([file], record_defaults=[tf.float32, tf.float32, tf.float32, tf.float32, tf.int32], header=True).map(arrayToObject)

#100 out 120 for train
def load_train_dataset():
    train_file, _ = maybe_download()
    return load_file_as_dataset(train_file).take(100)

#last 20 for evaluation.
def load_eval_dataset():
    train_file, _ = maybe_download()
    return load_file_as_dataset(train_file).skip(100)

#last 30 for evaluation.
def load_test_dataset():
    _, test_file = maybe_download()
    return load_file_as_dataset(test_file)

if __name__ == "__main__":
    tf.enable_eager_execution()
    ds = load_train_dataset().repeat().batch(10)
    for features, labels in ds:
        print(features)
        print(labels.numpy())
        break
    data = tf.constant([
        [1,2,3],
        [6,5,4],
        [9,10,8]
    ])
    print(data)
    max=tf.argmax(data, axis=1)
    print(max.numpy())
