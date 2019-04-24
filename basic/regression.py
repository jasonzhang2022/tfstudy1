from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin==1)*1.0
dataset['Europe'] = (origin==2)*.10
dataset['Other'] = (origin ==3)*.10

train = dataset.sample(frac=0.8, random_state=0)
print(len(train))
train_labels = train.pop("MPG")
print(len(train_labels))

test = dataset.drop(train.index)
test_labels = test.pop("MPG")

train_stats = train.describe()
train_stats = train_stats.transpose()
cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']

def norm(df):
    for col in cols:
        df[col]= (df[col]-train_stats.loc[col, 'mean'])/train_stats.loc[col, 'std']
    return df
#categorical column should not be normalized
train_dataset = norm(train)
print(len(train_dataset))

test_dataset = norm(test)
batch_size = 32
'''
def dataframe_to_dataset(data, label, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices( (dict(data), label))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = dataframe_to_dataset(train_dataset, train_labels)
test_dataset = dataframe_to_dataset(test_dataset, test_labels, shuffle=False)
for batch in train_dataset.take(1):
    print(batch)
'''
col_num=len(train_dataset.columns)
#---------------------------------
model = keras.Sequential([
    keras.layers.Dense(8, activation="relu", input_shape=(col_num,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
tb= keras.callbacks.TensorBoard("./logs", write_graph=True)
print(len(train_dataset))
print(len(train_labels))
model.fit(train_dataset, train_labels, epochs=40, validation_data =[test_dataset, test_labels], callbacks=[tb])
