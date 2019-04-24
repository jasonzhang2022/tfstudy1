from __future__ import print_function, absolute_import, division
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import tensorflow.feature_column as feature_column
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import basename


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
filename = basename(URL)
file = keras.utils.get_file(filename, URL)
df = pd.read_csv(file)

'''
60% train size
20% val
20% test
'''
'''
size = len(df.index)
val_size = int(size*0.1)
train = df.iloc[:(size-val_size*2)]
val = df.iloc[(size-val_size*2):(size-val_size)]
test = df.iloc[(size-val_size):]
'''
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train))
print(len(val))
print(len(test))

batch_size = 32
def dataframe_to_dataset(df, shuffle=True, batch_size=32):
    target = df.pop("target")
    dataset = tf.data.Dataset.from_tensor_slices( (dict(df), target))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = dataframe_to_dataset(train)
val_dataset = dataframe_to_dataset(val)
test_dataset = dataframe_to_dataset(test)

#---------------------------
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

age_col = feature_column.numeric_column("age")
# bucketized cols
age_buckets = feature_column.bucketized_column(age_col, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

model = keras.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
tensorboard = keras.callbacks.TensorBoard(log_dir="logs", write_graph=True, write_grads=True, histogram_freq=1 )
model.fit(train_dataset, epochs=50, callbacks=[tensorboard], validation_data=val_dataset)
eval_statistics=model.evaluate(test_dataset)
print(eval_statistics)
