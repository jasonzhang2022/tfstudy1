from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow.keras as keras

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.utils import shuffle
import json
import re
import pickle
from glob import glob
import time
import absl.app as app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("op", "train", "train|pre|eval")

keras_datasets_dir = os.path.join(os.environ['HOME'], ".keras", "datasets")

chkpt_dir = "./image_checkpoints"
checkpoint_prefix = os.path.join(chkpt_dir, "chkpt")
if not os.path.exists(chkpt_dir):
    os.mkdir(chkpt_dir)

'''
General idea. 

The model is same as neural network translation with attention.
Except, 
1. The encoder's changed. 
  a) there is no embedding layer since we deal with image data
  b) There is no GRU layer. FC is used since relationship in temporal dimension doesn't exist. 
  c) since there is no GRU unit, we need a way to share parameters. Compute all encoders' state before decoder.
2) Image data is huge. It can't fit in memory. This is why we definitely need tf.data.Dataset
3) Image features are processed offline and saved to disk.
'''

'''
files and data
Download using keras is too slow. I downloaded manually use gsutil as is instructed in COCO website.
'''
image_dir = os.path.join(keras_datasets_dir, "coco_images_train2014")
annotation_file = os.path.join(keras_datasets_dir, "annotations", "captions_train2014.json")

'''
Annotation JSON file structure
{
images: [
{"license": 3,"file_name": "COCO_train2014_000000143952.jpg",
"coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000143952.jpg","height": 480,"width": 640,"date_captured": "2013-11-17 03:12:55","flickr_url": "http://farm9.staticflickr.com/8552/8699185361_9b7d472287_z.jpg","id": 143952},
...],
licenses: [ ...]
annotations: [{"image_id": 318556,"id": 48,"caption": "A very clean and well decorated empty bathroom"}, , ...]
}

Each image has a different size. We need to resize it to a format model expects
'''

'''
Keep this number small to make sure all code works.
'''
num_examples = 10000
# a different size from 64 to figure the shape easily in development phase.
# change this to other size after code is verified.
batch = 32

'''
Step 1: create input data and target data
'''
with open(annotation_file, "r") as f:
    annotations = json.load(f)

original_captions = []
image_infos = []

for annotation in annotations['annotations']:
    original_captions.append(annotation['caption'])
    image_id = annotation['image_id']

    '''
    Why we need to process all these information in the beginning
    1. Once tf starts a graph, all operations has to be supported by TF.
    Usually these operations are in the tf package.
    For example you can use tf.read_file, but not os.open(...).read().
    If your operations is not supported  by tf, there are two options.
    a) use tf.py_function or tf.numpy_function. This has impact on performance.
    b) extract these operations to data preparation stage. When it comes to ML, 
     data is ready. TF only needs to do math operations.
    
    TF flow starts from dataset.     
    '''
    image_file = image_dir + "/COCO_train2014_{:012d}.jpg".format(image_id)
    feature_file = "./features/COCO_features_{:012d}".format(image_id)

    # we need to convert image_id to string since Tensor can't have data item of different types
    image_infos.append((str(image_id), image_file, feature_file))


def create_image_feature_extract_model():
    # weights="imagenet" to load the trained weight from imagenet.
    # The other option is "none". Pass "none" if we want to use the Network architecture instead of weights
    model = keras.applications.inception_v3.InceptionV3(include_top=False, weights="imagenet")
    input = model.input
    hidden_layer = model.layers[-1].output

    return keras.Model(input, hidden_layer)


image_feature_extract_model = create_image_feature_extract_model()

'''
load and process image so it is good for feature_extraction
'''
def load_image(info):
    image_id = info[0]
    image_file = info[1]
    feature_file = info[2]
    file_content = tf.io.read_file(image_file)
    img = tf.image.decode_jpeg(file_content, channels=3)
    img = tf.image.resize(img, (299, 299))
    return img, image_id, image_file, feature_file


def pre_compute_and_save_features(image_infos1):
    if not os.path.exists("./features"):
        os.mkdir("./features")
    unique_infos = list(set(image_infos1))
    infos = tf.data.Dataset.from_tensor_slices(unique_infos).map(load_image).batch(batch)
    for img_batch, _, _, feature_file_batch in infos:
        # images = tf.image.resize(img_batch, (299, 299))
        images = keras.applications.inception_v3.preprocess_input(img_batch)
        features = image_feature_extract_model(images)
        # collapse 8*8 to 64
        features = tf.reshape(features, shape=(features.shape[0], -1, features.shape[3]))
        for i in range(len(features)):
            tf.io.write_file(feature_file_batch[i], tf.io.serialize_tensor(features[i]))

top_k = 5000
'''
1. We don't need to normalize text since we use English.
2. We don't need to treat punctuation as token in viocabulary. We filter all of them. 

compare with neural translation example.
in Neural translation, sentence beginning/ending character is treated as token. Other punctuations are replaced with space.
So in the neural translation, we have character-level pre-process

In this example, typical logic is used. Punctuation is filtered out(removed). "<, >" are handled specially. 

'''

def _start_end(s):
    s= "<start> " + s
    s= s + " <end>"
    return s

def preprocess_caption(captions1):
    captions1 = [ _start_end(s) for s in captions1]

    #remove < and > from filters since we use < and > in start, end and unk
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(captions1)

    tokenized = tokenizer.texts_to_sequences(captions1)
    padded = keras.preprocessing.sequence.pad_sequences(tokenized, padding="post")

    return padded, tokenizer

captions, tokenizer = preprocess_caption(original_captions)

def create_train_dataset(howmany):
    feature_ds = tf.data.Dataset.from_tensor_slices(image_infos[:howmany]).map(
        lambda info: tf.io.parse_tensor(tf.io.read_file(info[2]), out_type=tf.float32))
    caption_ds = tf.data.Dataset.from_tensor_slices(captions[:howmany])
    return tf.data.Dataset.zip( (feature_ds, caption_ds)).shuffle(buffer_size=howmany).batch(batch, drop_remainder=True)


#convert the features into hidden states accpets by decoder
class Encoder(keras.Model):
    def __init__(self, fc_units):
        super(Encoder, self).__init__()
        self.fc = keras.layers.Dense(fc_units, activation="relu")

    #features shape(batch, 64, 2048)
    def __call__(self, features):
        '''
         features here is viewed as image encoding. Its values are real number.
         Text sequences are real number. So an embeding can be used.
         For real number, we can't use embedding. A Fully-Connected layer is used here.
        :param features:
        :return:
        '''

        # this has parameters share
        # Last dimension goes through Dense calculation.
        x = self.fc(features)
        return x


class Attention(keras.Model):
    def __init__(self, attention_units):
        super(Attention, self).__init__()
        self.attention_unit = attention_units
        self.W1 = keras.layers.Dense(attention_units)
        self.W2 = keras.layers.Dense(attention_units)
        self.V = keras.layers.Dense(1)


    '''
    query is the previous hidden state from decoder's GRU so its shape will be
     (batch, gru_unit)
     
     hidden_states are the states from decoder. Its shape is 
     (batch, 64(fake timestep),  2018)
    '''
    def __call__(self, query, hidden_states):

        query_with_time_axis = tf.expand_dims(query, axis=1)

        #(batch, 1, attention_units)
        w1 = self.W1(query_with_time_axis)

        #(batch, 64, attention_units)
        w2 = self.W2(hidden_states)

        #(batch, 64, attention_units)
        score = tf.nn.tanh(w1 + w2)

        #(batch, 64, 1)
        # Map attention_units to one number
        attention_weights = self.V(score)

        # converts z score to probability
        #(batchm 64, 1)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)

        context_vectors =  attention_weights * hidden_states

        #sum on timestep dimension. shape will be (batch, gru_unit)
        context_vector = tf.reduce_sum(context_vectors, axis=1)

        return context_vector, attention_weights


'''
 Like Neural transatioon
'''
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_unit, attention_unit, fc_unit):
        super(Decoder, self).__init__()
        self.gru_unit = gru_unit
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru  = keras.layers.GRU(gru_unit, return_sequences=True, return_state=True,
                                     recurrent_initializer="glorot_uniform")
        self.attention = Attention(attention_unit)

        self.fc1 = keras.layers.Dense(fc_unit, activation="relu")
        self.fc2 = keras.layers.Dense(vocab_size)

    '''
    Inputs:
      previous word from caption. initialize with <start>: (batch, 1) 
      hidden state from GRU: (batch, gru_unit)
      hidden_states from encoder: (batch, 64, encoder_gru_unit)
    '''
    def __call__(self, prev_word, encoder_hidden_states, prev_decoder_hidden_state):
        # the shape is (batch, 1, embdding_dim)
        embedding = self.embedding(prev_word)

        # shape is (batch, 1 , encoder_gru_unit)
        context_vector, attention_weight = self.attention(prev_decoder_hidden_state, encoder_hidden_states)

        value = tf.concat([tf.expand_dims(context_vector, axis=1) , embedding], axis=-1)

        # x shape: (batch, 1, gru_unit), state shape(batch, gru_unit)
        x, state = self.gru(value)

        # or we can use reshape since we know timestep is 1.
        x = tf.squeeze(x, axis=1)

        x = self.fc1(x)

        logits = self.fc2(x)

        #logits: (batch, vocab_size)
        # state: (batch, gru_unit)
        #attention_weight: (batch, timestep=64, 1)
        return logits,state,  attention_weight

    def zero_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.gru_unit))

embedding_dim = 256
units = 512
#vocab_size = len(tokenizer.word_index) +1
vocab_size = top_k
encoder = Encoder(embedding_dim)
decoder = Decoder(top_k, embedding_dim, units, units, units)
num_of_step_per_epoch = num_examples//batch

optimizer = keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction="none")

'''
for one timestep, 
'''
def calculate_batch_loss(real, pred):
    padding_mask = tf.math.not_equal(real, 0)
    loss = loss_object(real, pred)
    padding_mask = tf.cast(padding_mask, loss.dtype)
    loss = padding_mask*loss

    #sample loss since this is mean
    loss = tf.reduce_mean(loss)
    return loss

'''
train one batch
'''
@tf.function
def one_step_train(batch_input, batch_target, batch_size):

    #initial pre words
    prev_words = tf.reshape( [tokenizer.word_index['<start>']]*batch_size, (batch_size, 1))

    loss = 0
    prev_decoder_hidden_state = decoder.zero_hidden_state(batch_size)


    with tf.GradientTape() as tape:
        encoder_hidden_states = encoder(batch_input)
        '''
         here we don't use hidden state from encoder as initial input to the decoder since 
         encoder is FC(not GRU) and there is NO final hidden state. 
        '''
        for t in range(1, batch_target.shape[1]):
            logits, prev_decoder_hidden_state, attention_weight = decoder(prev_words, encoder_hidden_states, prev_decoder_hidden_state)
            temp_loss = calculate_batch_loss(batch_target[:, t], logits)
            loss =loss + temp_loss

            prev_words = tf.expand_dims(batch_target[:, t], axis=1)

    #loss is a single sample loss over all timestep
    #total loss is the loss for a single timestep
    total_loss = loss / int(batch_target.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss, total_loss, attention_weight


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder= encoder, decoder=decoder)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, 5)

def train():
    EPOCHS = 10

    start_epoch = 0
    if checkpoint_manager.latest_checkpoint:
        start_epoch = int(checkpoint_manager.latest_checkpoint.split("-")[-1]) + 1
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    ds = create_train_dataset(num_examples)

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        epoch_loss = 0
        for batch_index, (batch_feature, batch_caption) in enumerate(ds.take((num_of_step_per_epoch))):
            batch_loss, batch_loss_per_timestep, _ =  one_step_train(batch_feature, batch_caption, batch)
            epoch_loss += batch_loss_per_timestep

            if ( (batch_index+1) % 5 ==0):
                print ("epoch {:d} batch {:d} loss {:05f} ".format(epoch+1, batch_index+1, batch_loss/batch_caption.shape[1]))

        epoch_loss = epoch_loss / num_of_step_per_epoch
        checkpoint_manager.save(epoch)
        print ("epoch {:d} loss {:05f} ".format(epoch+1, epoch_loss))
        print("time taken for this epoch {:05f}".format(time.time() - start))


def prepare_last_batch_fortest():
    size = len(image_infos)
    test_batch = image_infos[size-batch:size]
    pre_compute_and_save_features(test_batch)

def create_test_dataset():
    size = len(image_infos)
    test_batch = image_infos[size-batch:size]
    feature_ds = tf.data.Dataset.from_tensor_slices(test_batch).map(
        lambda info: tf.io.parse_tensor(tf.io.read_file(info[2]), out_type=tf.float32))
    test_captions = captions[size-batch:size]
    caption_ds = tf.data.Dataset.from_tensor_slices(test_captions)
    return tf.data.Dataset.zip( (feature_ds, caption_ds)).batch(batch, drop_remainder=True)


def decode(encoded):
    words = []
    for c in encoded:
        if c ==tokenizer.word_index["<end>"]:
            break
        words.append(tokenizer.index_word[c])
    return ' '.join(words)

'''
evaluate one batch in test 
'''
def evaluate():
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    ds = create_test_dataset()

    for batch_features, batch_captions in ds.take(1):
        pre_words = tf.reshape( [tokenizer.word_index['<start>']]*batch_features.shape[0], (batch_features.shape[0], 1))
        encoder_hidden_states = encoder(batch_features)
        prev_hidden_state = decoder.zero_hidden_state(batch)

        results = pre_words
        for t in range(1, batch_captions.shape[1]):
            logits, prev_hidden_state, attention_weight = decoder(pre_words, encoder_hidden_states, prev_hidden_state)
            predicted_index = tf.argmax(logits, axis=1, output_type=tf.int32)
            predicted_index = tf.reshape(predicted_index, (batch_features.shape[0], 1))
            pre_words = predicted_index
            results= tf.concat ([results, predicted_index], 1)
            if tf.reduce_all(tf.math.equal(predicted_index, 0)):
                break

        for b in range(batch_captions.shape[0]):
            target = batch_captions[b]
            predicted = results[b]
            print("   real   caption : ", decode(target.numpy()))
            print("predicted caption : ", decode(predicted.numpy()))


def test_train_dataset():
    ds = create_train_dataset(num_examples)
    for feature_batch, captions_batch in ds.take(1):
        print(feature_batch.shape)
        print(captions_batch)

def test_encoder():
    ds = create_train_dataset(num_examples)
    for feature_batch, captions_batch in ds.take(1):
        encoded = encoder(feature_batch)
        print(encoded.shape)

def test_attention():
    ds = create_train_dataset(num_examples)
    query = tf.random.normal((batch, units))
    attention = Attention(16)
    for feature_batch, captions_batch in ds.take(1):
        encoded = encoder(feature_batch)
        context_vector, attention_weight = attention(query, encoded)
        print (context_vector.shape)
        print(attention_weight.shape)

def test_decoder():
    ds = create_train_dataset(num_examples)
    query = tf.random.normal((batch, units))
    first_word = tf.expand_dims([tokenizer.word_index["<start>"]]*batch, axis=1)
    for feature_batch, captions_batch in ds.take(1):
        encoded = encoder(feature_batch)
        logits, state, attention_weights = decoder(first_word, encoded, query)
        assert logits.shape[1] ==  top_k
        assert state.shape[1] == units
        print("logits shape ", logits.shape)
        print("loss shape ", state.shape)
        print("attention_weights", attention_weights.shape)

def test_one_batch_train():
    ds = create_train_dataset(num_examples)
    for feature_batch, captions_batch in ds.take(1):
        loss, total_loss, attention_weights = one_step_train(feature_batch, captions_batch, batch)
        print("loss shape ", loss.shape)
        print("total loss shape ", total_loss.shape)
        print("attention weight shape ", attention_weights.shape)


def main(args):
    del args
    if FLAGS.op == 'pre':
        pre_compute_and_save_features(num_examples[:num_examples])
    elif FLAGS.op == 'train':
        train()
    elif FLAGS.op == 'testpre':
        prepare_last_batch_fortest()
    elif FLAGS.op == 'evaluate':
        evaluate()
    else:
        None

if __name__ == '__main__':
    app.run(main)