from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import time
import numpy as np
import matplotlib.pyplot as plt
import os


'''
step 1: text preparation

This example uses dataset from TFDS. 
We will 
1. encode. Text to integer
2. filter sentences longer than 40
3. cache, shuffle, padded_batch, prefetch


'''
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_ds = examples['train']
validation_ds = examples['validation']
en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
[en.numpy()  for (pt, en) in train_ds] , target_vocab_size=2**13)

pt_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    [pt.numpy()  for (pt, en) in train_ds] , target_vocab_size=2**13)


BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

def encode(pt_sentence, en_sentence):
    en_encoded = [en_tokenizer.vocab_size]+ en_tokenizer.encode(en_sentence.numpy()) + [en_tokenizer.vocab_size +1]
    pt_encoded = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(pt_sentence.numpy()) + [pt_tokenizer.vocab_size +1]
    return pt_encoded, en_encoded

def max_length_filter(pt_encoded, en_encoded, max_length=MAX_LENGTH):
    return tf.logical_and (tf.size(pt_encoded) <= max_length, tf.size(en_encoded) <= max_length)

def tf_encode(pt_sentence, en_sentence):
    return tf.py_function(encode, [pt_sentence, en_sentence], [tf.int64, tf.int64])

def create_train_ds():
    ds= train_ds.map(tf_encode).filter(max_length_filter).cache()
    ds = ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]), drop_remainder=True)
    ds= ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def create_validation_ds():
    return train_ds.map(tf_encode).filter(max_length_filter) \
        .padded_patch(BATCH_SIZE, padded_shapes=None)

train_dataset = create_train_ds()


'''
Step 2
prepare small components
'''
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    #zero position is turned to 1. Other value is turned to 1.
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    #shape(batch, 1, 1, seq_len
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    for self-attentipn
        seq_len_q == seq_len_k == seq_len_v
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    #shape (..., seq_len_q, seq_len_k)
    #Each row is a result for each query. Each element in row is a alignment score for a key
    matmul_tk = tf.matmul(q, k, transpose_b=True)

    depth = tf.cast( tf.shape(k)[-1], tf.float32)

    scaled_matmul_tk = matmul_tk/tf.sqrt(depth)

    # is mask is 1, mask * -1e9 will be a genative infinity.
    # the sum will be very small negative number.
    if mask is not None:
        scaled_matmul_tk += (mask *-1e9)

    #turn the score to probability weight
    attention_weights = tf.nn.softmax(scaled_matmul_tk, axis=-1) #(..., seq_len_q, seq_len_k)

    # matrix multiplication is magic: weight and sum is done in one calculation.
    # TODO is this (..., seq_len_q, depth) or (..., seq_len_v, depth)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

    return output, attention_weights

'''
* Linear layers and split into heads. 
* Scaled dot-product attention. 
* Concatenation of heads. 
* Final linear layer.

What this layer does:
Give query, key, value, produce an output. To do this.
1. pass Q, K, V through FC.
2. Split the result into num_heades part.
  Each part goes through attention and have output from V.
3. merge the different part of V into one.
4. pass the another FC layer.
'''
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.WQ = keras.layers.Dense(d_model)
        self.Wk = keras.layers.Dense(d_model)
        self.WV = keras.layers.Dense(d_model)

        self.fc = keras.layers.Dense(d_model)

    '''
    (..., seq, d_model)->(..., num_heads, seq, depth)
    '''
    def split_heads(self, v, batch_size):
        temp = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        temp = tf.transpose(temp, perm=[0, 2, 1, 3])
        return temp

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        wq = self.WQ(q) #(..., seq_len_q, d_model)
        wk = self.Wk(k)  #(..., seq_len_k, d_model)
        wv = self.WV(v) #(..., seq_len_v, d_model)

        wq = self.split_heads(wq, batch_size)  #(..., num_heads, seq_len_q, depth)
        wk = self.split_heads(wk, batch_size)
        wv = self.split_heads(wv, batch_size)

        #(..., num_heads, seq_len_q, depth)
        output, attention_weights = scaled_dot_product_attention(wq, wk, wv, mask)

        #(..., seq_len_q, d_model)
        #(..., num_heads, seq_len_q, depth) ->  #(..., seq_len_q, num_heads,depth)
        output = tf.transpose(output, perm=[0, 2,1,3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.fc(output)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, fc_units, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(d_model, num_heads)

        #TODO there is no LayerNormalization layer
        self.batch_normal1 = keras.layers.BatchNormalization(epsilon=1e-6)
        self.batch_normal2 = keras.layers.BatchNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.point_wise_ff = point_wise_feed_forward_network(d_model, fc_units)

    def call(self, x, training, mask):
        #attention_output: (batch, seq_len, d_model)
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        normalized = self.batch_normal1( x+attention_output)

        ff_output  = self.point_wise_ff(normalized)
        ff_output = self.dropout2(ff_output, training=training)

        normalized2 = self.batch_normal2(normalized + ff_output)

        return normalized2
class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = keras.layers.Dropout(rate)
        self.norm1 = keras.layers.BatchNormalization(epsilon=1e-6)

        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = keras.layers.Dropout(rate)
        self.norm2 = keras.layers.BatchNormalization(epsilon=1e-6)

        self.ff = point_wise_feed_forward_network(d_model, dff)
        self.dropout3 = keras.layers.Dropout(rate)
        self.norm3 = keras.layers.BatchNormalization(epsilon=1e-6)

    def call(self, x, encoder_states, training, look_ahead_mask, padding_mask):

        # mask should be forward mask
        # for current inut, we only pay attention to character before current current charater
        # this is because during translation, we can't predict nets character unless current
        #current character is predicted.
        attention_output, attention_weights1 = self.self_attention(x, x,x, look_ahead_mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.norm1(attention_output + x)

        # then we ask encoder's state to decide current character.
        # here we should use padding mask to ignore padding character in input.
        output_with_encode_state, attention_weights2 = self.decoder_attention(encoder_states, encoder_states, attention_output, padding_mask )
        output_with_encode_state = self.dropout2(output_with_encode_state, training= training)
        output_with_encode_state = self.norm2(output_with_encode_state + attention_output)

        ff = self.ff(output_with_encode_state)
        ff = self.dropout3(ff, training=training)
        ff= self.norm3(ff+ output_with_encode_state)

        return ff, attention_weights1, attention_weights2

class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ff_units, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, d_model)

        self.enc_layers = [ EncoderLayer(d_model, num_heads, ff_units, rate) for _ in range(num_layers) ]

        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        #TODO why this
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x
class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ff_units, output_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(output_vocab_size, d_model)
        self.pos_encoding = positional_encoding(output_vocab_size, d_model)

        self.dec_layers = [ DecoderLayer(d_model, num_heads, ff_units, rate) for _ in range(num_layers) ]

        self.dropout = keras.layers.Dropout(rate)
    def call(self, y, enc_states, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(y)[1]

        attention_weights = {}

        y = self.embedding(y)
        y *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        y += self.pos_encoding[:, :seq_len, :]

        y = self.dropout(y, training = training)

        for i in range(self.num_layers):
            y, self_attention_weights, encoder_attention_weights = self.dec_layers[i](y, enc_states, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_self_attention'.format(i+1)]= self_attention_weights
            attention_weights['decoder_layer{}_encoder_attention_weights'.format(i+1)] = encoder_attention_weights

        return y, attention_weights

class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, fc_unit, input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, fc_unit, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, fc_unit, target_vocab_size, rate)

        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, input_batch, target_batch, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_batch, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(target_batch, enc_output, training, look_ahead_mask, dec_padding_mask)

        logits = self.final_layer(dec_output)

        return logits, attention_weights


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = pt_tokenizer.vocab_size + 2
target_vocab_size = en_tokenizer.vocab_size + 2
dropout_rate = 0.1

transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                          fc_unit=dff,
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                          rate=dropout_rate)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
def loss_function(real, pred):
    real_value_mask = tf.math.not_equal(real, 0)
    _loss = loss_object(real, pred)
    real_value_mask = tf.cast(real_value_mask, _loss.dtype)
    _loss *=real_value_mask

    return tf.reduce_mean(_loss)


train_loss = tf.metrics.Mean(name="train_loss")
train_accuracy = tf.metrics.SparseCategoricalAccuracy(name="train_accuracy")


def create_mask(input, target):

    # The mask will be used in attention in Encoder.
    # It will reduce the attention for padding to negligble
    enc_padding_mask = create_padding_mask(input)

    # This mask will be used in attention in Decoder.
    # It will be used to find the attention for encoded_output
    dec_padding_mask = create_padding_mask(input)

    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    target_padding_mask = create_padding_mask(target)

    #ignore character down stream or any padding character upstream
    combined_mask = tf.math.maximum(look_ahead_mask, target_padding_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_dir = "transformer_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint = tf.train.Checkpoint(optimizer = optimizer, transformer=transformer)
chkpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
initial_epoch = 0
if chkpt_manager.latest_checkpoint:
    checkpoint.restore(chkpt_manager.latest_checkpoint)
    initial_epoch = int(chkpt_manager.latest_checkpoint.split("-")[-1])+1

@tf.function
def one_step_train(input, target):
    target_input = target[:, :-1]
    target_prediction = target[:, 1:]



    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(input, target_input)

    with tf.GradientTape() as tape:
        logits, _ = transformer(input, target_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(target_prediction, logits)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(target_prediction, logits)


def train():
    EPOCHS = 20

    for epoch in range(initial_epoch, EPOCHS):
        start = time.time()
        for step, (input_batch, target_batch) in enumerate(train_dataset):
            one_step_train(input_batch, target_batch)
            if step%100 == 0:
                print ("epoch {}, batch {} loss {:.4f}  accuracy: {:.4f}".format(epoch, step, train_loss.result(), train_accuracy.result()))

        chkpt_manager.save()
        print ("epoch {}, loss {:.4f}  accuracy: {:.4f}".format(epoch,  train_loss.result(), train_accuracy.result()))
        print (" time taken: {:.4f}".format(time.time() - start))


def evaluate(inp_sentence):
    encoded = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size +1]
    encoded = tf.convert_to_tensor(encoded)
    batched = encoded[tf.newaxis, :]


    #no mask is needed since the sentence is not masked.
    #no look_ahead mask is needed since 1)future word is not available, 2) there is no padding
    enc_output = transformer.encoder(batched, False, None)

    #shape (1,1)
    output = tf.convert_to_tensor([ [en_tokenizer.vocab_size]])
    print(output.shape)
    for i in range(MAX_LENGTH):
        dec_output, attention_weights = transformer.decoder(output, enc_output, False, None, None)
        logits = transformer.final_layer(dec_output)

        #extrat last word predicted. All proceeding words are discarded since they are already in output.
        predicted = logits[:, -1, :]
        predicted_index = tf.argmax(predicted, axis=-1, output_type=tf.int32)

        if tf.math.equal(predicted_index, en_tokenizer.vocab_size+1):
            break

        predicted_index=predicted_index[tf.newaxis, :]
        output = tf.concat([output, predicted_index], axis=-1)

    print(output.shape)
    return tf.squeeze(output, axis=0)

def translate(sentence, plot=''):
    result = evaluate(sentence)

    predicted_sentence = en_tokenizer.decode([i for i in result
                                              if i < en_tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

def evaluate_test():
    translate("este é um problema que temos que resolver.")
    print ("Real translation: this is a problem we have to solve .")

    translate("os meus vizinhos ouviram sobre esta ideia.")
    print ("Real translation: and my neighboring homes heard about this idea .")

    translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
    print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")


#train()
evaluate_test()


#---------------------------- test code

def test_scaled_dot_product_attention():
    q = tf.random.normal((1,3))
    k = tf.random.normal((2,3))
    v = tf.random.normal( (2,3))

    output, _ = scaled_dot_product_attention(q, k, v, None)
    print(output)
def test_multi_attention_layer():
    mha = MultiHeadAttention(512, 8)
    q = tf.random.normal((16, 2, 3))
    k = tf.random.normal((16, 2, 3))
    v = tf.random.normal( (16, 2, 3))
    output = mha(v, k, q, None)
    print(output.shape)

    mha = MultiHeadAttention(512, 8)
    q = tf.random.normal((16, 1, 3))
    k = tf.random.normal((16, 2, 3))
    v = tf.random.normal( (16, 2, 3))
    output = mha(v, k, q, None)
    print(output.shape)

    mha = MultiHeadAttention(512, 8)
    x = tf.random.uniform((64, 43, 512))
    output = mha(x, x, x, None)
    print(output.shape)

def test_encoder_layer():
    encoder_layer = EncoderLayer(512, 8, 2048)
    x = tf.random.uniform((64, 43, 512))
    output = encoder_layer(x, False, None)
    print(output.shape)

def test_decoder_layer():
    encoder_layer = EncoderLayer(512, 8, 2048)
    x = tf.random.uniform((64, 43, 512))
    encoder_output = encoder_layer(x, False, None)
    decoder_layer = DecoderLayer(512, 8, 2048)
    y = tf.random.uniform( (64, 50, 512))
    decoder_output, _, _ = decoder_layer(y, encoder_output, False, None, None)
    print(decoder_output.shape)

def test_encoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             ff_units=2048, input_vocab_size=8500)
    x = tf.random.uniform((64, 62))
    sample_encoder_output = sample_encoder(x, training=False, mask=None)
    print(sample_encoder_output.shape)

def test_decoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             ff_units=2048, input_vocab_size=8500)
    x = tf.random.uniform((64, 62))
    sample_encoder_output = sample_encoder(x, training=False, mask=None)
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             ff_units=2048, output_vocab_size=8000)
    y = tf.random.uniform((64, 26))
    decoder_output, _ = sample_decoder(y, sample_encoder_output, False, None, None)
    print(decoder_output.shape)


def test_transformer():
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, fc_unit=2048,
        input_vocab_size=8500, target_vocab_size=8000)

    input_batch = tf.random.uniform ((64, 62))
    target_batch = tf.random.uniform ((64, 26))

    fn_output, _ = sample_transformer(input_batch, target_batch, training=False,
                                   enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None)
    print(fn_output.shape)



