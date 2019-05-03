from __future__ import division, absolute_import, print_function
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import io
import os
import unicodedata
import re
import time
from absl import app
from absl import flags
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_bool("train", True, "Train if true. Generate if false")

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

#total 118964  lines
num_examples = 30000

chkpt_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(chkpt_dir, "chkpt")
if not os.path.exists(chkpt_dir):
    os.mkdir(chkpt_dir)

#-----------------------------------
'''
Step 1: create tf.data.Dataset and make it ready for training.
  The input has many lines.  Each line has input and target separated by \t.
  We need 
   a) clean character, make important puctuation as word. We have many assumption. For example, we replace rare character with space. 
   b) tokenizer sentences into word, then integer, at the same time, keep word_to_index dictionary. 
     Padding. The padding is for patch purpose. 
   c) convert tokenized text to t.data.Dataset.
     ¿
'''

# character level process
def preprocess_sentence(sentence):

    #unicode character normalization
    s = ''.join([c for c in unicodedata.normalize( "NFD", sentence.lower().strip()) if unicodedata.category(c)!='Mn'])

    #insert one space BEFORE and AFTER special character.
    s = re.sub(r"([?.!,¿])", r' \1 ', s)

    # collpase multiple (space, quote) into one.
    s = re.sub(r'[" "]]+', " ", s)

    #replace rare character with space
    s = re.sub(r'[^a-zA-Z?!.,¿]', " ", s)

    #strip
    s= s.rstrip().strip()

    s= "<start> " + s + " <end>"

    return s

def tokenize(sentences):
    tokenizer = keras.preprocessing.text.Tokenizer(filters='')

    tokenizer.fit_on_texts(sentences)

    seqs = tokenizer.texts_to_sequences(sentences)

    seqs= keras.preprocessing.sequence.pad_sequences(seqs, padding="post")

    return seqs, tokenizer


def load_data():
    lines = io.open(path_to_file, encoding="utf-8").read().split("\n")
    paired = [ [ preprocess_sentence(sentence) for sentence in line.split("\t")] for line in  lines[:num_examples]]

    inputs, targets= zip(*paired)

    input_seqs, input_tokenizer = tokenize(inputs)
    target_seqs, target_tokenizer = tokenize(targets)

    #!!!!!reverse target and input. Make spanish as input, and english as target.
    #input_seqs.shape= (30000, 16)
    # target_seqs.shape= (30000, 11)
    return target_seqs, target_tokenizer, input_seqs, input_tokenizer,

def max_length(seqs):
    return max( len(t) for t in seqs)

'''
input.lang, output.lang have 
word_index dictionary, 
and
index_word dictionary. 

zero is used for padding. But zero is not either dictionary

'''
input_seqs, input_lang, target_seqs, target_lang = load_data()
input_train_seqs, input_test_seqs, target_train_seqs, target_test_seqs = train_test_split(input_seqs, target_seqs, test_size=0.2)
print("input train shape ", input_train_seqs.shape)
print("target train shape ", target_train_seqs.shape)

max_input_len = max_length(input_seqs)
max_target_len = max_length(target_seqs)
assert max_input_len == input_seqs.shape[1]
assert max_target_len == target_seqs.shape[1]

#these vocab size is used for embedding. Input sequence has padding character (0) which has need an embedding entry.
# That is why we have +1 here.
input_vocab_size = len(input_lang.word_index) + 1
target_vocab_size = len(target_lang.word_index) + 1


#some global variables
batch_size = 64
gru_unit = 1024
embedding_dim = 256
step_per_epoch = input_train_seqs.shape[0]//batch_size
print("step_per_epoch", step_per_epoch)

def create_tf_train_dataset():
    return tf.data.Dataset.from_tensor_slices( (input_train_seqs, target_train_seqs))\
        .shuffle(buffer_size=input_train_seqs.shape[0]).batch(batch_size, drop_remainder=True)

def create_tf_eval_dataset():
    return tf.data.Dataset.from_tensor_slices( (input_test_seqs, target_test_seqs))\
        .shuffle(buffer_size=input_train_seqs.shape[0]).batch(batch_size, drop_remainder=True)

def seq_to_sentence(seq, lang):
    return ' '.join( lang.index_word[t] for t in seq if t!=0)

'''
Step 2 model creation
'''

'''
Give input, produce 
1) GRU hidden state: one state for each character
2) one enc_state: used as initial hidden state for decoder 
'''
class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_units, batch_size):
        super(Encoder, self).__init__()
        self.gru_units= gru_units
        self.batch_size = batch_size
        self.emedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(gru_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    '''
    batch_input shape: (batch, timestep). No feature dimension since each timestep only has one value.
    gru_initial_state: (batch, gru_unit). Not (batch, 1, gru_unit)
    '''
    def __call__(self, batch_input, gru_initial_state):

        #x shape is (batch, timeseries, embedding_dim)
        # Embedding layers only do embedding for -1 dimension.
        x= self.emedding(batch_input)
        # x shape is (batch, timeseries, gru_units). One (gru_units) for each timestep.
        # state  is the final encoded state. it is (batch, gru_units)
        x, state = self.gru(x, initial_state = gru_initial_state)

        return x, state

    def initialize_gru_state(self):
        return tf.zeros((self.batch_size, self.gru_units))

'''
Given prev decoder hidden state(query), and encoder's GRU's final hidden state
    Since we use encoder's final hidden state and decoder's initial state, 
    both GRUs must have the same unit's value (or we could use a Dense to translate dimension)
calculate attention weight,
output: context vector.
'''
class Attention(keras.Model):

    def __init__(self, attention_unit):
        super(Attention, self).__init__()
        self.attention_unit = attention_unit

        # W1 and W2 have the same attention_unit
        # so we can + result.
        self.W1 = keras.layers.Dense(attention_unit)
        self.W2 = keras.layers.Dense(attention_unit)
        self.V = keras.layers.Dense(1)

    '''
    query: decoder previous hidden state. Initial value is encoder's final state.
       shape: (batch, decoder's gru_unit). Not (batch, 1, decoder's gru_unit)
    encoder hidden states (batch, timestep, gru_units)
    '''
    def __call__(self, query, encoder_hidden_states ):

        # shape: (batch, decoder's GRU_UNIT)->(batch, 1, decoder's GRU_UNIT)
        query = tf.expand_dims(query, 1)

        #------- Dense layer only works on last dimension.
        #should be [batch, timestep, attention_unit]
        w1 = self.W1(encoder_hidden_states)

        #should be [batch, 1, attention_unit]
        w2 = self.W2(query)

        #w1 and w2 are compatible since W1 and W2 has the same unit.
        # w2 is broadcasted
        #should be [batch, timestep, attention_unit]
        w3 = w1 + w2

        #should be [batch, timestep, attention_unit]
        squashed = tf.nn.tanh(w3)

        #squash last dimention from attention_unit to 1.
        #should be [batch, timestep, 1]
        score = self.V(squashed)

        # shape : (batch, timestep,1)
        attention_weight = tf.nn.softmax(score, axis=1)

        # attention_weight is broadcasted ( batch, timestep, 1->encoder's gru unit)
        # Element-wise multiplication.
        context_vector = attention_weight * encoder_hidden_states

        #add all timestep together. remove timestep.
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weight

'''
 Outputs one character.
 it needs 3 inputs.
  1. previous decoder's GRU hidden state.
  2. previous target character (language model, context character, then next character)
  3. context vector
 output: 
   GRU hidden state, used for next character
   prediciton (logit). Used to compute loss

'''
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_unit, batch_size,attention_unit):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(gru_unit, return_sequences=True, return_state=True,
                                    recurrent_initializer="glorot_uniform")
        self.attention = Attention(attention_unit)
        self.fc = keras.layers.Dense(vocab_size)

    '''
    c: previous character. initialized as <start>
    pre_hidden_state: previous hidden state from GRU unit. Initialized with value from encoder final state.
    encoder_hidden_states: encoder GRU hidden states. Used to compute context vector.
    
    shapes:
    c: (batch, 1)
    pre_hidden_state: (batch, decoder_gru_unit)
    encoder_hidden_states: (batch, timestep, encoder_gru_unit)
    '''
    def __call__(self, c, prev_hidden_state, encoder_hidden_states):

        #shape: (batch, 1, embedding_dim)
        embedded = self.embedding(c)

        #contex_vector shape: [batch_size, encoder's GRU unit)
        context_vector, attention_weights = self.attention(prev_hidden_state, encoder_hidden_states)
        context_vector = tf.expand_dims(context_vector, axis=1)
        #shape: (batch, 1,  embedding_dim + encoders' GRU unit)
        x = tf.concat([embedded, context_vector], axis=-1)

        x, state = self.gru(x, initial_state=prev_hidden_state)

        # x has timestep. But the timestep here is 1.
        x = tf.squeeze(x, axis=1)
        logits = self.fc(x)
        return logits, state, attention_weights


optimizer = keras.optimizers.Adam()
#NOTE: need to understand parameters for SparsecategoricalCrossentropy.
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

#model
encoder = Encoder(input_vocab_size, embedding_dim, gru_unit, batch_size)

#Attention unit doesn't have to been gru _unit. It is gru_unit to keep the structure simple.
decoder = Decoder(target_vocab_size, embedding_dim, gru_unit, batch_size, gru_unit)

# everywhere, when 'loss' is mentioned, it is loss per sample.
def calculate_loss(real, pred):
    not_padding_mask = tf.math.logical_not(tf.math.equal(real, 0))

    #shape (64,). One loss for each sample.
    loss_ = loss_object(real, pred)

    not_padding_mask = tf.cast(not_padding_mask, dtype=loss_.dtype)

    # if not padding, keep the value, otherwise, use 0 as loss_value
    loss_ *= not_padding_mask

    #average loss for all samples
    # loss per samples.
    loss_ = tf.reduce_mean(loss_)

    return loss_


'''
train one batch step.
'''
@tf.function
def train_one_step(input_batch, target_batch, encoder_initial_state):
    loss = 0
    with tf.GradientTape() as  tape:
        encoder_hidden_states, encoder_state = encoder(input_batch, encoder_initial_state)

        decoder_prev_state= encoder_state

        # First character <start> for target batch
        # We could use target_batch[:, 0] since the first character is always <start>
        #[0]*64. The [0] is repeated 64 time.
        decoder_prev_character = tf.reshape( [target_lang.word_index['<start>']]*batch_size, shape=(batch_size, 1))

        # t start from 1, zero character is already extracted.
        for t in range(1, max_target_len):
            logits, decoder_prev_state, attention_weight = decoder(decoder_prev_character, decoder_prev_state, encoder_hidden_states)

            loss += calculate_loss(target_batch[:, t], logits)

            #shape: (batch_size,)
            decoder_prev_character = target_batch[:, t]

            #increase rank by one. shape is (batch, 1)
            #as gru expect timestep in input
            decoder_prev_character = tf.expand_dims(decoder_prev_character, 1)

        #average loss. Should we consider padding factor?
    batch_loss = loss / target_batch.shape[1]

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients( zip(gradients, variables))

    return batch_loss


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder = encoder, decoder= decoder)


'''
go through many epochs
output batch loss every 100 batches
output total loss every epoch
save checkpoint every two epochs.
save time.
'''
def train(EPOCH):

    for epoch in range(EPOCH):
        encoder_initial_state = encoder.initialize_gru_state()
        total_loss = 0

        start_time = time.time()
        for batch_num, (input_batch, target_batch) in enumerate(create_tf_train_dataset().take(step_per_epoch)):
            batch_loss = train_one_step(input_batch, target_batch, encoder_initial_state)
            total_loss += batch_loss

            if batch_num%100 == 0 :
                print( "epoch {} batch {} loss {:.4f} ".format(epoch+1, batch_num, batch_loss.numpy()))

        checkpoint.save(checkpoint_prefix)

        print("epoch {} average batch loss {:.4f} ".format(epoch+1, total_loss/step_per_epoch))
        print(" time taken for one epoch {}".format(time.time() -start_time))




'''
 translate input_sentence to target_sentence
'''
def evaluate(input_sentence):
    processed = preprocess_sentence(input_sentence)
    inputs = input_lang.texts_to_sequences([processed])

    #shape: [batch=1, timestep]
    padded = keras.preprocessing.sequence.pad_sequences(inputs, max_input_len, padding="post")

    # converted numpy or nested list to tensor
    padded = tf.convert_to_tensor(padded)

    # this line doesn't work since it uses cached batch parameter
    # encoder_initial_state = encoder.initialize_gru_state()
    encoder_initial_state = tf.zeros((1, gru_unit))
    encoder_hidden_states, encoder_final_state = encoder(padded, encoder_initial_state)

    #shape(1,1)
    prev_target_char = tf.convert_to_tensor([[ target_lang.word_index['<start>']]])

    end_target_char = target_lang.word_index['<end>']

    decoder_prev_hidden_state = encoder_final_state

    result = ''
    attentions = []

    print(prev_target_char)
    print(end_target_char)

    while prev_target_char[0][0].numpy() != end_target_char:
        logits, decoder_prev_hidden_state, attention_weight = decoder(prev_target_char, decoder_prev_hidden_state, encoder_hidden_states)

        attention_weight = tf.reshape(attention_weight, (attention_weight.shape[1],))
        attentions.append(attention_weight.numpy())
        #convert logits to a character prediction
        #logits shape should be (batch=1, target_vocab_size)
        predicted_char = tf.argmax(logits, -1)
        result += target_lang.index_word[predicted_char[0].numpy()]+" "
        prev_target_char = tf.expand_dims(predicted_char, 0)

    return result, processed, attentions

'''
attentions: 
   Each row is for one result character. Show what input character this result character pays attention to 
'''
def plot_attention(attentions, result_tokens, input_token):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1,1,1)
    ax.matshow(attentions, cmap="viridis")

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + input_token, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + result_tokens, fontdict=fontdict)

    plt.show()

def translate():
    input_sentence = u'hace mucho frio aqui.'

    #input sentence is right now processed
    result, input_sentence, attentions = evaluate(input_sentence)
    print("input: ", input_sentence)
    print("result: ", result)
    input_tokens = input_sentence.split(" ")
    result_tokens = result.split(" ")
    # we need to convert this to numpy array so we can use [:, :] slice.
    # List can only use [:] slice. List has no dimension concept.
    attentions = np.array(attentions)
    attentions = attentions[:len(result_tokens), :len(input_tokens)]
    print(attentions.shape)
    plot_attention(attentions, result_tokens, input_tokens)

def main(args):
    del args
    if FLAGS.train:
        train(10)
    else:
        checkpoint.restore(tf.train.latest_checkpoint(chkpt_dir))
        translate()

if __name__ == '__main__':
    app.run(main)


#------------------------
def test_preprocess():
    en_sentence = u"May I borrow this book?"
    sp_sentence = u"¿Puedo tomar prestado este libro?"
    print(preprocess_sentence(en_sentence))
    print(preprocess_sentence(sp_sentence))

def seq_to_text(seq, lang):
    #skip 0 since 0 is padding
    return ' '.join([ lang.index_word[c] for c in seq if c!=0])


def test_encoder():
    encoder = Encoder(input_vocab_size, embedding_dim, gru_unit, batch_size)
    input = tf.random.uniform ((batch_size, max_input_len, ), minval=1, maxval=input_vocab_size-1, dtype=tf.int32)
    return encoder(input, encoder.initialize_gru_state())

def test_attention():
    encoder_hidden_states, encoder_final_state = test_encoder()
    attention = Attention(10)
    attention(encoder_final_state, encoder_hidden_states)

def test_decoder():
    target = tf.random.uniform ((batch_size, 1), minval=1, maxval = target_vocab_size -1, dtype=tf.int32)
    print("target shape ", target.shape)
    encoder_hidden_states, encoder_final_state = test_encoder()
    decoder = Decoder(target_vocab_size, embedding_dim, gru_unit, batch_size, 10)
    logits, state, _ = decoder(target, encoder_final_state, encoder_hidden_states)

def test_train_one_step():
    input_batch, target_batch = next(iter(create_tf_train_dataset()))
    initial_state = encoder.initialize_gru_state()
    train_one_step(input_batch, target_batch, initial_state)
