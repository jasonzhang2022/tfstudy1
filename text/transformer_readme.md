# Why transformer
1. parallel execution. Traditional RNN model is series.  There are for loop over the sequences. Matrix 
   multiplication can't be used. 
2. it can learn long-range dependences.
3. no assumption on temporal/spatial relationship. So directional GRU is not needed.
##preprocess
Text preprocess is done using tfds.features.text.SubwordTextEncoder. It is not clear how punctuation is handled.
tfds gives tf.data.Dataset so all the preprocess has to be done in dataset flow. Operations include map, filter, 
cache, padded_batch, prefetch. Pay attention to **padded_patch**. The padding is done at batch level. So each 
batch has its own unique seq length. This is different from keras.preprocessing.sequence.pad_sequences which 
pads all input once. So all inputs have the same sequence length. 

The model doesn't assume batch size or seq length size. So it uses tf.shape to retrieve tensor shape dynamically. 
tensor.shape only gives shape statically. If the static shape is not available, the shape will be (?). 

##Mask
* **padding_mask**. When encoder does self_attention or decoder does attention on encoder's output, attention
  is only needed for  real words. A mask (value = 1) is created for padding word for input batch. 
  After we have attention score, attention is modified: when mask =1, the score is reduced by negative infinity
  **attention_score \*= mask\*-1e9**. -1e9 is close to negative infinity. The end result is that attention_score 
  for padding word will be close to zero.
* **look ahead mask**. When decoder does self attention on its input(target input), it only needs to pay 
  attention to word before itself.  New word is predicted from encoder's output and predicted words before 
  current one . New word will not depend on words after it since those words don't exist yet.  This is the 
  reason behind look ahead mask.  Decoder actually uses a combination of look ahead mask and padding_mask. 
* Mask is only used to calculate attention weights. It only needs to pay weights to meaningful words, not word
  added for computation convenience. 
## position encoding

## Scaled dot production attention
[reference](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms)

dot product attention is used: dot(query, key)/sqrt(d_model). This is different from Neural Machine translation
model which uses additive attentions score = W([query + key])

## Multi head attention  
1. Query, Key, Value go through Fully connected layer first. 
2. Instead of use one big attention, The vectors are divided into num_heads spaces. The division and merge at late time
   are matrix operation (reshape, transpose). Instead of 3-D matrix, 4-D matrix is used in attention. 
##Encoder, Decoder, Transformer
1. Encoder layer has one self-attention. To produce one output at position i, attention checks all values in its input
2. Decoder layer hs two attentions: 
   + self attention (with look ahead mask) as it is in encoder
   + It also takes final output from encoder. It queries the output for attention.
3. dropout, residual, BatchNormalization are used in every layer.

## Train Management.
1. It uses CheckpointManager to manage checkpoint so the train can be interrupted and restarted.
2. it used tf.train.Metrics to aaccumulate loss and accuracy. By this approach, we don't need to calculate
  metrics manually.
3. It uses tf.train.metrics.SparseCategoricalAccuracy to calcuate accuracy for seq-2-seq model.

## Train and evaluation
1. Train is done like Dense model. Just one batch call on transformer. There is no for loop as it is in RNN model.
2. For loop is used in evaluation and prediction.  Each for loop, take predicted sentences so far and predict a 
   new word.
 