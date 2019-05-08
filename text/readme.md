# Neural Translation vs Image Caption
## text pre-process
1. Neural translation has character normalization since it contains spanish.  Image caption only has english word
   so no character normalization
1. Neural translation treats sentence ending punctuation as token. All other tokens are replaced with space. Image 
   caption filters out all punctuations.
1. tokenization: Neural translation considers vocabulary word. Image captions only considers top_k word.

Commonality: 
+ Don't need to consider all character(word). Only consider top_k word. Treat punctuation basing on purpose.

## Encoder
1. Neural translation uses embedding and GRU for input processes. Image caption uses a Fully-Connected layer.
   As a result, FC layer in image caption doesn't have a final hidden state while Neural translation has 
   **final hidden state** from GRU.  
   Why we don't use GRU for image caption?
   GRU is used to capture temporal relations in input data. The encoded feature for encoded image has no 
   temporal relation. 
   Image on the other hand has spatial relation. But when it comes to encoded feature layer, the relation 
   feature may already get lost. 

## Attention
These two models has the exact the same attention layer. 
The attention uses sum attention. 
 
 
## Decoder
1. Neural translation uses encoder's **final hidden state** as decoder's initial hidden state. This hidden 
   state is used as query to calculate attention weights for first output. 
   Image caption doesn't have this state. So it uses zero as initial hidden state.
1. Decoder in image caption has a n extra Fully Connected layer.
  
## checkpoint management
1.  Both uses checkpoints which captures optimizer and optimizer.
2. Image capture uses CheckpointManager to save checkpoint so it can delete old checkpoint and only maintain latest 5 
   checkpoints. 
 
## Other thoughts
1. Image caption use keras.application to encode image. 

## Image caption special
+ Image caption has separate flow to pre-process image. An important lesson learned that using tf.XXX if 
  possible once in dataset. Image preprocess uses tf.io.read_file(write_file), tf.io.serialize_tensor 
  (parse tensor), and tf.image.decode_jpeg(tf.image.resize)
  
  
## TODO
+ refresh knowledge about different kinds of attention
+ How is gru hidden state is passed down?
  in Neural translation, Two GRU are used. 
  + Encoder's GRU is called like this gru(batch_input, initial_state=zeros). Batch input has 
    the shape (batch_size, timestep, features). I guess gru will automatically pass the hidden state from
    one time step to next time step.  For next batch, it will use initial_state again since we didn't 
    specify stateful=True
  + Decoders' GRU is called like this gru(single_timestep_input). First the single_timestep_input has 
    the shape (batch_size, 1, features). There is only features for one timestep.  Seond there is no
    initial_state. So what initial_state is used? (zeros?) How is hidden state passed from one call to 
    next call? 
  

