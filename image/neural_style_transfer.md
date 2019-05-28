#Tensorflow library learned
+ tf.linalg.einsum
+ tf.image.resize, tf.image.decode_image

#Knowledge
+ Use dot product to calculate correlation
+ Gram matrix

#Neural model
VGG19 model is used. However, the whole model is not trainable.
 We only use the model to extract information about the 
image.  Then we use customize loss function: content_loss(feature to feature MSE) 
and style loss.  Optimizer is used to optimize the loss

There is no Deep Neural network. We only use VGG to extract image representation.


***TODO***
1. why the generated image is blurred? The load_image is different from tutorial.


