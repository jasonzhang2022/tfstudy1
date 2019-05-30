1. Generator model does upsampling use Conv2Dtranspose
2. Both models are called with training=True during training.
3. Two GradientTapes are used. One for discriminator and one for generator
4. Discriminator are called twice in one training step and results from both
   calls are used to calculate loss.
 
