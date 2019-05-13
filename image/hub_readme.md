#What do we learn from Hub tutorial
+ hub.KeraLayer() can be used to encode input or as classifier depending on the model url
+ image can be processed ing PIL.Image or tf.image library
+ keras.preprocessing.image.ImageDataGenerator can be used to load image
  1. it can do data agumentation on fly(rotate, scale, color)
  1. it can flow image from source(directory) to avoid load all images in memory
  1. it some other properties: num_classes, class_indices, num_examples
+ modelcheckpoint. It uses keras standard. tf.callbacks.ModelCheckpoint.  
  1. to judge whether there is checkpoint or latest checkpoint, it uses tf.train.latest_checkpoint.
  2. model.load_weights. 
 
  The keras's checkpoint is different from pur TF facility (tf.train.CheckpointManager, tf.train.Checkpoint)

  