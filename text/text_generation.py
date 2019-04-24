#https://www.tensorflow.org/alpha/tutorials/text/text_generation
from __future__ import absolute_import, unicode_literals, print_function, division

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

'''
flow: 
1. collect all characters.
2. language model: 30 character as input. 
3. generate: always pick up the character of greatest possibility.  
'''

'''step 1:
'''
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
