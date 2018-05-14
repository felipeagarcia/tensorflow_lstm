# -*- coding: utf-8 -*-
'''
Created on May 14 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

'''

import tensorflow as tf
import numpy as np
import pickle
hello = tf.constant('Hello, TensorFlow!')

# when you run sess, you should see a bunch of lines with the word gpu in them (if install worked)
# otherwise, not running on gpu
sess = tf.Session()
print(sess.run(hello))