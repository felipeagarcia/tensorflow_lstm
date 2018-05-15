# -*- coding: utf-8 -*-
'''
Created on May 14 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

'''

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize

#openning data file
with open('../data/X_train.txt', 'r') as data:
    #separating lines
    content = data.readlines() 
#removing whitespaces	
content = [x.split() for x in content] 
content = normalize(content, axis=1, norm='l1')

#openning label file
with open('../data/y_train.txt', 'r') as data:
    #separating lines
    labels = data.readlines() 
#removing whitespaces	
labels = [x.split() for x in labels] 