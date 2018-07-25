# -*- coding: utf-8 -*-
'''
Created on May 14 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

Open and prepare the data
content: train data
test_content: test data
'''

import numpy as np
from sklearn.preprocessing import normalize
import math

#openning data file
with open('../data/angles_train.txt', 'r') as data:
    #separating lines
    content = data.readlines() 
#removing whitespaces	
content = [list(map(float,x.split())) for x in content]
content = [list(map(int,x)) for x in content]

#openning label file
with open('../data/y_train.txt', 'r') as data:
    #separating lines
    labels = data.readlines() 
#removing whitespaces	
labels = [x.split() for x in labels]

with open('../data/angles_test.txt', 'r') as data:
    #separating lines
    test_content = data.readlines() 
#removing whitespaces	
test_content = [list(map(float,x.split())) for x in test_content]
test_content = [list(map(int,x)) for x in test_content]

#openning label file
with open('../data/y_test.txt', 'r') as data:
    #separating lines
    test_labels = data.readlines() 
#removing whitespaces	
test_labels = [x.split() for x in test_labels]
