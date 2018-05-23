# -*- coding: utf-8 -*-
'''
Created on May 14 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

'''

import numpy as np
from sklearn.preprocessing import normalize
import cmath

#openning data file
with open('../data/X_train.txt', 'r') as data:
    #separating lines
    content = data.readlines() 
#removing whitespaces	
content = [x.split() for x in content]
angles = []
for i in range(len(content)):
	angles.append([])
	for j in range(554,560):
		angles[i].append(float(content[i][j])*180/cmath.pi + 90)
	for j in range(len(angles[i])):
		angles[i][j] = int(angles[i][j])

#openning label file
with open('../data/y_train.txt', 'r') as data:
    #separating lines
    labels = data.readlines() 
#removing whitespaces	
labels = [x.split() for x in labels]

with open('../data/X_test.txt', 'r') as data:
    #separating lines
    test_content = data.readlines() 
#removing whitespaces	
test_content = [x.split() for x in test_content]
test_angles = []
for i in range(len(test_content)):
	test_angles.append([])
	for j in range(554,560):
		test_angles[i].append(float(test_content[i][j])*180/cmath.pi + 90)
	for j in range(len(test_angles[i])):
		test_angles[i][j] = int(test_angles[i][j])

#openning label file
with open('../data/y_test.txt', 'r') as data:
    #separating lines
    test_labels = data.readlines() 
#removing whitespaces	
test_labels = [x.split() for x in test_labels]