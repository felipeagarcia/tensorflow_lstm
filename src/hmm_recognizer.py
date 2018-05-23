# -*- coding: utf-8 -*-
"""
Created on May 19 2018

@author: Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/
"""

from HmmClass import HmmScaled as hmm
import data_handler as data
import cmath
import pickle

model_name = "model"
n = 6
m = 150
num_classes = 6
model = []

# for i in range(num_classes):
# 	with open(model_name + str(i), 'rb+') as file:
# 		model.append(pickle.load(file))


for i in range(num_classes):
	model.append(hmm(model_name + str(i), n, m))

for i in range(len(data.content)):
	model[int(data.labels[i][0]) - 1].train(data.angles[i])
count = 0
for i in range(len(data.test_content)):
	probs = []
	for j in range(num_classes):	
		probs.append(model[j].computeProb(data.test_angles[i]))
	recognized = probs.index(min(probs)) + 1
	print("Expected: ", data.test_labels[i][0], " recognized: ", recognized)
	if int(recognized) == int(data.test_labels[i][0]):
		count = count + 1
print("Accuracy: ", float(count)/ float(len(data.test_content)))