# -*- coding: utf-8 -*-
"""
Created on May 19 2018

@author: Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/
"""
from HmmClass import hmm
import data_handler as data
import numpy as np
import pickle

num_classes = 6
n = 6
m = 180
model = []
m_data = []
for i in range(num_classes):
	with open('a' + str(i), 'rb') as file:
		model.append(pickle.load(file))
	m_data.append([])
for i in range(len(data.content)):
	# model[int(data.labels[i][0]) - 1].train(data.content[i])
	# print('***********')
	if (len(m_data[ int(data.labels[i][0]) - 1 ]) == 0):
		m_data[ int(data.labels[i][0]) - 1 ] = (data.content[i])
	else:
		m_data[ int(data.labels[i][0]) - 1 ] = np.concatenate(
			[m_data[ int(data.labels[i][0]) - 1 ], (data.content[i])] )

print(m_data)
for i in range(6):
	model[i].train(m_data[i])

count = 0
for i in range(len(data.test_content)):
	probs = []
	for j in range(6):
		probs.append( model[j].compute_prob(data.test_content[i]))
	predicted = np.array(probs).argmin() + 1
	print('expected:', int(data.test_labels[i][0]), 'predicted:', predicted)
	if(predicted == int(data.test_labels[i][0])):
		count += 1
print("precision:", float(count)/len(data.test_content), "count:", count, "len:", len(data.test_content))