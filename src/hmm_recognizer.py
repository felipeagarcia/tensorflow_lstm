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
    with open('modelo' + str(i), 'rb') as file:
        model.append(pickle.load(file))
# for i in range(num_classes):
#     model.append(hmm('modelo' + str(i), n, m))
current_label = int(data.labels[0][0])
for i in range(len(data.content)):
    if(current_label == int(data.labels[i][0])):
        if(len(m_data) == 0):
            m_data = (data.content[i])
        else:
            m_data = np.concatenate([m_data, data.content[i]])
    else:
        model[current_label - 1].train(m_data)
        current_label = int(data.labels[i][0])
        m_data = (data.content[i])

count = 0
train_count = 0
current_label = int(data.test_labels[0][0])
for i in range(len(data.test_content)):
    if(current_label == int(data.test_labels[i][0])):
        if(len(m_data) == 0):
            m_data = (data.test_content[i])
        else:
            m_data = np.concatenate([m_data, (data.test_content[i])])
    else:
        probs = []
        for j in range(num_classes):
            probs.append(model[j].compute_prob(data.test_content[i]))
        predicted = np.array(probs).argmin() + 1
        print('expected:', int(data.test_labels[i][0]),
              'predicted:', predicted)
        if(predicted == int(data.test_labels[i][0])):
            count += 1
        current_label = int(data.test_labels[i][0])
        m_data = (data.test_content[i])
        train_count += 1

print("precision:", float(count)/len(data.test_content), "count:", count,
      "len:", len(data.test_content))
