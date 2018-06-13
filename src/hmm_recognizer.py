# -*- coding: utf-8 -*-
"""
Created on May 19 2018

@author: Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/
"""
from hmmlearn import hmm
import data_handler as data
from sklearn.externals import joblib
import numpy as np

num_classes = 6
model = []
scores = []
for i in range(20):
	model.append( hmm.GMMHMM(n_components = num_classes, verbose = False, n_iter = 1000, tol = 0.00001))
	model[i].fit(data.content)
	scores.append(model[i].score(data.test_content))
arg_model = np.array(scores).argmin()
_, predictions = model[arg_model].score_samples(data.test_content)
print(predictions)
predicted = [a.argmin() + 1 for a in predictions]
print(predicted)
print(data.test_labels)
count = 0
for i in range(len(predicted)):
	if(predicted[i] == int(data.test_labels[i][0])):
		count += 1
print('precision:', float(count)/len(predicted), 'count:', count, 'len:', len(predicted))
joblib.dump(model[arg_model], "model.pkl")