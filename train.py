# -*- coding: utf-8 -*-

from neuralNetwork import NeuralNetwork
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import pickle

x = []
y = []
# wordList = {"wo": "我", "ni": "你", "ren": "人"}
# [1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]
wordList = {"wo": "我",  "ren": "人", "ni": "你"}
path = r"./train"
files = os.listdir(path)
for f in files:
    d = np.loadtxt(path + "/" + f)
    x.append(d)
    y_name = f.split("_")[0]
    y.append(wordList[y_name])
# print np.array(x), np.array(y)
x_data = np.array(x)
y_data = np.array(y)
l = LabelBinarizer()
y_data = l.fit_transform(y_data)
result = l.classes_
pickle.dump(result, open('result.pkl', 'wb'))
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
# labels_train = LabelBinarizer().fit_transform(y_train)
# labels_test = LabelBinarizer().fit_transform(y_test)

# print labels_test

nn = NeuralNetwork([960, 1500, 3], "logistic")
print "start"
nn.fit(x_data, y_data, epochs=1000)
pickle.dump(nn, open('nn.pkl', 'wb'))
predictions = []
for i in range(x_data.shape[0]):
    o = nn.predict(x_data[i])
    d = result[np.argmax(o)]
    predictions.append(d)

for i in predictions:
    print i
